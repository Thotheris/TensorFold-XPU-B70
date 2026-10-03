"""Bounded S0 probes record Triton capabilities and known hazards on the XPU."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

__all__ = ["run_probes"]


def _s64(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _kernels(triton, tl):
    @triton.jit
    def add(X, Y, Z, B: tl.constexpr):
        i = tl.arange(0, B)
        tl.store(Z + i, tl.load(X + i) + tl.load(Y + i))

    @triton.jit
    def indirect(TABLE, OUT, B: tl.constexpr):
        i = tl.arange(0, B)
        ptr = tl.load(TABLE).to(tl.pointer_type(tl.int32))
        tl.store(OUT + i, tl.load(ptr + i))

    @triton.jit
    def mix(X, Y, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X + i, i < N, other=0).to(tl.uint64)
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        tl.store(Y + i, x ^ (x >> 31), i < N)

    @triton.jit
    def transcend(X, E, L, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X + i, i < N, other=1.0)
        tl.store(E + i, tl.exp(x), i < N)
        tl.store(L + i, tl.log(x), i < N)

    @triton.jit
    def casts(X, BF, FP8, N: tl.constexpr, B: tl.constexpr):
        i = tl.arange(0, B)
        x = tl.load(X + i, i < N, other=0.0)
        tl.store(BF + i, x.to(tl.bfloat16).to(tl.float32), i < N)
        tl.store(FP8 + i, x.to(tl.float8e4nv).to(tl.float32), i < N)

    @triton.jit
    def commit(P, BASE, OUT, R: tl.constexpr, CD: tl.constexpr, BC: tl.constexpr, INPLACE: tl.constexpr):
        ch = tl.program_id(0) * BC + tl.arange(0, BC)
        j = tl.arange(0, 4)
        src = R - 3 + j
        keep = (j < 3)[:, None] & (ch < CD)[None, :]
        new = tl.load(P + tl.maximum(src, 0)[:, None] * CD + ch[None, :],
                      keep & (src >= 0)[:, None], other=0.0)
        old = tl.load(BASE + tl.where(src < 0, R + j, 0)[:, None] * CD + ch[None, :],
                      keep & (src < 0)[:, None], other=0.0)
        rows = tl.where((src >= 0)[:, None], new, old)
        if INPLACE:
            tl.debug_barrier()
        tl.store(OUT + j[:, None] * CD + ch[None, :], rows, keep)

    @triton.jit
    def shrink_expand(X, A, B, OUT, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
                      NPID_FACTOR: tl.constexpr):
        pid = tl.program_id(0)
        rows = (pid // NPID_FACTOR) * BM + tl.arange(0, BM)
        k = tl.arange(0, 64)
        rank = tl.arange(0, 16)
        x = tl.load(X + rows[:, None] * 64 + k[None, :], rows[:, None] < M, other=0.0)
        a = tl.load(A + k[:, None] * 16 + rank[None, :])
        tmp = tl.dot(x, a).to(tl.bfloat16)
        width = tl.cdiv(N, NPID_FACTOR)
        lo = (pid % NPID_FACTOR) * width
        hi = tl.minimum(lo + width, N)
        for start in range(lo, hi, 32):
            cols = start + tl.arange(0, 32)
            b = tl.load(B + rank[:, None] * N + cols[None, :], cols[None, :] < hi, other=0.0)
            acc = tl.dot(tmp, b)
            tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
                     (rows[:, None] < M) & (cols[None, :] < hi))

    return add, indirect, mix, transcend, casts, commit, shrink_expand


def _numeric(values) -> list:
    """Non-finite diagnostic values retain their meaning in strict JSON."""
    import math

    return [float(x) if math.isfinite(float(x)) else str(float(x)) for x in values]


def run_probes(out_dir: Path) -> dict:
    """Mandatory controls gate the ladder; hazard observations retain their actual outcomes."""
    from .kbench import triton_kernel_stats

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "triton-smoke.json"
    result = {"ok": False, "torch": None, "triton": None,
              "probes": {}, "stopped_at": None}

    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def check(name, fn):
        try:
            probe = fn()
        except Exception as exc:  # noqa: BLE001 - compilation failures must leave a results artifact.
            probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["probes"][name] = probe
        if not probe["ok"]:
            result["stopped_at"] = name
        save()
        return probe["ok"]

    try:
        import numpy as np
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        result["probes"]["imports"] = {"ok": False, "error": str(exc)}
        result["stopped_at"] = "imports"
        save()
        return result
    result.update(torch=str(torch.__version__), triton=str(triton.__version__))

    if not check("available", lambda: {"ok": bool(torch.xpu.is_available())}):
        return result
    dev = torch.device("xpu", 0)
    names = ("has_fp64", "sub_group_sizes", "has_subgroup_matrix_multiply_accumulate",
             "has_subgroup_2d_block_io", "max_work_group_size")
    properties = {}

    def device_properties():
        props = torch.xpu.get_device_properties(dev)
        properties.update({name: getattr(props, name, None) for name in names})
        return {"ok": all(properties[name] is True for name in names[2:4]), **properties}

    if not check("properties", device_properties):
        return result
    add, indirect, mix, transcend, casts, commit, shrink_expand = _kernels(triton, tl)

    def vector_add():
        x = torch.arange(128, device=dev, dtype=torch.int32)
        y = torch.full_like(x, 7)
        out = torch.empty_like(x)
        kernel = add[(1,)](x, y, out, B=128, num_warps=4, enable_fp_fusion=False)
        torch.xpu.synchronize()
        return {"ok": out.cpu().equal(x.cpu() + 7), **triton_kernel_stats(kernel)}

    def pointer():
        # The source remains resident through Python ownership, but is absent from launcher arguments.
        source = torch.arange(128, dtype=torch.int32, device=dev)
        address = source.data_ptr()
        table = torch.tensor([_s64(address)], dtype=torch.int64, device=dev)
        out = torch.empty_like(source)
        kernel = indirect[(1,)](table, out, B=128, num_warps=4)
        torch.xpu.synchronize()
        return {"ok": source.cpu().equal(out.cpu()), "address": address, "high_bit": address >= 1 << 63,
                "signed_table_value": _s64(address), "source_passed_to_launcher": False,
                **triton_kernel_stats(kernel)}

    def uint64_mix():
        rng = np.random.default_rng(70)
        bits = rng.integers(0, 1 << 64, size=4096, dtype=np.uint64)
        bits[:6] = [0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 2, (1 << 64) - 1]
        expected = bits.copy()
        with np.errstate(over="ignore"):
            expected = (expected ^ (expected >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            expected = (expected ^ (expected >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            expected ^= expected >> np.uint64(31)
        x = torch.from_numpy(bits.view(np.int64)).to(dev)
        out = torch.empty_like(x)
        kernel = mix[(16,)](x, out, N=4096, B=256, num_warps=4)
        torch.xpu.synchronize()
        actual = out.cpu().numpy().view(np.uint64)
        return {"ok": bool(np.array_equal(actual, expected)), "elements": len(bits),
                "mismatches": int(np.count_nonzero(actual != expected)), **triton_kernel_stats(kernel)}

    def fp64():
        if not properties["has_fp64"]:
            return {"ok": False, "error": "device does not report fp64 support"}
        values = np.geomspace(2 ** -53, 16, 100_000, dtype=np.float64)
        x = torch.from_numpy(values).to(dev)
        exp = torch.empty_like(x)
        log = torch.empty_like(x)
        grid = (triton.cdiv(len(values), 256),)

        def launch():
            return transcend[grid](x, exp, log, N=len(values), B=256, num_warps=4, enable_fp_fusion=False)

        kernel = launch()
        torch.xpu.synchronize()
        expected_exp, expected_log = np.exp(values), np.log(values)
        actual_exp, actual_log = exp.cpu().numpy(), log.cpu().numpy()
        rel_exp = float(np.max(np.abs(actual_exp - expected_exp) / expected_exp))
        abs_log = float(np.max(np.abs(actual_log - expected_log)))
        times = []
        for _ in range(10):
            start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            torch.xpu.synchronize()
            times.append(float(start.elapsed_time(end)) * 1000)
        # This threshold detects a float32 downgrade; it is not a sampler exactness certificate.
        return {"ok": rel_exp <= 1e-11 and abs_log <= 1e-11, "elements": len(values),
                "max_exp_relative_error": rel_exp, "max_log_absolute_error": abs_log,
                "median_us": statistics.median(times), "repeats": len(times),
                "accuracy_threshold": 1e-11, **triton_kernel_stats(kernel)}

    def rounding():
        values = np.array([0, -0.0, 1, 1.00390625, 1.01171875, 2 ** -149, 2 ** -126,
                           2 ** -9, 2 ** -10, 448, 449, 464, 480, -480, np.inf, -np.inf, np.nan], np.float32)
        x = torch.from_numpy(values).to(dev)
        bf, fp8 = torch.empty_like(x), torch.empty_like(x)
        kernel = casts[(1,)](x, bf, fp8, N=len(values), B=32, num_warps=4)
        torch.xpu.synchronize()
        # CPU torch's bf16 conversion is the rounding oracle, including special-value categories.
        expected = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
        actual = bf.cpu().numpy()
        nan_mask = np.isnan(expected)
        bf_ok = np.array_equal(actual[~nan_mask].view(np.uint32), expected[~nan_mask].view(np.uint32))
        bf_ok = bool(bf_ok and np.all(np.isnan(actual[nan_mask])))
        return {"ok": bf_ok, "bf16_ok": bf_ok, "input": _numeric(values), "bf16": _numeric(actual),
                "fp8_e4nv": _numeric(fp8.cpu().numpy()), "fp8_status": "range observation; serving refused",
                **triton_kernel_stats(kernel)}

    def barrier():
        cases = []
        controls_ok = True
        for rows in (1, 2):
            old = torch.arange(3 * 257, dtype=torch.float32).reshape(3, 257)
            proj = (torch.arange(rows * 257, dtype=torch.float32).reshape(rows, 257) + 4096).to(dev)
            expected = torch.cat((old, proj.cpu()))[-3:]
            control = torch.empty_like(old, device=dev)
            source = old.to(dev)
            safe = commit[(3,)](proj, source, control, R=rows, CD=257, BC=128, INPLACE=False, num_warps=4)
            torch.xpu.synchronize()
            control_ok = expected.equal(control.cpu())
            controls_ok &= control_ok
            mismatches = 0
            kernel = None
            for _ in range(20):
                source = old.to(dev)
                kernel = commit[(3,)](proj, source, source, R=rows, CD=257, BC=128, INPLACE=True, num_warps=4)
                torch.xpu.synchronize()
                mismatches += int(not expected.equal(source.cpu()))
            cases.append({"rows": rows, "control_ok": control_ok, "hazard_ok": mismatches == 0,
                          "hazard_mismatch_launches": mismatches, "repeats": 20,
                          "control_kernel": triton_kernel_stats(safe), "hazard_kernel": triton_kernel_stats(kernel)})
        return {"ok": bool(controls_ok), "hazard_observed": any(not case["hazard_ok"] for case in cases),
                "scope": "disjoint channel blocks; in-place same-workgroup read/write; no cross-workgroup fence",
                "production_requirement": "use separate destination; observations do not establish fence semantics",
                "cases": cases}

    def block_m():
        # A bounded reduced two-dot/N-loop check, not the full vLLM MoE/LoRA issue reproducer.
        m, n = 33, 192
        rng = np.random.default_rng(8121)
        x_cpu = rng.integers(-1, 2, (m, 64)).astype(np.float32)
        a_cpu = rng.integers(-1, 2, (64, 16)).astype(np.float32)
        b_cpu = rng.integers(-1, 2, (16, n)).astype(np.float32)
        x = torch.from_numpy(x_cpu).to(dev, dtype=torch.bfloat16)
        a = torch.from_numpy(a_cpu).to(dev, dtype=torch.bfloat16)
        b = torch.from_numpy(b_cpu).to(dev, dtype=torch.bfloat16)
        # These small integer reductions and intermediate bf16 casts are exact.
        expected = torch.from_numpy((x_cpu @ a_cpu) @ b_cpu)
        cases = []
        for bm in (16, 32, 64):
            for factor in (1, 3):
                out = torch.full((m, n), -1.0, device=dev)
                kernel = shrink_expand[(triton.cdiv(m, bm) * factor,)](
                    x, a, b, out, M=m, N=n, BM=bm, NPID_FACTOR=factor, num_warps=4, enable_fp_fusion=False)
                torch.xpu.synchronize()
                actual = out.cpu()
                cases.append({"block_m": bm, "npid_factor": factor, "ok": actual.equal(expected),
                              "mismatches": int((actual != expected).sum()), **triton_kernel_stats(kernel)})
        safe_ok = all(case["ok"] for case in cases if case["block_m"] >= 32 or case["npid_factor"] == 1)
        return {"ok": safe_ok, "hazard_observed": any(not case["ok"] for case in cases), "cases": cases,
                "scope": "reduced shrink/expand with runtime N-axis loop; does not certify full MoE-LoRA kernel",
                "issue": "https://github.com/intel/intel-xpu-backend-for-triton/issues/8121"}

    def launch_latency():
        """Per-launch cost of a trivial kernel: host submit time and device time over 20 queued launches; never gates."""
        import time

        x = torch.arange(128, device=dev, dtype=torch.int32)
        y = torch.full_like(x, 7)
        out = torch.empty_like(x)
        launches = {
            "triton_add": lambda: add[(1,)](x, y, out, B=128, num_warps=4, enable_fp_fusion=False),
            "torch_add": lambda: torch.add(x, y, out=out),
        }
        report = {"ok": True, "batch": 20, "samples": 10}
        for name, launch in launches.items():
            launch()
            torch.xpu.synchronize()
            host, device = [], []
            for _ in range(10):
                start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
                t0 = time.perf_counter()
                start.record()
                for _ in range(20):
                    launch()
                end.record()
                host.append((time.perf_counter() - t0) * 1e6 / 20)
                torch.xpu.synchronize()
                device.append(float(start.elapsed_time(end)) * 1e3 / 20)
            report[name] = {"host_submit_us": statistics.median(host), "device_us": statistics.median(device)}
        return report

    for name, fn in (("vector_add", vector_add), ("pointer_round_trip", pointer), ("uint64_mix", uint64_mix),
                     ("fp64_exp_log", fp64), ("cast_ranges", rounding), ("debug_barrier", barrier),
                     ("block_m16", block_m), ("launch_latency", launch_latency)):
        if not check(name, fn):
            return result
    result["ok"] = True
    save()
    return result
