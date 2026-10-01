"""The CLI's XPU path: backend choice, quantization refusals and option refusals that run before any GPU work."""

import argparse
import json
from types import SimpleNamespace

import pytest

from tensorfold import accel, cli, families, serve_options
from tensorfold.families import _require_xpu_w4a16, require_readable


def _family(**members):
    return SimpleNamespace(title="Test family", model_type="test", package=SimpleNamespace(**members))


def _engine(*a, **k):
    return None


def _auto_round(**over):
    block = {"quant_method": "auto-round", "packing_format": "auto_round:auto_gptq", "bits": 4, "group_size": 128,
             "sym": True, "data_type": "int"}
    return {"model_type": "qwen3_5", "quantization_config": {**block, **over}}


def _gptq(**over):
    block = {"quant_method": "gptq", "bits": 4, "group_size": 64, "sym": True, "desc_act": False,
             "checkpoint_format": "gptq"}
    return {"model_type": "nemotron_h", "quantization_config": {**block, **over}}


def _ct(weights=None, **over):
    w = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128, "actorder": None}
    group = {"weights": {**w, **(weights or {})}, "input_activations": None, "targets": ["Linear"]}
    block = {"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups": {"group_0": group}}
    return {"model_type": "qwen3_5", "quantization_config": {**block, **over}}


def _qwen():
    return families.families()["qwen3_5"]


def test_serve_parses_the_xpu_backend():
    args = cli.build_parser().parse_args(["serve", "owner/model", "--backend", "xpu"])
    assert args.backend == "xpu"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "owner/model", "--backend", "sycl"])


def test_auto_backend_picks_the_device_the_machine_has(monkeypatch):
    every = _family(load=_engine, cuda_engine=_engine, xpu_engine=_engine)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(accel, "device_type", lambda *a: "xpu")
    assert cli._backend("auto", every) == "xpu"
    monkeypatch.setattr(accel, "device_type", lambda *a: "cuda")
    assert cli._backend("auto", every) == "cuda"
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(accel, "device_type", lambda *a: pytest.fail("macOS must not ask torch"))
    assert cli._backend("auto", every) == "mlx"


def test_xpu_needs_a_family_with_an_xpu_engine():
    with pytest.raises(ValueError, match="no XPU engine yet"):
        cli._backend("xpu", _family(load=_engine, cuda_engine=_engine))
    assert cli._backend("xpu", _family(xpu_engine=_engine)) == "xpu"


@pytest.mark.parametrize("kind", ["qwen3_5", "nemotron_h"])
def test_serving_on_xpu_is_refused_before_any_download(tmp_path, capsys, kind):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": kind}))
    code = cli.main(["serve", str(tmp_path), "--backend", "xpu", "--no-update-check"])
    assert code == 1
    assert "no XPU engine yet" in capsys.readouterr().err


def test_a_family_with_an_xpu_engine_is_served_on_xpu(tmp_path, monkeypatch, capsys):
    import tensorfold.cuda.server as server
    from tensorfold import hub

    made, served = [], []
    family = _family(xpu_engine=lambda *a, **k: made.append(k) or SimpleNamespace(max_len=8192))
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)
    monkeypatch.setattr(server, "App", lambda *a, **k: served.append(k) or SimpleNamespace(effective_context_window=64))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "xpu", "--no-update-check", "--no-drafts"]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    out = capsys.readouterr().out
    assert made and "on XPU" in out and "on CUDA" not in out and "bf16 activations" in out


def test_cuda_logs_still_say_cuda(tmp_path, monkeypatch, capsys):
    import tensorfold.cuda.server as server

    family = _family(cuda_engine=lambda *a, **k: SimpleNamespace(max_len=8192))
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=64))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    assert cli._serve_cuda(args, family, tmp_path, 128) == 0
    assert "on CUDA" in capsys.readouterr().out


def test_backends_of_lists_xpu_only_for_a_family_with_an_engine(monkeypatch):
    for family in families.families().values():                    # real families gain xpu_engine in WS5
        assert ("xpu" in families.backends_of(family)) == hasattr(family.package, "xpu_engine")
    fake = families.Family("t", "T", "types", False)
    monkeypatch.setattr(families.Family, "package", property(lambda self: SimpleNamespace(xpu_engine=_engine)))
    assert families.backends_of(fake) == ("xpu",)


def test_xpu_reads_the_recipe_checkpoints_headers():
    qwen, nemotron = _qwen(), families.families()["nemotron_h"]
    require_readable(qwen, _auto_round(), "xpu")
    require_readable(nemotron, _auto_round(group_size=64), "xpu")
    require_readable(nemotron, _gptq(), "xpu")
    require_readable(qwen, _gptq(group_size=128, quant_method="gptq"), "xpu")
    require_readable(qwen, _ct(), "xpu")
    require_readable(qwen, _auto_round(packing_format="auto_gptq"), "xpu")
    require_readable(qwen, _auto_round(extra_config={"lm_head": {"bits": 16}, "x": {"bits": 4, "sym": True}}), "xpu")


@pytest.mark.parametrize("config, message", [
    ({"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4}}, "MLX 4-bit"),
    ({"model_type": "qwen3_5", "quantization": {"mode": "mxfp4", "bits": 4}}, "mlx-mxfp4"),
    ({"model_type": "qwen3_5", "quantization_config": {"quant_method": "exl3"}}, "exl3"),
    ({"model_type": "qwen3_5", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}},
     "NVFP4"),
    ({"model_type": "qwen3_5"}, "unquantized"),
    (_ct({"type": "float", "num_bits": 8}, format="float-quantized"), "float"),
    (_ct({"type": "float", "num_bits": 4}), "float"),
    (_ct({"symmetric": False}), "symmetric"),
    (_ct({"actorder": "group"}), "act-order"),
    (_ct({"num_bits": 8}), "4-bit"),
    (_ct({"group_size": 32}), "group_size 64 or 128"),
    (_ct({"strategy": "channel", "group_size": None}), "group strategy"),
    (_ct(format="int-quantized"), "pack-quantized"),
    (_auto_round(sym=False), "symmetric"),
    (_auto_round(desc_act=True), "act-order"),
    (_auto_round(bits=8), "4-bit"),
    (_auto_round(bits=2), "4-bit"),
    (_auto_round(group_size=32), "group_size 64 or 128"),
    (_auto_round(packing_format="auto_awq"), "auto_round:auto_gptq"),
    (_auto_round(extra_config={"mlp": {"bits": 8}}), "mixed bits"),
    (_auto_round(extra_config={"mlp": {"bits": 4, "sym": False}}), "mixed bits"),
    (_gptq(checkpoint_format="gptq_v2"), "gptq_v2"),
    (_gptq(sym=False), "symmetric"),
    (_gptq(desc_act=True), "act-order"),
    (_gptq(bits=3), "4-bit"),
])
def test_xpu_refuses_what_it_does_not_read(config, message):
    with pytest.raises(ValueError, match=message) as err:
        require_readable(_qwen(), config, "xpu")
    assert "Intel GPUs (XPU)" in str(err.value) and "Tested checkpoints" in str(err.value)


def test_a_compressed_tensors_input_quantization_is_refused():
    config = _ct()
    config["quantization_config"]["config_groups"]["group_0"]["input_activations"] = {"num_bits": 8, "type": "int"}
    with pytest.raises(ValueError, match="W4A16"):
        require_readable(_qwen(), config, "xpu")


def test_xpu_refusal_names_the_accepted_formats():
    with pytest.raises(ValueError, match="symmetric INT4 weight-only checkpoints"):
        require_readable(_qwen(), {"model_type": "qwen3_5", "quantization": {"bits": 4}}, "xpu")
    with pytest.raises(ValueError, match="this checkpoint has"):
        _require_xpu_w4a16(_gptq(bits=8), "Intel GPUs (XPU)", "none")


def test_cuda_and_mlx_readability_is_unchanged():
    assert families.readable_quants(_family(), "mlx") == ("mlx", None)
    assert families.readable_quants(_family(), "cuda") == ("mlx",)
    assert families.readable_quants(_family(), "xpu") == families.XPU_QUANTS
    fam = SimpleNamespace(title="T", package=SimpleNamespace(QUANT_METHODS={"xpu": ("auto-round", None)}))
    assert families.readable_quants(fam, "xpu") == ("auto-round", None)


def _args(**over):
    base = dict(kv_dtype="bf16", tp=1, decode_share=None, prefill_fp8=None, checkpoint_slots=None,
                mtp_confidence=None, parallel="auto")
    return argparse.Namespace(**{**base, **over})


@pytest.mark.parametrize("over, message", [
    ({"prefill_fp8": True}, "FP8 hardware"),
    ({"kv_dtype": "int8"}, "--kv-dtype int8 is not served on XPU yet"),
    ({"kv_dtype": "int4"}, "--kv-dtype int4 is not served on XPU yet"),
    ({"tp": 2}, "--tp 2 is not served on XPU yet"),
    ({"decode_share": 0.5}, "--decode-share is not served on XPU yet"),
])
def test_xpu_refuses_options_it_does_not_serve(over, message):
    with pytest.raises(ValueError, match=message):
        serve_options.check(_args(**over), _family(xpu_engine=_engine), "xpu")


def test_xpu_accepts_the_defaults_and_explicit_no_fp8():
    serve_options.check(_args(), _family(xpu_engine=_engine), "xpu")
    serve_options.check(_args(prefill_fp8=False, kv_dtype="bf16"), _family(xpu_engine=_engine), "xpu")


def test_xpu_mtp_confidence_looks_up_the_xpu_engine():
    def with_rule(mtp_confidence=None):
        return None

    serve_options.check(_args(mtp_confidence=0.5), _family(xpu_engine=with_rule), "xpu")
    with pytest.raises(ValueError, match="on XPU has no such rule"):
        serve_options.check(_args(mtp_confidence=0.5), _family(xpu_engine=_engine), "xpu")


def test_the_same_flags_on_cuda_and_mlx_behave_as_before():
    cuda = _family(cuda_engine=_engine)
    serve_options.check(_args(kv_dtype="bf16", tp=2), cuda, "cuda")
    with pytest.raises(ValueError, match="CUDA serves a bf16 KV cache, not --kv-dtype int8"):
        serve_options.check(_args(kv_dtype="int8"), cuda, "cuda")
    with pytest.raises(ValueError, match="is a CUDA engine option"):
        serve_options.check(_args(kv_dtype="int8"), cuda, "mlx")
    with pytest.raises(ValueError, match="picks FP8 prompt kernels on CUDA; Test family on CUDA has none"):
        serve_options.check(_args(prefill_fp8=True), cuda, "cuda")
    with pytest.raises(ValueError, match="Test family on MLX has none"):
        serve_options.check(_args(prefill_fp8=True), cuda, "mlx")
    with pytest.raises(ValueError, match="--decode-share sets the Mac server's share"):
        serve_options.check(_args(decode_share=0.5), cuda, "cuda")
    with pytest.raises(ValueError, match="Test family on CUDA has no such rule"):
        serve_options.check(_args(mtp_confidence=0.5), cuda, "cuda")
