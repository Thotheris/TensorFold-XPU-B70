"""The shared GDN kernels: tree nodes, chains and commit replays run the serial step, bit for bit."""

import random

import pytest
import torch

from tests.devices import DEV as DEVICE
from tests.devices import device_available

if not device_available():
    pytest.skip("needs CUDA or XPU", allow_module_level=True)

from tensorfold.cuda.kernels import gdn  # noqa: E402

pytestmark = pytest.mark.xpu_kernel("gdn")
cuda_only = pytest.mark.cuda_only           # the nvcc extension and its pointer tables
xpu_only = pytest.mark.skipif(DEVICE != "xpu", reason="the XPU Triton GDN")


def _inputs(nodes: int, kh: int = 2, hv: int = 6, dv: int = 8, keys=torch.bfloat16, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(41 + nodes + seed)
    q = (torch.randn((nodes, kh, 128), generator=gen, device="cuda") * 0.01).to(keys)
    k = (torch.randn((nodes, kh, 128), generator=gen, device="cuda") * 0.01).to(keys)
    v = torch.randn((nodes, hv, dv), generator=gen, device="cuda").bfloat16()
    g = torch.rand((nodes, hv), generator=gen, device="cuda") * 0.4 + 0.5
    beta = torch.rand((nodes, hv), generator=gen, device="cuda")
    state = torch.randn((hv, dv, 128), generator=gen, device="cuda") * 0.05
    return q, k, v, g, beta, state


def _step(args, row, state):
    """A serial step: the node alone as a one-row window, then its one-row commit."""

    one = [x[row:row + 1].contiguous() for x in args[:5]]
    y = gdn.tree(*one, gdn.plan([[-1]], "cuda"), state=state)
    return y[0], _replay(one, [[0]], [state])[0]


def _replay(args, paths, states):
    """Each stream's accepted window rows replayed from its state (one layer)."""

    q, k, v, g, beta = args[:5]
    width = max(1, max(len(p) for p in paths))
    table = gdn.to_device(gdn.replay_table([k], [v], [g], [beta], [[s] for s in states]), torch.int64, "cuda")
    rows = torch.tensor([p + [0] * (width - len(p)) for p in paths], dtype=torch.int32, device="cuda")
    counts = torch.tensor([len(p) for p in paths], dtype=torch.int32, device="cuda")
    return gdn.replay(table, 1, len(paths), rows, counts, k, v)[:, 0]


def _path(parents, node):
    path = []
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


def _check_tree(parents, args, nodes_to_check=None):
    out = gdn.tree(*args[:5], gdn.plan([parents], "cuda"), state=args[5])
    for node in nodes_to_check or range(len(parents)):
        state = args[5]
        for row in _path(parents, node):
            expected, state = _step(args, row, state)
        assert torch.equal(out[node], expected), f"node {node} differs from its serial steps"
        assert torch.equal(_replay(args, [_path(parents, node)], [args[5]])[0], state), f"replay to {node} differs"


SHAPES = {
    "binary": lambda n: [-1] + [(i - 1) // 2 for i in range(1, n)],
    "chain": lambda n: list(range(-1, n - 1)),
    "spine": lambda n: list(range(-1, min(n, 16) - 1)) + [i % min(n, 16) for i in range(n - min(n, 16))],
    "deep": lambda n: [-1] + list(range(min(n, 32) - 1)) + [min(n, 32) - 2] * (n - min(n, 32)),
}


@pytest.mark.parametrize("nodes", [1, 5, 16, 17, 32, 33, 128])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@cuda_only
def test_tree_nodes_match_serial_paths(shape, nodes):
    parents = SHAPES[shape](nodes)
    check = sorted({0, nodes // 3, nodes // 2, nodes - 1} | set(range(min(nodes, 6))))
    _check_tree(parents, _inputs(nodes), check)


@cuda_only
def test_random_trees_and_fp32_keys():
    rng = random.Random(3)
    for trial in range(6):
        n = rng.randint(2, 128)
        parents = [-1] + [rng.randint(max(0, i - 6), i - 1) for i in range(1, n)]
        args = _inputs(n, keys=torch.float32 if trial % 2 else torch.bfloat16, seed=trial)
        _check_tree(parents, args, [n - 1, n // 2, rng.randrange(n)])


@cuda_only
def test_checkpoint_head_layout():
    """The 27B's heads (16 key, 48 value, 128 wide): every value row of every head."""

    args = _inputs(6, kh=16, hv=48, dv=128)
    _check_tree([-1, 0, 0, 1, 3, 2], args)


@cuda_only
def test_several_streams_in_one_launch():
    """Each stream's rows equal its own one-stream launch; replays of every stream in one launch too."""

    trees = [SHAPES["spine"](12), SHAPES["chain"](7), [-1], SHAPES["binary"](16)]
    total = sum(len(t) for t in trees)
    args = _inputs(total)
    states = [args[5] * (1.0 + 0.1 * s) for s in range(len(trees))]
    table = gdn.to_device(gdn.pointers(states), torch.int64, "cuda")
    together = gdn.tree(*args[:5], gdn.plan(trees, "cuda"), table=table)
    base, paths = 0, []
    for s, parents in enumerate(trees):
        rows = [x[base:base + len(parents)].contiguous() for x in args[:5]]
        alone = gdn.tree(*rows, gdn.plan([parents], "cuda"), state=states[s])
        assert torch.equal(together[base:base + len(parents)], alone), f"stream {s}"
        paths.append([base + r for r in _path(parents, len(parents) - 1)])
        base += len(parents)
    replayed = _replay(args, paths, states)
    for s, path in enumerate(paths):
        assert torch.equal(replayed[s], _replay(args, [path], [states[s]])[0]), f"stream {s} replay"
    # rows and counts as strided views of one packed copy: [rows padded to W, count] per stream
    flat = torch.tensor([x for p in paths for x in p + [0] * (total - len(p)) + [len(p)]], dtype=torch.int32,
                        device="cuda").view(len(paths), total + 1)
    q, k, v, g, beta = args[:5]
    table = gdn.to_device(gdn.replay_table([k], [v], [g], [beta], [[s] for s in states]), torch.int64, "cuda")
    packed = gdn.replay(table, 1, len(paths), flat[:, :total], flat[:, total], k, v)[:, 0]
    assert torch.equal(packed, replayed)


@cuda_only
def test_forty_streams_of_small_trees():
    """32+ streams in one launch (grid z past 32), each up to 16 rows: every stream equals its own launch."""

    rng = random.Random(21)
    trees = []
    for _ in range(40):
        n = rng.randint(1, 16)
        trees.append([-1] + [rng.randint(0, i - 1) for i in range(1, n)])
    total = sum(len(t) for t in trees)
    args = _inputs(total, kh=16, hv=48, dv=128, seed=4)
    states = [args[5] * (1.0 - 0.01 * s) for s in range(len(trees))]
    together = gdn.tree(*args[:5], gdn.plan(trees, "cuda"), table=gdn.to_device(gdn.pointers(states), torch.int64,
                                                                                 "cuda"))
    base = 0
    for s, parents in enumerate(trees):
        rows = [x[base:base + len(parents)].contiguous() for x in args[:5]]
        alone = gdn.tree(*rows, gdn.plan([parents], "cuda"), state=states[s])
        assert torch.equal(together[base:base + len(parents)], alone), f"stream {s}"
        base += len(parents)


@cuda_only
def test_replay_of_many_layers_and_zero_rows():
    layers = [_inputs(12, seed=i) for i in range(5)]
    path = [0, 1, 3, 7]
    k, v, g, b = ([a[j] for a in layers] for j in (1, 2, 3, 4))
    table = gdn.to_device(gdn.replay_table(k, v, g, b, [[a[5] for a in layers]]), torch.int64, "cuda")
    rows = torch.tensor([path], dtype=torch.int32, device="cuda")
    many = gdn.replay(table, 5, 1, rows, torch.tensor([4], dtype=torch.int32, device="cuda"), k[0], v[0])[0]
    for i, args in enumerate(layers):
        assert torch.equal(many[i], _replay(args, [path], [args[5]])[0]), f"layer {i}"
    none = gdn.replay(table, 5, 1, rows, torch.zeros(1, dtype=torch.int32, device="cuda"), k[0], v[0])[0]
    for i, args in enumerate(layers):
        assert torch.equal(none[i], args[5])


@cuda_only
def test_replay_in_place_writes_each_state_back():
    layers = [_inputs(12, seed=30 + i) for i in range(3)]
    path = [0, 2, 5]
    k, v, g, b = ([a[j] for a in layers] for j in (1, 2, 3, 4))
    states = [[a[5].clone() for a in layers], [(a[5] * 0.5).contiguous() for a in layers]]
    table = gdn.to_device(gdn.replay_table(k, v, g, b, states), torch.int64, "cuda")
    rows = torch.tensor([path, path[:2] + [0]], dtype=torch.int32, device="cuda")
    counts = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    fresh = gdn.replay(table, 3, 2, rows, counts, k[0], v[0])
    assert gdn.replay(table, 3, 2, rows, counts, k[0], v[0], in_place=True) is None
    for s in range(2):
        for layer in range(3):
            assert torch.equal(states[s][layer], fresh[s, layer]), (s, layer)


@cuda_only
def test_pending_path_folds_the_commit_into_the_next_tree():
    """A tree with the last window's kept rows pending equals replay then tree, in outputs and committed states."""

    prev = _inputs(12, kh=16, hv=48, dv=128, seed=50)
    paths = [[0, 2, 5], [0, 1], []]
    trees = [SHAPES["spine"](9), SHAPES["chain"](4), [-1, 0, 0]]
    total = sum(len(t) for t in trees)
    now = _inputs(total, kh=16, hv=48, dv=128, seed=51)
    base = [prev[5] * (1.0 + 0.1 * s) for s in range(3)]
    width = max(len(p) for p in paths)
    rows = torch.tensor([p + [0] * (width - len(p)) for p in paths], dtype=torch.int32, device="cuda")
    counts = torch.tensor([len(p) for p in paths], dtype=torch.int32, device="cuda")
    committed = [(_replay(prev, [p], [b])[0] if p else b) for p, b in zip(paths, base)]
    want = gdn.tree(*now[:5], gdn.plan(trees, "cuda"), table=gdn.to_device(gdn.pointers(committed), torch.int64, "cuda"))
    states = [b.clone() for b in base]
    got = gdn.tree(*now[:5], gdn.plan(trees, "cuda"), table=gdn.to_device(gdn.pointers(states), torch.int64, "cuda"),
                   pending=(prev[1], prev[2], prev[3], prev[4], rows, counts))
    assert torch.equal(got, want)
    for s in range(3):
        assert torch.equal(states[s], committed[s]), f"stream {s} state"
    one = base[0].clone()
    alone = gdn.tree(*[x[:9].contiguous() for x in now[:5]], gdn.plan([trees[0]], "cuda"), state=one,
                     pending=(prev[1], prev[2], prev[3], prev[4], rows[:1], counts[:1]))
    assert torch.equal(alone, want[:9]) and torch.equal(one, committed[0])


@pytest.mark.parametrize("nodes", [1, 7, 1025, 2048])
@cuda_only
def test_long_chains_and_their_final_state(nodes):
    """Chains of any length (no plan): outputs equal the tree path's steps, and ``final`` equals replaying the chain."""

    args = _inputs(nodes, kh=16, hv=48, dv=128, seed=nodes)
    chain = list(range(-1, nodes - 1))
    final = torch.empty_like(args[5])
    out = gdn.tree(*args[:5], gdn.plan([chain], "cuda"), state=args[5], final=final)
    assert torch.equal(final, _replay(args, [list(range(nodes))], [args[5]])[0])
    for node in sorted({0, nodes // 2, nodes - 1}):
        state = args[5]
        if node:
            state = _replay(args, [list(range(node))], [args[5]])[0]
        one = [x[node:node + 1].contiguous() for x in args[:5]]
        assert torch.equal(out[node], gdn.tree(*one, gdn.plan([[-1]], "cuda"), state=state)[0]), f"node {node}"
    finals = [torch.empty_like(args[5]) for _ in range(2)]
    two = [x.repeat((2,) + (1,) * (x.dim() - 1)) if x.dim() > 1 else x for x in args[:5]]
    gdn.tree(*two, gdn.plan([chain, chain], "cuda"), table=gdn.to_device(gdn.pointers([args[5], args[5]]), torch.int64,
             "cuda"), final=gdn.to_device(gdn.pointers(finals), torch.int64, "cuda"))
    assert torch.equal(finals[0], final) and torch.equal(finals[1], final)


def test_schedule_reads_each_parent_state():
    """On the host: every scheduled node reads its parent's state from where the schedule left it."""

    rng = random.Random(11)
    for _ in range(3000):
        n = rng.randint(1, 128)
        parents = [-1] + [rng.randint(0, i - 1) for i in range(1, n)]
        entries, slots = gdn.schedule(parents)
        cur, slot = None, {}
        for i in range(0, len(entries), 3):
            node, source, dest = entries[i:i + 3]
            got = "committed" if source == -1 else cur if source == -2 else slot[source]
            assert got == ("committed" if parents[node] < 0 else parents[node])
            assert max(source, dest) < slots or (source < 0 and dest < 0)
            cur = node
            if dest >= 0:
                slot[dest] = node
    assert gdn.schedule(SHAPES["chain"](128))[1] == 0
    assert gdn.schedule(SHAPES["spine"](128))[1] <= 2


@pytest.mark.parametrize("keys", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hv", [6, 48])
@cuda_only
def test_prefill_chain_rows_do_not_depend_on_chunking(keys, hv):
    """Prefill's chain: a prompt split anywhere gives the same outputs and states; close to the verify kernel."""

    rows = 300
    q, k, v, g, beta, state = _inputs(rows, kh=2, hv=hv, dv=128, keys=keys)
    last = torch.empty_like(state)
    whole = gdn.chain(q, k, v, g, beta, state, last)
    for size in (1, 7, 16, 64, 256):
        cur, parts = state, []
        for a in range(0, rows, size):
            nxt = torch.empty_like(state)
            parts.append(gdn.chain(*(x[a:a + size].contiguous() for x in (q, k, v, g, beta)), cur, nxt))
            cur = nxt
        assert torch.equal(whole, torch.cat(parts)) and torch.equal(last, cur), size
    ref_last = torch.empty_like(state)
    ref = gdn.tree(q, k, v, g, beta, gdn.plan([list(range(-1, rows - 1))], "cuda"), state=state, final=ref_last)
    assert ((whole.float() - ref.float()).norm() / ref.float().norm()).item() < 1e-2
    assert ((last - ref_last).norm() / ref_last.norm()).item() < 1e-4



# ---- XPU (Triton T0): one step with a fixed halving sum, stacked states, no pointer tables ----


def _xinputs(dev, nodes: int, kh: int = 2, hv: int = 6, dv: int = 8, keys=torch.bfloat16, seed: int = 0):
    gen = torch.Generator(device=dev).manual_seed(41 + nodes + seed)
    q = (torch.randn((nodes, kh, 128), generator=gen, device=dev) * 0.01).to(keys)
    k = (torch.randn((nodes, kh, 128), generator=gen, device=dev) * 0.01).to(keys)
    v = torch.randn((nodes, hv, dv), generator=gen, device=dev).bfloat16()
    g = torch.rand((nodes, hv), generator=gen, device=dev) * 0.4 + 0.5
    beta = torch.rand((nodes, hv), generator=gen, device=dev)
    state = torch.randn((hv, dv, 128), generator=gen, device=dev) * 0.05
    return q, k, v, g, beta, state


def _xreplay(args, paths, states):
    """Each stream's rows replayed from its state (one layer) through the XPU replay."""

    from tensorfold.xpu.kernels import gdn as xgdn

    k, v, g, beta = args[1:5]
    width = max(1, max(len(p) for p in paths))
    rows = torch.tensor([p + [0] * (width - len(p)) for p in paths], dtype=torch.int32, device=k.device)
    counts = torch.tensor([len(p) for p in paths], dtype=torch.int32, device=k.device)
    stack = torch.stack(list(states))[:, None].contiguous()
    return xgdn.replay(k[None], v[None], g[None], beta[None], stack, rows, counts)[:, 0]


def _xstep(args, row, state):
    from tensorfold.xpu.kernels import gdn as xgdn

    one = [x[row:row + 1].contiguous() for x in args[:5]]
    y = xgdn.tree(*one, gdn.plan([[-1]], one[0].device), state=state)
    return y[0], _xreplay(one, [[0]], [state])[0]


def _xcheck_tree(parents, args, nodes_to_check=None):
    from tensorfold.xpu.kernels import gdn as xgdn

    out = xgdn.tree(*args[:5], gdn.plan([parents], args[0].device), state=args[5])
    torch.xpu.synchronize()
    for node in nodes_to_check or range(len(parents)):
        state = args[5]
        for row in _path(parents, node):
            expected, state = _xstep(args, row, state)
        assert torch.equal(out[node].view(torch.int16), expected.view(torch.int16)), f"node {node} != serial"
        assert torch.equal(_xreplay(args, [_path(parents, node)], [args[5]])[0], state), f"replay to {node}"


@xpu_only
@pytest.mark.parametrize("nodes", [1, 5, 16, 17, 33])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_xpu_tree_nodes_match_serial_paths(DEV, shape, nodes):
    parents = SHAPES[shape](nodes)
    check = sorted({0, nodes // 3, nodes // 2, nodes - 1} | set(range(min(nodes, 6))))
    _xcheck_tree(parents, _xinputs(DEV, nodes), check)


@xpu_only
def test_xpu_random_trees_and_fp32_keys(DEV):
    rng = random.Random(3)
    for trial in range(6):
        n = rng.randint(2, 40)
        parents = [-1] + [rng.randint(max(0, i - 6), i - 1) for i in range(1, n)]
        args = _xinputs(DEV, n, keys=torch.float32 if trial % 2 else torch.bfloat16, seed=trial)
        _xcheck_tree(parents, args, [n - 1, n // 2, rng.randrange(n)])


@xpu_only
def test_xpu_checkpoint_head_layout_and_repeats(DEV):
    """The 27B's heads (16 key, 48 value, 128 wide), every node; 20 repeats give the same bits."""

    from tensorfold.xpu.kernels import gdn as xgdn

    args = _xinputs(DEV, 6, kh=16, hv=48, dv=128)
    parents = [-1, 0, 0, 1, 3, 2]
    _xcheck_tree(parents, args)
    first = xgdn.tree(*args[:5], gdn.plan([parents], DEV), state=args[5])
    for _ in range(20):
        assert torch.equal(xgdn.tree(*args[:5], gdn.plan([parents], DEV), state=args[5]).view(torch.int16),
                           first.view(torch.int16))


@xpu_only
def test_xpu_several_streams_in_one_launch(DEV):
    """Each stream's rows equal its own launch, with states stacked and picked by index; replays of all streams too."""

    from tensorfold.xpu.kernels import gdn as xgdn

    trees = [SHAPES["spine"](12), SHAPES["chain"](7), [-1], SHAPES["binary"](16)]
    total = sum(len(t) for t in trees)
    args = _xinputs(DEV, total, kh=16, hv=48, dv=128, seed=4)
    states = [args[5] * (1.0 + 0.1 * s) for s in range(len(trees))]
    stack = torch.stack(states[::-1]).contiguous()                         # stored in reverse: the index picks
    index = torch.tensor(list(range(len(trees)))[::-1], dtype=torch.int32, device=DEV)
    together = xgdn.tree(*args[:5], gdn.plan(trees, DEV), states=stack, state_index=index)
    base, paths = 0, []
    for s, parents in enumerate(trees):
        rows = [x[base:base + len(parents)].contiguous() for x in args[:5]]
        alone = xgdn.tree(*rows, gdn.plan([parents], DEV), state=states[s])
        assert torch.equal(together[base:base + len(parents)], alone), f"stream {s}"
        paths.append([base + r for r in _path(parents, len(parents) - 1)])
        base += len(parents)
    replayed = _xreplay(args, paths, states)
    for s, path in enumerate(paths):
        assert torch.equal(replayed[s], _xreplay(args, [path], [states[s]])[0]), f"stream {s} replay"


@xpu_only
def test_xpu_replay_of_many_layers_zero_rows_and_in_place(DEV):
    from tensorfold.xpu.kernels import gdn as xgdn

    layers = [_xinputs(DEV, 12, seed=i) for i in range(5)]
    k, v, g, b = (torch.stack([a[j] for a in layers]).contiguous() for j in (1, 2, 3, 4))
    states = torch.stack([torch.stack([a[5] for a in layers]), torch.stack([a[5] * 0.5 for a in layers])]).contiguous()
    rows = torch.tensor([[0, 1, 3, 7], [0, 2, 0, 0]], dtype=torch.int32, device=DEV)
    counts = torch.tensor([4, 2], dtype=torch.int32, device=DEV)
    many = xgdn.replay(k, v, g, b, states, rows, counts)
    for s, path in enumerate(([0, 1, 3, 7], [0, 2])):
        for i, args in enumerate(layers):
            assert torch.equal(many[s, i], _xreplay(args, [path], [states[s, i]])[0]), (s, i)
    none = xgdn.replay(k, v, g, b, states, rows, torch.zeros(2, dtype=torch.int32, device=DEV))
    assert torch.equal(none, states)
    copy = states.clone()
    assert xgdn.replay(k, v, g, b, copy, rows, counts, in_place=True) is None
    assert torch.equal(copy, many)


@xpu_only
def test_xpu_pending_path_folds_the_commit_into_the_next_tree(DEV):
    from tensorfold.xpu.kernels import gdn as xgdn

    prev = _xinputs(DEV, 12, kh=16, hv=48, dv=128, seed=50)
    paths = [[0, 2, 5], [0, 1], []]
    trees = [SHAPES["spine"](9), SHAPES["chain"](4), [-1, 0, 0]]
    total = sum(len(t) for t in trees)
    now = _xinputs(DEV, total, kh=16, hv=48, dv=128, seed=51)
    base = [prev[5] * (1.0 + 0.1 * s) for s in range(3)]
    width = max(len(p) for p in paths)
    rows = torch.tensor([p + [0] * (width - len(p)) for p in paths], dtype=torch.int32, device=DEV)
    counts = torch.tensor([len(p) for p in paths], dtype=torch.int32, device=DEV)
    committed = [(_xreplay(prev, [p], [b])[0] if p else b) for p, b in zip(paths, base)]
    want = xgdn.tree(*now[:5], gdn.plan(trees, DEV), states=torch.stack(committed).contiguous())
    states = torch.stack(base).contiguous()
    got = xgdn.tree(*now[:5], gdn.plan(trees, DEV), states=states,
                    pending=(prev[1], prev[2], prev[3], prev[4], rows, counts))
    assert torch.equal(got, want)
    for s in range(3):
        assert torch.equal(states[s], committed[s]), f"stream {s} state"


@xpu_only
@pytest.mark.parametrize("nodes", [1, 7, 1025])
def test_xpu_long_chains_and_their_final_state(DEV, nodes):
    """Chains (no slots): outputs equal one-row steps, ``final`` equals the replay, and chain() equals the tree."""

    from tensorfold.xpu.kernels import gdn as xgdn

    args = _xinputs(DEV, nodes, kh=16, hv=48, dv=128, seed=nodes)
    path = list(range(nodes))
    final = torch.empty_like(args[5])
    out = xgdn.tree(*args[:5], gdn.plan([list(range(-1, nodes - 1))], DEV), state=args[5], final=final)
    torch.xpu.synchronize()
    assert torch.equal(final, _xreplay(args, [path], [args[5]])[0])
    for node in sorted({0, nodes // 2, nodes - 1}):
        state = _xreplay(args, [path[:node]], [args[5]])[0] if node else args[5]
        assert torch.equal(out[node], _xstep(args, node, state)[0]), f"node {node}"
    last = torch.empty_like(args[5])
    assert torch.equal(xgdn.chain(*args[:5], args[5], last), out) and torch.equal(last, final)


@xpu_only
@pytest.mark.parametrize("keys", [torch.bfloat16, torch.float32])
def test_xpu_prompt_chain_rows_do_not_depend_on_chunking(DEV, keys):
    from tensorfold.xpu.kernels import gdn as xgdn

    rows = 300
    q, k, v, g, beta, state = _xinputs(DEV, rows, kh=2, hv=6, dv=128, keys=keys)
    last = torch.empty_like(state)
    whole = xgdn.chain(q, k, v, g, beta, state, last)
    for size in (1, 7, 16, 64, 256):
        cur, parts = state, []
        for a in range(0, rows, size):
            nxt = torch.empty_like(state)
            parts.append(xgdn.chain(*(x[a:a + size].contiguous() for x in (q, k, v, g, beta)), cur, nxt))
            cur = nxt
        assert torch.equal(whole, torch.cat(parts)) and torch.equal(last, cur), size


@xpu_only
@pytest.mark.parametrize("rows_per_program", [1, 4, 16, 32])
def test_xpu_row_tile_changes_no_bits(DEV, monkeypatch, rows_per_program):
    from tensorfold.xpu.kernels import gdn as xgdn

    args = _xinputs(DEV, 9, kh=16, hv=48, dv=128, seed=9)
    parents = SHAPES["binary"](9)
    want = xgdn.tree(*args[:5], gdn.plan([parents], DEV), state=args[5])
    want_state = _xreplay(args, [[0, 1, 3, 7]], [args[5]])[0]
    monkeypatch.setattr(xgdn, "ROWS", rows_per_program)
    assert torch.equal(xgdn.tree(*args[:5], gdn.plan([parents], DEV), state=args[5]), want)
    assert torch.equal(_xreplay(args, [[0, 1, 3, 7]], [args[5]])[0], want_state)


@xpu_only
def test_xpu_matches_an_fp64_reference(DEV):
    from tensorfold.xpu.kernels import gdn as xgdn

    q, k, v, g, beta, state = _xinputs(DEV, 24, kh=16, hv=48, dv=128, seed=77)
    last = torch.empty_like(state)
    y = xgdn.chain(q, k, v, g, beta, state, last)
    rep = 3
    s = state.double()
    ys = []
    for t in range(24):
        kt = k[t].double().repeat_interleave(rep, 0)
        qt = q[t].double().repeat_interleave(rep, 0)
        s = s * g[t].double()[:, None, None]
        delta = (v[t].double() - (s * kt[:, None, :]).sum(-1)) * beta[t].double()[:, None]
        s = s + kt[:, None, :] * delta[:, :, None]
        ys.append((s * qt[:, None, :]).sum(-1))
    ref = torch.stack(ys)
    assert ((y.double() - ref).abs().max() <= ref.abs().max() * 2 ** -7).item()
    assert ((last.double() - s).abs().max() <= s.abs().max() * 1e-5).item()
