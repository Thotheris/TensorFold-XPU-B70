# CLAUDE.md

@AGENTS.md

The import above loads the full agent manual. These notes cover how Claude Code should work in this repo specifically.

## Start of every session
1. Find your role: read the task and match it to a role in AGENTS.md §7. Stay within the paths that role owns.
2. Check the current state:
   - read `docs/xpu/STATUS.md` (if present);
   - run `git fetch origin results && git show origin/results:index.jsonl | tail -n 20` to see recent B70 runs;
   - read the kernel card for anything you touch.
3. Branch from `xpu/main` as `xpu/<ws>/<topic>` before editing. Don't commit port code directly to `main`.

## Working style here
- **This machine is not the B70 box.** You can run `python -m pytest tests -q` and `ruff check src tests` locally. GPU
  tests only run on the B70 through `.b70/run.yml` and the `results` branch. Say plainly which tests ran where, and
  never report a GPU result you did not read from a bundle.
- **Exactness first.** Before you write or change a kernel, write down its arithmetic contract in the kernel card.
  After the change, name the invariance tests that cover it. If no test pins the bits you changed, add one.
- **Keep CUDA unchanged.** When you edit a shared module, re-read the CUDA branch of the code and confirm it is
  byte-for-byte the same behaviour. A good self-check is to diff the CUDA path in your head before committing.
- **Verify facts in the guides.** Hardware and Triton facts tagged `[UNVERIFIED]` or `[CONFLICT]` are hypotheses. Add
  a probe to the `env` / `triton-smoke` suite rather than coding around a guess.
- **Match upstream style:** one-line docstrings that state what is true, sparse comments, and 120 columns. Read a
  neighbouring file first.
- **Keep changes small.** One kernel or one layer per branch keeps B70 runs short and diffs reviewable.

## Subagents and parallel work
- The plan is designed for several agents working in parallel on separate `xpu/*` branches with disjoint file
  ownership (AGENTS.md §7). When spawning subagents, give each one role, one branch and the paths it owns. Tell it to
  read AGENTS.md and the relevant guide first.
- Use read-only exploration agents for surveys (for example "which tests pin `_scan` bits"). Use writing agents only
  for owned paths.

## Things to never do
- Source oneAPI `setvars.sh` / put `icpx` on PATH in a shell that runs Python. Native builds go through
  `tools/xpu/build_ext.sh`.
- `pip install` anything that could replace the pinned torch or triton on the B70 venv.
- Force-push `main`, `xpu/main` or `results`; merge `results` into code; commit weights, `.so` files or secrets.
- Mark a kernel or milestone done without a green B70 bundle for that exact SHA.

## Useful commands
```bash
python -m pip install -e '.[test]'
python -m pytest tests -q
ruff check src tests
git fetch origin results && git show origin/results:index.jsonl | tail -n 20
git fetch upstream    # ashhart/TensorFold, for reference; syncs are the Integrator's job
```
