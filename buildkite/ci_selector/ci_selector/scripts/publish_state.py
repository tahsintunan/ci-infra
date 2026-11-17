# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the RepoState at one vLLM commit and write it under its cache key.

Prints the key on stdout; the caller uploads `<out-dir>/<key>` to
`ci/repostate/<key>`. Builds from a clean worktree at the commit, the same
input a reader's `state_for` builds from, so the tree in the key is exactly
what was parsed. Never reads the cache.

The state is built twice and written only when both builds encode to the same
bytes and neither recorded a degraded part: the build swallows some IO errors,
and a key is written once, so a state hit by one would be served for good.

Exit codes (2 is left to uv's own errors):
  0 written, or the key printed with --key-only
  1 the build or the encoding failed
  3 no checkout of the commit to build from
  4 refused: degraded, or the two builds differ
  5 skipped: a build switch is not at its default
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from ..codemap import state_cache
from ..codemap.state import RepoState
from ..codemap.worktree import git_out, worktree_at

NO_CHECKOUT = 3
REFUSED = 4
NOT_DEFAULT = 5


def degraded(state: RepoState) -> list[str]:
    """The parts a build caught any exception for, MemoryError and OSError
    included. Not rust_workspace.parse_failed: the tree's layout can set it,
    and it widens selection rather than narrowing it."""
    found = []
    if state.build_map.error:
        found.append(f"build_map.error: {state.build_map.error}")
    if state.native_ops.error:
        found.append(f"native_ops.error: {state.native_ops.error}")
    if state.docs_deps.degraded:
        found.append("docs_deps.degraded")
    return found


def _built(checkout: Path, tree: str) -> tuple[bytes | None, list[str]]:
    state = RepoState.build(checkout)
    problems = degraded(state)
    if problems:
        return None, problems
    return state_cache.dumps(state, tree), []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True, help="a vLLM clone")
    ap.add_argument("--commit", required=True, help="the commit to build at")
    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--out-dir", type=Path, help="write <out-dir>/<key>")
    out.add_argument(
        "--key-only", action="store_true", help="print the key, build nothing"
    )
    args = ap.parse_args(argv)

    switches = state_cache.build_switches()
    if switches != state_cache.DEFAULT_SWITCHES:
        print(
            f"not publishing: build switches {switches} are not the defaults "
            f"{state_cache.DEFAULT_SWITCHES}",
            file=sys.stderr,
        )
        return NOT_DEFAULT
    try:
        commit = git_out(
            args.repo, "rev-parse", "--verify", f"{args.commit}^{{commit}}"
        )
        tree = git_out(args.repo, "rev-parse", f"{commit}^{{tree}}")
        if args.key_only:
            print(state_cache.cache_key(tree))
            return 0
        checkout = worktree_at(args.repo, commit)
    except Exception as error:  # noqa: BLE001 - reported, exit 3
        print(f"error: no checkout of {args.commit}: {error}", file=sys.stderr)
        return NO_CHECKOUT
    started = time.monotonic()
    blob, problems = _built(checkout, tree)
    if blob is not None:
        again, problems = _built(checkout, tree)
        if again is not None and again != blob:
            problems = ["two builds of the same checkout encode differently"]
    if problems:
        print(
            f"refusing to publish {commit[:12]}: {'; '.join(problems)}",
            file=sys.stderr,
        )
        return REFUSED
    key = state_cache.cache_key(tree)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Renamed into place, so a crash cannot leave a partial file to upload.
    with tempfile.NamedTemporaryFile(dir=args.out_dir, delete=False) as tmp:
        tmp.write(blob)
    os.replace(tmp.name, args.out_dir / key)
    print(
        f"built {commit[:12]} (tree {tree[:12]}) twice in "
        f"{time.monotonic() - started:.1f}s, {len(blob) / 1e6:.1f} MB gzipped",
        file=sys.stderr,
    )
    print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
