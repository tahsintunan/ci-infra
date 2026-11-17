# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A RepoState published once per main commit, so a PR build loads its base's
state instead of building it.

Main builds publish (`ci-publish-state`); PR builds only read, over public
HTTPS. The object key names everything the state is a function of: the base
tree, the selector source that built it, and this file's schema. A key is
written once, so a reader gets either the state for exactly its inputs or a
404, and anything short of a clean load builds as before.

Gzipped JSON, never pickle: unpickling runs whatever the bucket holds. Every
field of every stored class is listed in `SPECS` with its codec.
"""

from __future__ import annotations

import codecs
import gzip
import hashlib
import json
import locale
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import regex

from .build_map import BuildMap
from .docs_deps import DocsDeps, Ref
from .graph.assets import AssetParse
from .graph.build import FullGraph
from .graph.demote import DispatchParse
from .graph.factories import FactoryParse
from .graph.imports import DynamicSite, ImportGraph
from .graph.model_registry import QuantParse, RegistryParse
from .graph.platform import PlatformParse
from .graph.spawn import SpawnParse
from .guards import PreflightReport
from .native_ops import NativeOps
from .pipeline.images import ArtifactGraph
from .pipeline.step import LoadReport, PipelineConfig, Step, optional_mode
from .pipeline.targets import StepTargets, Target
from .registered_names import KeyIndex
from .repo import ModuleIndex
from .rust_workspace import RustWorkspace
from .state import PipelineData, RepoState

# Bump on any change to how a state is written, including a codec change that
# keeps the field list.
SCHEMA_VERSION = 1
DEFAULT_URL = "https://vllm-ci-selector.s3.us-west-2.amazonaws.com/ci/repostate"
URL_ENV = "CI_SELECTOR_GRAPH_CACHE_URL"
# The reader's switch. Unset, it follows READ_IN_CI when BUILDKITE is set and
# is off elsewhere, so replays and local runs build their own state. Any value
# it does not know reads as off.
SWITCH_ENV = "CI_SELECTOR_GRAPH_CACHE"
READ_IN_CI = True
_OFF = frozenset({"off", "0", "false", "no", "disabled"})
_ON = frozenset({"on", "1", "true", "yes"})
LOG_PREFIX = "ci-selector repostate cache:"
FETCH_TIMEOUT = 10  # per socket operation
FETCH_DEADLINE = 60  # the whole download
MAX_COMPRESSED = 64 << 20
MAX_RAW = 512 << 20
# The switches RepoState.build reads, at their defaults. Main publishes only
# with these.
DEFAULT_SWITCHES = {"CI_SELECTOR_OPTIONAL": "ordinary"}


class CacheMismatch(ValueError):
    """The file is not the state for this tree, selector and schema."""


# --- codecs -----------------------------------------------------------------


@dataclass(frozen=True)
class Codec:
    enc: Callable[[Any], Any]
    dec: Callable[[Any], Any]


def _scalar(*types: type) -> Codec:
    def check(v):
        # exact type: True is an int to isinstance, not here
        if type(v) not in types:
            raise TypeError(f"expected {types}, got {type(v).__name__}: {v!r}")
        return v

    return Codec(check, check)


STR = _scalar(str)
INT = _scalar(int)
BOOL = _scalar(bool)


def optional(c: Codec) -> Codec:
    return Codec(
        lambda v: None if v is None else c.enc(v),
        lambda j: None if j is None else c.dec(j),
    )


def list_of(c: Codec) -> Codec:
    return Codec(
        lambda v: [c.enc(x) for x in _exact(v, list)], lambda j: [c.dec(x) for x in j]
    )


def tuple_of(c: Codec) -> Codec:
    return Codec(
        lambda v: [c.enc(x) for x in _exact(v, tuple)],
        lambda j: tuple(c.dec(x) for x in j),
    )


def set_of(c: Codec) -> Codec:
    # sorted, so one tree always writes the same bytes
    return Codec(
        lambda v: sorted(c.enc(x) for x in _exact(v, set)),
        lambda j: {c.dec(x) for x in j},
    )


def frozenset_of(c: Codec) -> Codec:
    return Codec(
        lambda v: sorted(c.enc(x) for x in _exact(v, frozenset)),
        lambda j: frozenset(c.dec(x) for x in j),
    )


def pair_of(*cs: Codec) -> Codec:
    """A fixed-length tuple, one codec per position."""

    def enc(v):
        if type(v) is not tuple or len(v) != len(cs):
            raise TypeError(f"expected a {len(cs)}-tuple, got {v!r}")
        return [c.enc(x) for c, x in zip(cs, v)]

    def dec(j):
        if len(j) != len(cs):
            raise ValueError(f"expected {len(cs)} items, got {len(j)}")
        return tuple(c.dec(x) for c, x in zip(cs, j))

    return Codec(enc, dec)


def dict_of(k: Codec, v: Codec) -> Codec:
    """Insertion order is kept: code may iterate a dict and take the first hit.
    String keys make a JSON object, any other key a list of pairs."""
    if k is STR:
        return Codec(
            lambda d: {STR.enc(a): v.enc(b) for a, b in _exact(d, dict).items()},
            lambda j: {a: v.dec(b) for a, b in j.items()},
        )
    return Codec(
        lambda d: [[k.enc(a), v.enc(b)] for a, b in _exact(d, dict).items()],
        lambda j: {k.dec(a): v.dec(b) for a, b in j},
    )


def _exact(v, t: type):
    if type(v) is not t:
        raise TypeError(f"expected {t.__name__}, got {type(v).__name__}")
    return v


def _json_value(v):
    """A YAML value as it came from the pipeline file. Only what JSON holds
    exactly is accepted: an int key or a date would come back changed."""
    if v is None or type(v) in (str, int, float, bool):
        return v
    if type(v) is list:
        return [_json_value(x) for x in v]
    if type(v) is dict:
        if any(type(key) is not str for key in v):
            raise TypeError(f"non-string key in {v!r}")
        return {key: _json_value(x) for key, x in v.items()}
    raise TypeError(f"not a JSON value: {type(v).__name__}: {v!r}")


JSON_VALUE = Codec(_json_value, lambda j: j)

# What docs_deps compiles its excludes with; any other flags (DEBUG prints to
# stdout) are refused both ways.
_REGEX_FLAGS = regex.compile("").flags


def _pattern_enc(p):
    if p.flags != _REGEX_FLAGS:
        raise TypeError(f"regex flags {p.flags:#x}, expected {_REGEX_FLAGS:#x}")
    return [p.pattern, p.flags]


def _pattern_dec(j):
    pattern, flags = j
    if flags != _REGEX_FLAGS:
        raise ValueError(f"regex flags {flags:#x}, expected {_REGEX_FLAGS:#x}")
    return regex.compile(pattern)


PATTERN = Codec(_pattern_enc, _pattern_dec)

# Not written. LAZY is recomputed on first use; REBOUND is set by the reader.
LAZY = Codec(lambda v: None, lambda j: None)
REBOUND = Codec(lambda v: None, lambda j: None)


def obj(cls: type) -> Codec:
    """A dataclass, field by field as SPECS lists it. Looked up when used, so
    the table can refer to classes defined below it."""

    def enc(v):
        if type(v) is not cls:
            raise TypeError(f"expected {cls.__name__}, got {type(v).__name__}")
        out = {}
        for name, c in _spec(cls).items():
            if c is LAZY or c is REBOUND:
                continue
            try:
                out[name] = c.enc(getattr(v, name))
            except (TypeError, ValueError) as error:
                raise TypeError(f"{cls.__name__}.{name}: {error}") from None
        return out

    def dec(j):
        spec = _spec(cls)
        stored = {n for n, c in spec.items() if c is not LAZY and c is not REBOUND}
        if set(j) != stored:
            raise ValueError(f"{cls.__name__}: fields {sorted(set(j) ^ stored)} differ")
        values = {name: spec[name].dec(j[name]) for name in stored}
        # set by the reader right after; None until then
        values.update({n: None for n, c in spec.items() if c is REBOUND})
        return cls(**values)

    return Codec(enc, dec)


def _spec(cls: type) -> dict[str, Codec]:
    spec = SPECS[cls]
    names = {f.name for f in fields(cls)}
    if set(spec) != names:
        raise TypeError(
            f"{cls.__name__} fields and its codec table differ: "
            f"{sorted(names ^ set(spec))}"
        )
    return spec


S = STR
STRS = list_of(S)
STRSET = set_of(S)
FROZENSTRS = frozenset_of(S)
STR_TO_STR = dict_of(S, S)
STR_TO_STRSET = dict_of(S, STRSET)
STR_TO_INT = dict_of(S, INT)

SPECS: dict[type, dict[str, Codec]] = {
    RepoState: {
        "repo": REBOUND,
        "pipelines": list_of(obj(PipelineData)),
        "full": obj(FullGraph),
        "catalog": STRS,
        "load_report": obj(LoadReport),
        "invoked": STRSET,
        "keys": obj(KeyIndex),
        "auto_step_ids": STRSET,
        "auto_run_files": STRSET,
        "auto_prefixes": tuple_of(S),
        "legacy_invoked": STRSET,
        "preflight": obj(PreflightReport),
        "exclusive_disabled": STRSET,
        "docs_deps": obj(DocsDeps),
        "release_refs": FROZENSTRS,
        "docker_inputs": STR_TO_STR,
        "artifacts": obj(ArtifactGraph),
        "rust_workspace": obj(RustWorkspace),
        "build_map": obj(BuildMap),
        "native_ops": obj(NativeOps),
        "gpu_name_aliases": STR_TO_STR,
    },
    PipelineData: {
        "config": obj(PipelineConfig),
        "steps": list_of(obj(Step)),
        "targets": dict_of(S, obj(StepTargets)),
    },
    # YAML values as the pipeline file wrote them; JSON_VALUE keeps each type.
    PipelineConfig: {
        "name": JSON_VALUE,
        "config_file": S,
        "job_dirs": JSON_VALUE,
        "run_all_patterns": JSON_VALUE,
        "run_all_exclude_patterns": JSON_VALUE,
    },
    Step: {
        "pipeline": S,
        "source_file": S,
        "label": S,
        "key": JSON_VALUE,
        "group": JSON_VALUE,
        "commands": JSON_VALUE,
        "source_file_dependencies": JSON_VALUE,
        "device": JSON_VALUE,
        "num_devices": JSON_VALUE,
        "num_nodes": JSON_VALUE,
        "working_dir": JSON_VALUE,
        "timeout_in_minutes": JSON_VALUE,
        "optional": BOOL,
        "soft_fail": BOOL,
        "autorun_on_main": BOOL,
        "no_plugin": BOOL,
        "dind": BOOL,
        "parallelism": JSON_VALUE,
        "depends_on": JSON_VALUE,
        "env": JSON_VALUE,
        "mirror_hw": JSON_VALUE,
        "mirror_of": JSON_VALUE,
        "mirror_label": JSON_VALUE,
        "extra": JSON_VALUE,
    },
    StepTargets: {
        "step_id": S,
        "targets": list_of(obj(Target)),
        "data_files": STRS,
        "config_lists": STRS,
        "ignored": STRS,
        "benign": STRS,
        "unparsable": STRS,
        "dangling": STRS,
        "container_tests": STRS,
        "unlexable": STRS,
        "scripts_seen": STRS,
        "haystack": S,
    },
    Target: {
        "path": S,
        "kind": S,
        "narrowing": STRS,
        "via": optional(S),
    },
    LoadReport: {
        "duplicate_keys": STRS,
        "duplicate_ids": STRS,
        "unknown_fields": dict_of(S, STRS),
    },
    FullGraph: {
        "index": obj(ModuleIndex),
        "graph": obj(ImportGraph),
        "registry": obj(RegistryParse),
        "quant": obj(QuantParse),
        "factories": obj(FactoryParse),
        "dispatch": obj(DispatchParse),
        "platform": obj(PlatformParse),
        "spawn": obj(SpawnParse),
        "assets": obj(AssetParse),
        "plain_reverse": STR_TO_STRSET,
    },
    ModuleIndex: {
        "module_to_file": STR_TO_STR,
        "file_to_module": STR_TO_STR,
        "installable_roots": STR_TO_STR,
        "native_modules": FROZENSTRS,
    },
    ImportGraph: {
        "imports": STR_TO_STRSET,
        "dynamic_sites": list_of(obj(DynamicSite)),
        "parse_errors": STRS,
        "string_literals": STR_TO_STRSET,
        "lazy_imports": set_of(pair_of(S, S)),
        "external_imports": STR_TO_STRSET,
        "dropped_lazy": list_of(pair_of(S, S)),
        "method_calls": STR_TO_STRSET,
        "from_import_aliases": list_of(pair_of(S, S, S)),
        "spawn_sites": STR_TO_STRSET,
        "ambiguities": list_of(pair_of(S, S, S, S)),
        "boot_edges": set_of(pair_of(S, S)),
        "demoted_edges": set_of(pair_of(S, S)),
        "engine_fixture_files": STRSET,
        "table_files": STRSET,
        "_reverse": LAZY,
    },
    DynamicSite: {"file": S, "lineno": INT, "func": S},
    RegistryParse: {
        "entries": dict_of(S, pair_of(S, S)),
        "hf_ids": STR_TO_STRSET,
        "unresolved": STRS,
        "edges_added": INT,
    },
    QuantParse: {"methods": STR_TO_STR, "edges_added": INT, "claims": STRSET},
    FactoryParse: {
        "register_entries": STR_TO_STR,
        "parser_entries": STR_TO_STR,
        "parser_entry_files": STR_TO_STRSET,
        "parser_table_counts": STR_TO_INT,
        "enum_entries": STR_TO_STR,
        "enum_table_counts": STR_TO_INT,
        "module_attrs": STR_TO_STR,
        "pkgutil_dirs": STRS,
        "class_table_entries": STR_TO_STR,
        "lazy_table_counts": STR_TO_INT,
        "backend_literal_counts": STR_TO_INT,
        "parser_engine_entries": STR_TO_STR,
        "table_of": STR_TO_STRSET,
        "claims": STRSET,
        "edges_added": INT,
        "module_attr_resolved": INT,
    },
    DispatchParse: {
        "demotions": dict_of(pair_of(S, S), STRSET),
        "claims": STRSET,
        "edges_added": INT,
    },
    PlatformParse: {
        "candidates": STR_TO_STRSET,
        "non_literal": STRSET,
        "unresolved": STR_TO_STRSET,
        "consumers": STR_TO_STRSET,
        "edges_added": INT,
        "parse_errors": STRS,
    },
    SpawnParse: {
        "spawner_classes": STR_TO_STRSET,
        "edges_added": INT,
        "entrypoint_file": optional(S),
    },
    AssetParse: {"edges_added": INT, "files_claimed": STRSET},
    KeyIndex: {
        "keyed_modules": STR_TO_STRSET,
        "step_keys": STR_TO_STRSET,
        "searchable": STR_TO_STR,
        "commands": STR_TO_STR,
        "key_mechanism": STR_TO_STR,
        "refused": STR_TO_STR,
    },
    PreflightReport: {
        "run_all_reasons": STRS,
        "force_select": STR_TO_STR,
        "boot_gate_ok": BOOL,
        "parse_error_paths": FROZENSTRS,
        "unmapped_devices": FROZENSTRS,
        "unclassified_sites": FROZENSTRS,
        "warnings": STRS,
    },
    DocsDeps: {
        "infra_files": FROZENSTRS,
        "floor_prefixes": tuple_of(S),
        "vllm_exclude_prefixes": tuple_of(S),
        "vllm_exclude_regexes": tuple_of(PATTERN),
        "extension": STR_TO_STR,
        "refs": tuple_of(obj(Ref)),
        "degraded": BOOL,
    },
    Ref: {"kind": S, "target": S, "md_file": S, "line": INT},
    ArtifactGraph: {
        "dependents": STR_TO_STRSET,
        "defined_by": STR_TO_STRSET,
        "producers_of": STR_TO_STRSET,
        "unresolved": STRSET,
        "inputs_of": STR_TO_STRSET,
        "explicit_inputs_of": STR_TO_STRSET,
        "self_builders": STR_TO_STRSET,
    },
    RustWorkspace: {
        "members": STR_TO_STR,
        "deps": STR_TO_STRSET,
        "binary_crates": STRSET,
        "cdylib_crates": STRSET,
        "parse_failed": BOOL,
    },
    BuildMap: {
        "families": dict_of(S, FROZENSTRS),
        "unresolved_headers": INT,
        "total_headers": INT,
        "error": optional(S),
    },
    NativeOps: {
        "file_ops": dict_of(S, FROZENSTRS),
        "wrappers": dict_of(S, frozenset_of(pair_of(S, S))),
        "op_test_files": dict_of(S, FROZENSTRS),
        "namespaces": FROZENSTRS,
        "op_count": INT,
        "error": optional(S),
    },
}

_STATE = obj(RepoState)


# --- identity -----------------------------------------------------------------


def _counts(rel: Path, ignore_top: frozenset[str]) -> bool:
    if rel.parts[0] in ignore_top or rel.suffix == ".pyc":
        return False
    return not any(
        part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts
    )


def source_hash(
    roots: Iterable[tuple[Path, frozenset[str]]], extra: bytes = b""
) -> str:
    """A hash of every file under each root, by relative path and content.
    Bytecode, egg-info and the root's `ignore_top` directories do not count."""
    h = hashlib.sha256(extra)
    for root, ignore_top in roots:
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and _counts(p.relative_to(root), ignore_top)
        )
        for p in files:
            h.update(f"{root.name}/{p.relative_to(root).as_posix()}\0".encode())
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:16]


_SELECTOR_HASH: str | None = None


def selector_hash() -> str:
    """The code a state is a function of: this package, the generator modules
    it imports (step keys, AMD labels), the locked dependencies and the Python
    minor version, whose ast a state is parsed with. Writer and reader compute
    it the same way, from the checkout they run from."""
    global _SELECTOR_HASH
    if _SELECTOR_HASH is None:
        import buildkite_step

        package = Path(__file__).resolve().parents[1]
        generator = Path(buildkite_step.__file__).resolve().parent
        extra = f"python {sys.version_info[0]}.{sys.version_info[1]}\0".encode()
        for name in ("pyproject.toml", "uv.lock"):
            path = package.parent / name
            if path.is_file():
                extra += name.encode() + b"\0" + path.read_bytes() + b"\0"
        # setuptools writes build/ into the generator directory
        _SELECTOR_HASH = source_hash(
            [(package, frozenset()), (generator, frozenset({"build"}))], extra
        )
    return _SELECTOR_HASH


def cache_key(tree: str) -> str:
    return f"{tree}-{selector_hash()}-v{SCHEMA_VERSION}.json.gz"


def build_switches() -> dict[str, str]:
    """The environment switches `RepoState.build` reads."""
    return {"CI_SELECTOR_OPTIONAL": optional_mode()}


def build_env() -> dict[str, str]:
    """What a build depends on beyond tree and code: the switches, and the
    encoding every `read_text()` without one decodes with. Not in the key:
    main publishes with the defaults, and a run that differs gets a mismatch
    on load and builds its own state."""
    # canonical name: 3.12 says "utf-8" under C and "UTF-8" under en_US.UTF-8
    encoding = codecs.lookup(locale.getpreferredencoding(False)).name
    return {**build_switches(), "encoding": encoding}


# --- read and write -------------------------------------------------------------


def dumps(state: RepoState, tree: str) -> bytes:
    doc = {
        "schema": SCHEMA_VERSION,
        "tree": tree,
        "selector": selector_hash(),
        "build_env": build_env(),
        # recorded, not checked: ast output is stable within a minor version
        "python": platform.python_version(),
        "state": _STATE.enc(state),
    }
    raw = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()
    return gzip.compress(raw, mtime=0)


def loads(blob: bytes, repo: Path, tree: str) -> RepoState:
    """The state in `blob`, bound to `repo`. Raises CacheMismatch when it was
    written for another tree, selector, schema or build environment; anything
    unreadable raises too."""
    doc = json.loads(_gunzip(blob))
    want = {
        "schema": SCHEMA_VERSION,
        "tree": tree,
        "selector": selector_hash(),
        "build_env": build_env(),
    }
    for name, value in want.items():
        if doc.get(name) != value:
            raise CacheMismatch(f"{name} is {doc.get(name)!r}, expected {value!r}")
    state = _STATE.dec(doc["state"])
    state.repo = repo
    return state


def switch() -> tuple[bool, str | None]:
    """Whether to read the cache, and the line to log when the switch, not
    the default, decided it."""
    raw = os.environ.get(SWITCH_ENV) or ""
    value = raw.strip().lower()
    if value in _OFF:
        return False, f"disabled ({SWITCH_ENV}={raw})"
    if value in _ON:
        return True, None
    if value:
        return False, f"disabled: unknown {SWITCH_ENV}={raw!r}, expected on or off"
    if os.environ.get("BUILDKITE"):
        return READ_IN_CI, None if READ_IN_CI else "disabled (off by default in CI)"
    return False, None


def enabled() -> bool:
    return switch()[0]


def _get(url: str) -> bytes:
    deadline = time.monotonic() + FETCH_DEADLINE
    chunks, size = [], 0
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as r:  # noqa: S310 - https, or a file:// override
        # read1: at most one socket read per call, so the deadline is checked
        # even while the body trickles in
        read = getattr(r, "read1", r.read)
        while chunk := read(1 << 16):
            size += len(chunk)
            if size > MAX_COMPRESSED:
                raise ValueError(f"over {MAX_COMPRESSED} bytes")
            if time.monotonic() > deadline:
                raise TimeoutError(f"download took over {FETCH_DEADLINE}s")
            chunks.append(chunk)
    return b"".join(chunks)


def _gunzip(blob: bytes) -> bytes:
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = inflate.decompress(blob, MAX_RAW)
    if inflate.unconsumed_tail:
        raise ValueError(f"decompresses to over {MAX_RAW} bytes")
    if not inflate.eof:
        raise ValueError("truncated gzip stream")
    if inflate.unused_data:
        raise ValueError("bytes after the gzip stream")
    return raw


def git_tree(checkout: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


def load_or_build(worktree: Path) -> RepoState:
    """The published state for the tree checked out at `worktree` when the
    cache is on and has it, bound to `worktree`; otherwise a fresh build of
    `worktree`. Never writes, and never fails where a build would not."""
    on, note = switch()
    if not on:
        if note:
            _log(note)
        return RepoState.build(worktree)
    key = "?"
    started = time.monotonic()
    try:
        tree = git_tree(worktree)
        key = cache_key(tree)
        url = f"{(os.environ.get(URL_ENV) or DEFAULT_URL).rstrip('/')}/{key}"
        state = loads(_get(url), repo=worktree, tree=tree)
    except Exception as error:  # noqa: BLE001 - any failure is a miss
        why = f"{type(error).__name__}: {error}"[:300]
    else:
        _log(f"hit {key} loaded in {time.monotonic() - started:.1f}s")
        return state
    started = time.monotonic()
    state = RepoState.build(worktree)
    _log(f"miss {key} ({why}); built in {time.monotonic() - started:.1f}s")
    return state
