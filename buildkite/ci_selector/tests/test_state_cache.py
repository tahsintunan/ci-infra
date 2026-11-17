# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The published base RepoState: a loaded state must equal a fresh build, and
anything wrong with the cache must end in a build, never in a different state."""

import dataclasses
import datetime
import gzip
import importlib
import json
import platform
import re
import subprocess
import types
import typing
import urllib.error
from pathlib import Path

import pytest
from ci_selector.codemap import state_cache, worktree
from ci_selector.codemap.classify import select
from ci_selector.codemap.pipeline.step import Step
from ci_selector.codemap.pipeline.targets import Target
from ci_selector.codemap.state import RepoState
from ci_selector.scripts import publish_state

# The `hasattr(self, "_x")` caches in state.py and graph/build.py. Not stored;
# selection fills them in on the shared `state` fixture.
LAZY_ATTRS = {
    "_test_index",
    "_direct_test_importers",
    "_family_partition",
    "_table_of",
    "_import_cycle",
    "_engine_tests",
}


def _reachable_dataclasses() -> list[type]:
    found: list[type] = []
    queue = [RepoState]
    while queue:
        cls = queue.pop()
        if cls in found:
            continue
        found.append(cls)
        hints = typing.get_type_hints(
            cls, vars(importlib.import_module(cls.__module__))
        )
        for f in dataclasses.fields(cls):
            stack = [hints[f.name]]
            while stack:
                t = stack.pop()
                stack.extend(typing.get_args(t))
                if dataclasses.is_dataclass(t) and isinstance(t, type):
                    queue.append(t)
    return found


def _assert_same(fresh, loaded, path="state"):
    """Equal values of identical types, all the way down. Stricter than ==,
    which takes a list for a tuple and a set for a frozenset."""
    assert type(fresh) is type(loaded), f"{path}: {type(fresh)} vs {type(loaded)}"
    if dataclasses.is_dataclass(fresh):
        names = {f.name for f in dataclasses.fields(fresh)}
        assert set(vars(fresh)) - names <= LAZY_ATTRS, path
        assert set(vars(loaded)) == names, path
        for name in names:
            if name in ("repo", "_reverse"):
                continue
            _assert_same(getattr(fresh, name), getattr(loaded, name), f"{path}.{name}")
        if hasattr(fresh, "reverse"):
            # Key order follows set iteration, which string-hash randomization
            # varies between any two processes: compare content only.
            _assert_same(
                dict(sorted(fresh.reverse.items())),
                dict(sorted(loaded.reverse.items())),
                f"{path}.reverse",
            )
    elif isinstance(fresh, dict):
        assert list(fresh) == list(loaded), path
        for k in fresh:
            _assert_same(fresh[k], loaded[k], f"{path}[{k!r}]")
    elif isinstance(fresh, list | tuple):
        assert len(fresh) == len(loaded), path
        for i, (a, b) in enumerate(zip(fresh, loaded)):
            _assert_same(a, b, f"{path}[{i}]")
    elif isinstance(fresh, set | frozenset):
        assert fresh == loaded, path
    elif hasattr(fresh, "pattern") and hasattr(fresh, "flags"):
        assert (fresh.pattern, fresh.flags) == (loaded.pattern, loaded.flags), path
    else:
        assert fresh == loaded, path


def _header(**overrides) -> dict:
    doc = {
        "schema": state_cache.SCHEMA_VERSION,
        "tree": "a" * 40,
        "selector": state_cache.selector_hash(),
        "build_env": state_cache.build_env(),
        "python": platform.python_version(),
        "state": {},
    }
    doc.update(overrides)
    return doc


def _clean_state(build_map=None, native_ops=None, docs_deps=False):
    """Enough of a RepoState for the publisher's degraded check."""
    return types.SimpleNamespace(
        build_map=types.SimpleNamespace(error=build_map),
        native_ops=types.SimpleNamespace(error=native_ops),
        rust_workspace=types.SimpleNamespace(parse_failed=False),
        docs_deps=types.SimpleNamespace(degraded=docs_deps),
    )


def _hash(root):
    return state_cache.source_hash([(root, frozenset({"build"}))])


def _g(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _publish(tmp_repo, out):
    return publish_state.main(
        ["--repo", str(tmp_repo), "--commit", "HEAD", "--out-dir", str(out)]
    )


@pytest.fixture
def pkg(tmp_path):
    root = tmp_path / "pkg"
    for rel, text in {
        "a.py": "x = 1\n",
        "graph/build/edges.py": "y = 2\n",
        "tables/names.json": "[]\n",
        "__pycache__/a.cpython-312.pyc": "junk",
        "sub/__pycache__/b.cpython-312.pyc": "junk",
        "pkg.egg-info/PKG-INFO": "junk",
        "build/lib/a.py": "stale copy\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.fixture(scope="module")
def round_trip(state, tmp_path_factory):
    blob = state_cache.dumps(state, tree="t" * 40)
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    return blob, state_cache.loads(blob, repo=elsewhere, tree="t" * 40)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-q")
    _g(repo, "config", "user.email", "t@t")
    _g(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    _g(repo, "add", ".")
    _g(repo, "commit", "-qm", "one")
    monkeypatch.setattr(worktree, "WORKTREE_CACHE", tmp_path / "cache")
    worktree.clear_state_cache()
    yield repo
    worktree.clear_state_cache()


@pytest.fixture
def builds(monkeypatch):
    """Stand-in for RepoState.build: records each call, returns a marker."""
    calls = []

    def fake_build(repo):
        calls.append(repo)
        return ("built", repo)

    monkeypatch.setattr(RepoState, "build", staticmethod(fake_build))
    return calls


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A cache directory served as a file:// URL, read by an enabled cache."""
    root = tmp_path / "served"
    root.mkdir()
    monkeypatch.setenv(state_cache.URL_ENV, root.as_uri())
    monkeypatch.setenv(state_cache.SWITCH_ENV, "on")
    return root


# --- the format -------------------------------------------------------------


def test_every_field_of_every_stored_class_has_a_codec():
    """A field added to any class RepoState holds must be added to the codec
    table too, or the cache would silently drop it."""
    reachable = _reachable_dataclasses()
    assert set(state_cache.SPECS) == set(reachable)
    for cls in reachable:
        fields = {f.name for f in dataclasses.fields(cls)}
        assert set(state_cache.SPECS[cls]) == fields, cls.__qualname__


def test_a_loaded_state_equals_a_fresh_build(state, round_trip):
    _blob, loaded = round_trip
    _assert_same(state, loaded)


def test_the_repo_path_is_rebound_and_nothing_else_names_the_build_tree(
    state, round_trip
):
    blob, loaded = round_trip
    assert loaded.repo != state.repo
    assert str(state.repo).encode() not in gzip.decompress(blob)


def test_a_loaded_state_selects_what_a_fresh_build_selects(state, round_trip):
    _blob, loaded = round_trip
    loaded.repo = state.repo  # select() reads the base checkout's files
    for paths in (
        ["vllm/config/__init__.py"],
        ["vllm/model_executor/layers/fused_moe/layer.py", "docs/README.md"],
        ["csrc/activation_kernels.cu", "requirements/common.txt"],
        ["tests/kernels/test_activation.py", ".buildkite/test_areas/kernels.yaml"],
    ):
        assert select(loaded, paths) == select(state, paths), paths


def test_the_file_is_gzipped_json_with_its_identity_up_front(round_trip):
    blob, _loaded = round_trip
    doc = json.loads(gzip.decompress(blob))
    assert doc | {"state": {}} == _header(tree="t" * 40)


def test_the_key_names_tree_selector_and_schema():
    key = state_cache.cache_key("a" * 40)
    assert key == (
        f"{'a' * 40}-{state_cache.selector_hash()}-v{state_cache.SCHEMA_VERSION}"
        ".json.gz"
    )
    assert re.fullmatch(r"[0-9a-f]{16}", state_cache.selector_hash())


@pytest.mark.parametrize("rel", ["a.py", "graph/build/edges.py", "tables/names.json"])
def test_any_source_or_data_file_moves_the_hash(pkg, rel):
    """A nested `build/` package and non-.py files count too."""
    first = _hash(pkg)
    (pkg / rel).write_text("changed\n")
    assert _hash(pkg) != first


@pytest.mark.parametrize(
    "rel",
    [
        "__pycache__/a.cpython-312.pyc",
        "sub/__pycache__/b.cpython-312.pyc",
        "pkg.egg-info/PKG-INFO",
        "build/lib/a.py",
    ],
)
def test_build_output_does_not_move_the_hash(pkg, rel):
    first = _hash(pkg)
    (pkg / rel).write_text("changed\n")
    assert _hash(pkg) == first


def test_a_top_level_build_dir_counts_unless_the_root_ignores_it(pkg):
    first = state_cache.source_hash([(pkg, frozenset())])
    (pkg / "build" / "lib" / "a.py").write_text("changed\n")
    assert state_cache.source_hash([(pkg, frozenset())]) != first


@pytest.mark.parametrize(
    "codec, value",
    [
        (state_cache.INT, True),
        (state_cache.STRSET, frozenset({"a"})),
        (state_cache.FROZENSTRS, {"a"}),
        (state_cache.tuple_of(state_cache.STR), ["a"]),
        (state_cache.pair_of(state_cache.STR, state_cache.STR), ("a",)),
        (state_cache.JSON_VALUE, {"when": datetime.date(2026, 1, 1)}),
        (state_cache.JSON_VALUE, {1: "int key"}),
    ],
)
def test_a_value_of_another_type_is_refused_on_write(codec, value):
    """Written as is, it would come back as a different type."""
    with pytest.raises(TypeError):
        codec.enc(value)


def test_raw_yaml_values_keep_their_types():
    """The build copies these straight from the pipeline file."""
    codec = state_cache.obj(Step)
    step = Step(
        pipeline="p",
        source_file="f.yaml",
        label="L",
        key=123,
        group=None,
        commands=["a", 1],
        source_file_dependencies=None,
        timeout_in_minutes=30.0,
        num_devices="2",
    )
    back = codec.dec(json.loads(json.dumps(codec.enc(step))))
    assert back == step
    assert (type(back.key), type(back.timeout_in_minutes), type(back.num_devices)) == (
        int,
        float,
        str,
    )


def test_an_encoding_error_names_the_field():
    step = Step(
        pipeline="p",
        source_file="f.yaml",
        label="L",
        key=None,
        group=None,
        commands=[],
        source_file_dependencies=None,
        env={1: "x"},
    )
    with pytest.raises(TypeError, match=r"Step\.env: non-string key"):
        state_cache.obj(Step).enc(step)


def test_a_pattern_with_other_flags_is_refused_both_ways():
    import regex

    with pytest.raises(TypeError):
        state_cache.PATTERN.enc(regex.compile("a", regex.IGNORECASE))
    plain = state_cache.PATTERN.enc(regex.compile("a"))
    assert state_cache.PATTERN.dec(plain).pattern == "a"
    with pytest.raises(ValueError):
        state_cache.PATTERN.dec(["a", plain[1] | regex.DEBUG])


@pytest.mark.parametrize("change", ["drop", "add"])
def test_a_stored_object_with_other_fields_is_refused_on_read(change):
    codec = state_cache.obj(Target)
    stored = codec.enc(Target(path="tests/x.py", kind="pytest"))
    if change == "drop":
        del stored["via"]
    else:
        stored["new_field"] = 1
    with pytest.raises(ValueError, match="differ"):
        codec.dec(stored)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema", -1),
        ("tree", "b" * 40),
        ("selector", "0" * 16),
        ("build_env", {"CI_SELECTOR_OPTIONAL": "manual"}),
        ("build_env", {**state_cache.build_env(), "encoding": "iso8859-1"}),
    ],
)
def test_a_mismatched_header_is_refused(field, value):
    blob = gzip.compress(json.dumps(_header(**{field: value})).encode())
    with pytest.raises(state_cache.CacheMismatch):
        state_cache.loads(blob, repo=Path("/x"), tree="a" * 40)


def test_another_python_patch_release_still_loads(round_trip):
    blob, _loaded = round_trip
    doc = json.loads(gzip.decompress(blob))
    doc["python"] = "3.12.0"
    other = gzip.compress(json.dumps(doc).encode())
    assert state_cache.loads(other, repo=Path("/x"), tree="t" * 40).repo == Path("/x")


def test_an_oversized_decompression_is_refused(monkeypatch):
    monkeypatch.setattr(state_cache, "MAX_RAW", 100)
    with pytest.raises(ValueError, match="over 100 bytes"):
        state_cache.loads(gzip.compress(b"0" * 1000), repo=Path("/x"), tree="t")


def test_bytes_after_the_gzip_stream_are_refused():
    blob = gzip.compress(json.dumps(_header()).encode()) + b"junk"
    with pytest.raises(ValueError, match="after the gzip stream"):
        state_cache.loads(blob, repo=Path("/x"), tree="a" * 40)


@pytest.mark.parametrize("name", ["utf-8", "UTF-8", "utf8"])
def test_spellings_of_one_encoding_agree(monkeypatch, name):
    """3.12 says "utf-8" under a C locale and "UTF-8" under en_US.UTF-8."""
    monkeypatch.setattr(
        state_cache.locale, "getpreferredencoding", lambda do_setlocale: name
    )
    assert state_cache.build_env()["encoding"] == "utf-8"


def test_a_truncated_file_is_refused():
    with pytest.raises(ValueError, match="truncated"):
        state_cache.loads(gzip.compress(b"{}" * 1000)[:-20], repo=Path("/x"), tree="t")


def test_the_log_prefix_matches_what_the_gate_looks_for():
    import selector_gate

    assert selector_gate.STATE_CACHE_LOG_PREFIX == state_cache.LOG_PREFIX


# --- the reader inside state_for -------------------------------------------


def test_a_hit_loads_without_building(tmp_repo, builds, served, monkeypatch, capsys):
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    loaded_from = []

    def fake_loads(blob, repo, tree):
        loaded_from.append((blob, repo, tree))
        return ("loaded", repo)

    (served / state_cache.cache_key(tree)).write_bytes(b"blob")
    monkeypatch.setattr(state_cache, "loads", fake_loads)
    got = worktree.state_for(tmp_repo, "HEAD")
    assert builds == []
    base = worktree.worktree_at(tmp_repo, "HEAD")
    assert got == ("loaded", base)
    assert loaded_from == [(b"blob", base, tree)]
    err = capsys.readouterr().err
    assert f"{state_cache.LOG_PREFIX} hit {state_cache.cache_key(tree)}" in err


def test_a_missing_file_falls_back_to_a_build(tmp_repo, builds, served, capsys):
    got = worktree.state_for(tmp_repo, "HEAD")
    base = worktree.worktree_at(tmp_repo, "HEAD")
    assert got == ("built", base)
    assert builds == [base]
    err = capsys.readouterr().err
    assert f"{state_cache.LOG_PREFIX} miss" in err
    assert err.count(state_cache.LOG_PREFIX) == 1


def test_a_404_falls_back_to_a_build(tmp_repo, builds, served, monkeypatch, capsys):
    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(state_cache.urllib.request, "urlopen", not_found)
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"
    assert "404" in capsys.readouterr().err


def test_a_mismatched_file_falls_back_to_a_build(tmp_repo, builds, served):
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    doc = _header(tree="b" * 40)  # published under the wrong key
    (served / state_cache.cache_key(tree)).write_bytes(
        gzip.compress(json.dumps(doc).encode())
    )
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"


def test_a_corrupt_file_falls_back_to_a_build(tmp_repo, builds, served):
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    (served / state_cache.cache_key(tree)).write_bytes(b"not gzip")
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"


@pytest.mark.parametrize(
    "limit, error",
    [("MAX_COMPRESSED", "over 4 bytes"), ("FETCH_DEADLINE", "took over")],
)
def test_a_download_past_its_limits_falls_back_to_a_build(
    tmp_repo, builds, served, monkeypatch, capsys, limit, error
):
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    (served / state_cache.cache_key(tree)).write_bytes(b"x" * 100)
    monkeypatch.setattr(state_cache, limit, 4 if limit == "MAX_COMPRESSED" else -1)
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"
    assert error in capsys.readouterr().err


def test_a_key_that_cannot_be_computed_falls_back_to_a_build(
    tmp_repo, builds, served, monkeypatch, capsys
):
    def broken():
        raise OSError("source unreadable")

    monkeypatch.setattr(state_cache, "selector_hash", broken)
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"
    assert "source unreadable" in capsys.readouterr().err


@pytest.mark.parametrize(
    "value, line",
    [("off", "disabled (CI_SELECTOR_GRAPH_CACHE=off)"), ("bogus", "unknown")],
)
def test_switching_the_cache_off_is_logged(
    tmp_repo, builds, served, monkeypatch, capsys, value, line
):
    monkeypatch.setenv(state_cache.SWITCH_ENV, value)
    monkeypatch.setenv("BUILDKITE", "true")
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    (served / state_cache.cache_key(tree)).write_bytes(b"would be read")
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"
    err = capsys.readouterr().err
    assert f"{state_cache.LOG_PREFIX} disabled" in err and line in err


def test_the_cache_is_off_outside_ci(tmp_repo, builds, served, monkeypatch, capsys):
    monkeypatch.delenv(state_cache.SWITCH_ENV)
    monkeypatch.delenv("BUILDKITE", raising=False)
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    (served / state_cache.cache_key(tree)).write_bytes(b"would be read")
    assert worktree.state_for(tmp_repo, "HEAD")[0] == "built"
    assert state_cache.LOG_PREFIX not in capsys.readouterr().err


@pytest.mark.parametrize(
    "buildkite, switch, expected",
    [
        (None, None, False),
        ("true", None, True),
        ("true", "off", False),
        (None, "on", True),
        ("true", "OFF", False),
        ("true", " Off ", False),
        ("true", "0", False),
        ("true", "false", False),
        ("true", "no", False),
        ("true", "disabled", False),
        ("true", "bogus", False),
        (None, "1", True),
        (None, "yes", True),
    ],
)
def test_the_switch(monkeypatch, buildkite, switch, expected):
    """Unknown values read as off: a typo stops reading, it never keeps it on."""
    for name, value in (("BUILDKITE", buildkite), (state_cache.SWITCH_ENV, switch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert state_cache.enabled() is expected


def test_ci_reads_by_default_only_while_read_in_ci_is_set(monkeypatch):
    monkeypatch.delenv(state_cache.SWITCH_ENV, raising=False)
    monkeypatch.setenv("BUILDKITE", "true")
    assert state_cache.switch() == (True, None)
    monkeypatch.setattr(state_cache, "READ_IN_CI", False)
    assert state_cache.switch() == (False, "disabled (off by default in CI)")


# --- the publisher ----------------------------------------------------------


def test_publish_writes_the_keyed_file_and_prints_the_key(
    tmp_repo, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        state_cache, "dumps", lambda state, tree: b"payload:" + tree.encode()
    )
    monkeypatch.setattr(RepoState, "build", staticmethod(lambda repo: _clean_state()))
    commit = _g(tmp_repo, "rev-parse", "HEAD")
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    out = tmp_path / "out"
    rc = publish_state.main(
        ["--repo", str(tmp_repo), "--commit", commit, "--out-dir", str(out)]
    )
    assert rc == 0
    key = state_cache.cache_key(tree)
    assert capsys.readouterr().out.strip() == key
    assert [p.name for p in out.iterdir()] == [key]
    assert (out / key).read_bytes() == b"payload:" + tree.encode()


def test_key_only_prints_the_key_and_builds_nothing(tmp_repo, builds, capsys):
    tree = _g(tmp_repo, "rev-parse", "HEAD^{tree}")
    rc = publish_state.main(["--repo", str(tmp_repo), "--commit", "HEAD", "--key-only"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == state_cache.cache_key(tree)
    assert builds == []


def test_publish_exits_3_when_the_commit_is_not_there(tmp_repo, tmp_path, builds):
    rc = publish_state.main(
        ["--repo", str(tmp_repo), "--commit", "0" * 40, "--out-dir", str(tmp_path)]
    )
    assert rc == publish_state.NO_CHECKOUT == 3
    assert builds == []


@pytest.mark.parametrize(
    "flag, message",
    [
        ({"build_map": "OSError()"}, "build_map.error: OSError()"),
        ({"native_ops": "MemoryError()"}, "native_ops.error: MemoryError()"),
        ({"docs_deps": True}, "docs_deps.degraded"),
    ],
)
def test_publish_refuses_a_degraded_build(
    tmp_repo, tmp_path, monkeypatch, capsys, flag, message
):
    monkeypatch.setattr(
        RepoState, "build", staticmethod(lambda repo: _clean_state(**flag))
    )
    assert _publish(tmp_repo, tmp_path / "out") == publish_state.REFUSED
    assert not (tmp_path / "out").exists()
    assert message in capsys.readouterr().err


def test_publish_refuses_when_only_the_second_build_is_degraded(
    tmp_repo, tmp_path, monkeypatch, capsys
):
    states = iter([_clean_state(), _clean_state(native_ops="OSError()")])
    monkeypatch.setattr(RepoState, "build", staticmethod(lambda repo: next(states)))
    monkeypatch.setattr(state_cache, "dumps", lambda state, tree: b"blob")
    assert _publish(tmp_repo, tmp_path / "out") == publish_state.REFUSED
    assert not (tmp_path / "out").exists()
    assert "native_ops.error: OSError()" in capsys.readouterr().err


def test_publish_does_not_refuse_a_rust_workspace_that_widens(
    tmp_repo, tmp_path, monkeypatch
):
    """parse_failed comes from the tree's layout and widens selection."""
    state = _clean_state()
    state.rust_workspace.parse_failed = True
    monkeypatch.setattr(RepoState, "build", staticmethod(lambda repo: state))
    monkeypatch.setattr(state_cache, "dumps", lambda state, tree: b"blob")
    assert _publish(tmp_repo, tmp_path / "out") == 0


def test_publish_refuses_two_builds_that_differ(
    tmp_repo, tmp_path, monkeypatch, capsys
):
    """One build hit by a swallowed IO error, the other not."""
    blobs = iter([b"first", b"second"])
    monkeypatch.setattr(RepoState, "build", staticmethod(lambda repo: _clean_state()))
    monkeypatch.setattr(state_cache, "dumps", lambda state, tree: next(blobs))
    assert _publish(tmp_repo, tmp_path / "out") == publish_state.REFUSED
    assert not (tmp_path / "out").exists()
    assert "encode differently" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--key-only"], ["--out-dir", "out"]])
def test_publish_skips_non_default_switches(tmp_repo, builds, monkeypatch, args):
    monkeypatch.setenv("CI_SELECTOR_OPTIONAL", "manual")
    rc = publish_state.main(["--repo", str(tmp_repo), "--commit", "HEAD", *args])
    assert rc == publish_state.NOT_DEFAULT
    assert builds == []
