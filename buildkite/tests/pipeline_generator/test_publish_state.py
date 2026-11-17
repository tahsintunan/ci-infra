"""Main builds publish the selector's base state for their own commit."""

import subprocess
from pathlib import Path

import pytest
import yaml

import buildkite_step
import pipeline_generator as pg
from constants import AgentQueue
from step import Step

SELECTOR_DIR = Path(__file__).resolve().parents[2] / "ci_selector"


def _generate(tmp_path, monkeypatch, config):
    config.update(job_dirs=["jobs"], list_file_diff=["vllm/engine.py"])
    monkeypatch.setattr(pg, "get_global_config", lambda: config)
    monkeypatch.setattr(
        pg,
        "read_steps_from_job_dir",
        lambda _: [Step(label="A", key="test-a", commands=["a"])],
    )
    monkeypatch.delenv("BUILDKITE_JOB_ID", raising=False)
    monkeypatch.setattr(
        pg.subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    generator = object.__new__(pg.PipelineGenerator)
    generator.output_file_path = str(tmp_path / "pipeline.yaml")
    generator.generate()
    pipeline = yaml.safe_load(open(tmp_path / "pipeline.yaml"))
    return {
        step["key"]: (group.get("group"), step)
        for group in pipeline["steps"]
        for step in group.get("steps", [group])
        if step.get("key")
    }


def test_a_main_build_publishes_from_the_postmerge_queue(
    tmp_path, monkeypatch, fake_global_config
):
    fake_global_config["branch"] = "main"
    steps = _generate(tmp_path, monkeypatch, fake_global_config)
    group, step = steps[buildkite_step.PUBLISH_STATE_KEY]
    assert group == buildkite_step.SELECTOR_GROUP
    # only postmerge agents can write the bucket
    assert step["agents"] == {"queue": AgentQueue.SMALL_CPU_POSTMERGE.value}
    assert step["depends_on"] == []
    assert step["soft_fail"] is True
    assert step["retry"] == buildkite_step.ensure_infra_failure_retry(None)


@pytest.mark.parametrize(
    "name, branch, pull_request, nightly",
    [
        ("vllm_ci", "test-branch", "false", "0"),
        ("vllm_ci", "releases/v0.12", "false", "0"),
        ("vllm_ci", "main", "false", "1"),
        ("vllm_ci", "main", "123", "0"),
        ("vllm_intel", "main", "false", "0"),
    ],
)
def test_no_other_build_publishes(
    tmp_path, monkeypatch, fake_global_config, name, branch, pull_request, nightly
):
    fake_global_config.update(
        name=name, branch=branch, pull_request=pull_request, nightly=nightly
    )
    steps = _generate(tmp_path, monkeypatch, fake_global_config)
    assert buildkite_step.PUBLISH_STATE_KEY not in steps
    assert steps, "a pipeline was generated"


def test_the_step_fetches_the_script_from_the_generating_branch(monkeypatch):
    monkeypatch.setenv("VLLM_CI_BRANCH", "my-branch")
    command = buildkite_step.publish_state_group().steps[0].commands[-1]
    assert "ci-infra/my-branch/buildkite/ci_selector/publish_state.sh" in command
    name = command.split("buildkite/ci_selector/")[1].split('"')[0]
    assert (SELECTOR_DIR / name).is_file(), f"{name} is not in {SELECTOR_DIR}"


def test_each_job_downloads_the_script_to_a_path_of_its_own():
    command = buildkite_step.publish_state_group().steps[0].commands[-1]
    assert "/tmp/" not in command
    assert command.startswith('f="$$(mktemp)" && ')
    assert command.endswith('{ bash "$$f"; rc=$$?; rm -f "$$f"; exit $$rc; }')
