#!/usr/bin/env bash
# Main-build step: publish the selector's RepoState for this commit, so PR
# builds based on it load the state instead of building it.
#
# Exit 0 means published, or skipped for an expected reason: not a vLLM main
# build, the key already exists, or a build switch is not at its default.
# Every way of failing to publish exits 1, so the (soft_fail) step shows it.
# The object key names the tree, the selector source and the schema, and is
# written once.
set -uo pipefail

BRANCH="${VLLM_CI_BRANCH:-main}"
REPO_URL="https://github.com/vllm-project/ci-infra"
# Readers fetch from PUBLIC_URL (state_cache.DEFAULT_URL); trying another
# bucket means setting CI_SELECTOR_GRAPH_CACHE_URL for them too.
BUCKET="${CI_SELECTOR_BUCKET:-vllm-ci-selector}"
PREFIX="ci/repostate"
PUBLIC_URL="${REPOSTATE_PUBLIC_URL:-https://${BUCKET}.s3.us-west-2.amazonaws.com/${PREFIX}}"
COMMIT="${BUILDKITE_COMMIT:?}"
VLLM_CHECKOUT="${REPOSTATE_VLLM_REPO:-${BUILDKITE_BUILD_CHECKOUT_PATH:?}}"
# A ci-infra checkout to take the selector from, instead of cloning one. For
# running this by hand, and for test_publish_state.sh.
CI_INFRA="${REPOSTATE_CI_INFRA:-}"

WORK="$(mktemp -d)"
cleanup() {
  local rc=$?
  rm -rf -- "${WORK}"
  # The state is built in a worktree of the agent's checkout; drop its entry.
  git -C "${VLLM_CHECKOUT}" worktree prune >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT
cd "${WORK}" || exit 1
fail() { echo "$1; not publishing" >&2; exit 1; }

if [[ "${BUILDKITE_BRANCH:-main}" != "main" ]]; then
  echo "branch ${BUILDKITE_BRANCH} is not main; PR bases are main commits, nothing to publish"
  exit 0
fi
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
  echo "pull request ${BUILDKITE_PULL_REQUEST}; only main commits are published"
  exit 0
fi
case "${BUILDKITE_REPO:-}" in
  *github.com[:/]vllm-project/vllm|*github.com[:/]vllm-project/vllm.git) ;;
  *) echo "repo ${BUILDKITE_REPO:-unset} is not vllm-project/vllm; nothing to publish"; exit 0 ;;
esac
command -v aws >/dev/null 2>&1 || fail "aws cli not on this agent"
aws sts get-caller-identity >/dev/null 2>&1 || fail "no AWS identity on this agent"

if [[ -n "${CI_INFRA}" ]]; then
  echo "--- :package: Using the selector at ${CI_INFRA}"
else
  echo "--- :package: Fetching the selector"
  git clone --quiet --depth 1 --branch "${BRANCH}" "${REPO_URL}" ci-infra \
    || fail "cannot clone ${REPO_URL}@${BRANCH}"
  CI_INFRA="${WORK}/ci-infra"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf --retry 3 https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${WORK}/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
  export PATH="${WORK}/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || fail "cannot install uv"
# The same install the bootstrap's selector run uses, so both compute the
# same selector hash.
selector() {
  CI_SELECTOR_WORKTREE_CACHE="${WORK}/worktrees" \
    uv run --quiet --no-dev --python 3.12 --project "${CI_INFRA}/buildkite/ci_selector" "$@"
}

# ci-publish-state exit codes: 3 no checkout, 4 refused, 5 non-default
# switches; 2 is uv's own.
KEY="$(selector ci-publish-state --repo "${VLLM_CHECKOUT}" --commit "${COMMIT}" --key-only)"
rc=$?
case "$rc" in
  3) echo "no checkout of ${COMMIT} to build from; not publishing" >&2; exit 0 ;;
  5) echo "build switches are not at their defaults; main publishes only default states"; exit 0 ;;
esac
[[ "$rc" == 0 && -n "${KEY}" ]] || fail "cannot compute the cache key (exit ${rc})"
DEST="s3://${BUCKET}/${PREFIX}/${KEY}"
if aws s3api head-object --bucket "${BUCKET}" --key "${PREFIX}/${KEY}" >/dev/null 2>&1; then
  echo "${DEST} already exists; keys are written once"
  exit 0
fi

echo "--- :hammer: Building the RepoState at ${COMMIT}"
selector ci-publish-state --repo "${VLLM_CHECKOUT}" --commit "${COMMIT}" --out-dir out
rc=$?
case "$rc" in
  0) ;;
  # The key-only call resolved the commit, so 3 here is a failed worktree.
  3) fail "cannot create a worktree at ${COMMIT}" ;;
  4) fail "the state was degraded or not reproducible" ;;
  *) fail "state build failed (exit ${rc})" ;;
esac

echo "--- :s3: Publishing ${DEST}"
aws s3 cp "out/${KEY}" "${DEST}" --only-show-errors --content-type application/gzip \
  || fail "S3 upload failed (bucket or write permission not in place?)"
# Readers fetch anonymously: a bucket policy that does not cover the prefix
# would make every read a miss.
curl -sfI --retry 3 --max-time 30 "${PUBLIC_URL}/${KEY}" >/dev/null \
  || { echo "${DEST} uploaded but not publicly readable at ${PUBLIC_URL}; readers will miss" >&2; exit 1; }
echo "published ${DEST}"
