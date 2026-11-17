#!/usr/bin/env bash
# publish_state.sh under stubbed aws, uv and curl on PATH. Checks its exit
# codes and what reaches S3:
#
#   happy path                    -> the keyed file uploaded, exit 0
#   not main, a PR, another repo  -> exit 0, nothing built or uploaded
#   no AWS identity               -> exit 1, nothing built
#   the key already exists        -> exit 0, nothing built, object unchanged
#   non-default build switches, or the commit is not in the checkout
#                                 -> exit 0, nothing built
#   no worktree (build exit 3)    -> exit 1
#   refused (exit 4), build fails (exit 1), uv's own error (exit 2) -> exit 1
#   upload fails                  -> exit 1
#   uploaded but not public       -> exit 1
#
# The uv stub stands in for `uv run ... ci-publish-state`: what is under test
# is the script's gates, not the build.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=tree-hash-v1.json.gz
fail=0

run_case() { # name mode expect_rc expect_s3 expect_built
  local name=$1 mode=$2 want_rc=$3 want_s3=$4 want_built=$5
  local T; T=$(mktemp -d)
  mkdir -p "$T/bin" "$T/s3/ci/repostate" "$T/vllm"
  git -C "$T/vllm" init --quiet

  cat > "$T/bin/aws" <<'EOF'
#!/usr/bin/env bash
case "$1 $2" in
  "sts get-caller-identity") [[ "$MODE" != no-identity ]] ;;
  "s3api head-object") [[ -f "$S3/$6" ]] ;;
  "s3 cp")
    [[ "$MODE" != upload-fails ]] || exit 1
    cp "$3" "$S3/${4#s3://*/}" ;;
  *) echo "unexpected aws $*" >&2; exit 2 ;;
esac
EOF
  cat > "$T/bin/uv" <<'EOF'
#!/usr/bin/env bash
if [[ " $* " == *" --key-only "* ]]; then
  [[ "$MODE" == non-default ]] && exit 5
  [[ "$MODE" == no-commit ]] && exit 3
  echo "$KEY"; exit 0
fi
echo build >> "$LOG"
case "$MODE" in
  no-checkout) exit 3 ;;
  refused) exit 4 ;;
  build-fails) exit 1 ;;
  uv-error) exit 2 ;;
esac
mkdir -p out && echo "state" > "out/$KEY" && echo "$KEY"
EOF
  cat > "$T/bin/curl" <<'EOF'
#!/usr/bin/env bash
[[ "$MODE" != not-public ]]
EOF
  chmod +x "$T/bin/aws" "$T/bin/uv" "$T/bin/curl"
  [[ "$mode" == exists ]] && echo "old" > "$T/s3/ci/repostate/$KEY"

  local branch=main pr=false repo=https://github.com/vllm-project/vllm.git
  [[ "$mode" == branch ]] && branch=feature
  [[ "$mode" == pull-request ]] && pr=123
  [[ "$mode" == other-repo ]] && repo=https://github.com/someone/vllm.git
  PATH="$T/bin:$PATH" MODE="$mode" S3="$T/s3" KEY="$KEY" LOG="$T/uv.log" \
    BUILDKITE_BRANCH="$branch" BUILDKITE_PULL_REQUEST="$pr" BUILDKITE_REPO="$repo" \
    BUILDKITE_COMMIT=abc123 REPOSTATE_VLLM_REPO="$T/vllm" REPOSTATE_CI_INFRA="$T" \
    bash "$HERE/publish_state.sh" > "$T/out.log" 2>&1
  local rc=$?

  local got_s3=none built=no
  [[ -f "$T/s3/ci/repostate/$KEY" ]] && got_s3=$(cat "$T/s3/ci/repostate/$KEY")
  [[ -s "$T/uv.log" ]] && built=yes
  if [[ "$rc" == "$want_rc" && "$got_s3" == "$want_s3" && "$built" == "$want_built" ]]; then
    echo "ok   $name"
  else
    echo "FAIL $name: rc=$rc (want $want_rc), s3=$got_s3 (want $want_s3), built=$built (want $want_built)"
    sed 's/^/     /' "$T/out.log"
    fail=1
  fi
  rm -rf "$T"
}

run_case "happy path"              ok            0 state yes
run_case "not main"                branch        0 none  no
run_case "pull request"            pull-request  0 none  no
run_case "another repo"            other-repo    0 none  no
run_case "no AWS identity"         no-identity   1 none  no
run_case "key already published"   exists        0 old   no
run_case "non-default switches"    non-default   0 none  no
run_case "commit not in checkout"  no-commit     0 none  no
run_case "no worktree"             no-checkout   1 none  yes
run_case "refused"                 refused       1 none  yes
run_case "build fails"             build-fails   1 none  yes
run_case "uv's own error"          uv-error      1 none  yes
run_case "upload fails"            upload-fails  1 none  yes
run_case "not publicly readable"   not-public    1 state yes
exit "$fail"
