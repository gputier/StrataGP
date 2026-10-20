#!/usr/bin/env bash
# check_ik_patches.sh - do ik_routing.patch then ik_tiers.patch apply on ik_llama at the base commit?
#
# Run from the Mac.  A throwaway container (own name, --rm) receives the two patches through a tar stream (no bind
# mount), clones ik_llama without blobs, checks out the base commit, then runs `git apply --check` and `git apply` for
# each patch in order.  Exit code 0 only if every step passes.
#
# IK_PATCH_DIR overrides the directory holding the two patches (default: the directory of this script).
set -euo pipefail

BASE_COMMIT=c446f1e089f3dc2436b0600596c23abde00d988f
IK_REPO=https://github.com/ikawrakow/ik_llama.cpp
IMAGE=alpine/git
CONTAINER="ik-patches-check-$$"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_dir="${IK_PATCH_DIR:-$here}"

for p in ik_routing.patch ik_tiers.patch; do
  [[ -f "$patch_dir/$p" ]] || { echo "missing $patch_dir/$p" >&2; exit 2; }
done

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

tar -cf - -C "$patch_dir" ik_routing.patch ik_tiers.patch |
  docker run -i --rm --name "$CONTAINER" --entrypoint sh -e BASE_COMMIT="$BASE_COMMIT" -e IK_REPO="$IK_REPO" "$IMAGE" -c '
    set -eu
    mkdir /w && tar -xf - -C /w
    git clone --quiet --filter=blob:none --no-checkout "$IK_REPO" /ik
    cd /ik
    git checkout --quiet --detach "$BASE_COMMIT"
    echo "base: $(git rev-parse HEAD)"
    for p in ik_routing.patch ik_tiers.patch; do
      git apply --check "/w/$p"
      git apply "/w/$p"
      echo "applied: $p"
    done
    git status --short
  '
echo "OK: both patches apply on $BASE_COMMIT"
