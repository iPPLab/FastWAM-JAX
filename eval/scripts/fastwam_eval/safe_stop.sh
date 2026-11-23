#!/bin/bash
# Stop OUR campaign processes, and only ours. Print the candidate list, verify every PID
# belongs to us, then kill one PID at a time.
#
# WHY THIS FILE EXISTS (2026-08-04, near-miss):
#   for p in $(pgrep -f 'timeout -k 60 7200'); do kill -9 "$p"; done
# was run to clear one stuck client. That pattern ALSO matches the c1 campaign's clients —
# c1 uses the same `timeout -k 60 7200` wrapper. c1 survived on timing luck, not on
# anything the command did. An unqualified pattern is a red line whether or not it lands.
#
# The rule, in executable form:
#   1. every pgrep pattern carries a qualifier that is ours: port 6033/6035, or a path
#      under this repository (see OURS_RE below);
#   2. the candidate list is PRINTED and each PID re-verified against its own /proc argv
#      before anything is signalled;
#   3. kill goes one PID at a time, never a loop over an unverified list;
#   4. stop files first. kill is the last resort — a killed worker leaves an orphaned
#      python holding ~4 GB of VRAM while its lock is already released.
#
#   usage: safe_stop.sh lanes|clients|server|all [--force]
set -u
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
FW=$FW_ROOT
# What makes a process OURS: it is running out of THIS repository, or it holds one of our
# ports. The path arm is first because a server that has not bound its port yet still has
# to be recognisable. `$FW_REPO` is a literal in the regex, so a co-tenant campaign run
# from a different clone is correctly NOT matched.
OURS_RE="${FW_REPO}|--port 6033|--port 6035"
FORCE=0
[ "${2:-}" = "--force" ] && FORCE=1

# A PID is ours only if its own argv says so. Never trust the pattern that found it.
verify() {
  local pid=$1 argv
  argv=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  [ -z "$argv" ] && return 1
  # `--` before the pattern: without it grep parses a leading "--port" as an option and
  # ERRORS OUT. Chained with && that made the c1 exclusion silently never run — the hole
  # was in the guard itself. Caught by dry-running this script, 2026-08-04.
  echo "$argv" | grep -qE -- "$OURS_RE" || return 1
  # explicit exclusion: c1 lives on 6023/6024 and under eval_harness/
  echo "$argv" | grep -qE -- '(--port 602[34]|eval_harness/)' && return 1
  # never match a shell wrapper whose argv merely CONTAINS our pattern (this script, or
  # the `bash -c` block that invoked it). Same self-match trap as `pkill -f`.
  case "$argv" in *"bash -c"*|*safe_stop.sh*) return 1 ;; esac
  return 0
}

collect() {  # $1 = qualified pgrep pattern
  local pat=$1 pid
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    verify "$pid" && echo "$pid"
  done
}

show_and_kill() {  # $1=label  $2=qualified pattern
  local label=$1 pat=$2 pids pid
  pids=$(collect "$pat")
  if [ -z "$pids" ]; then echo "[$label] no matching processes of ours"; return 0; fi
  echo "[$label] candidates (verified ours):"
  for pid in $pids; do
    printf '    %s  %s\n' "$pid" "$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | head -c 100)"
  done
  if [ "$FORCE" -ne 1 ]; then
    echo "    (dry run — re-run with --force to signal these, one PID at a time)"
    return 0
  fi
  for pid in $pids; do
    echo "    kill $pid"
    kill "$pid" 2>/dev/null || echo "      (already gone)"
  done
}

case "${1:-}" in
  lanes)
    echo "preferred: touch $FW/stop-<WID>; kill only if a lane is wedged mid-retry"
    show_and_kill lanes 'fastwam-eval/scripts/fastwam_eval/worker.sh|fastwam_eval/worker.sh'
    ;;
  clients)  show_and_kill clients 'eval_policy_client.py --port 6033' ;;
  server)
    echo "preferred: touch $FW/stop-server-6033 (else the babysitter resurrects it in 30s)"
    show_and_kill server 'fastwam_model_server.py --port 6033'
    ;;
  all)
    show_and_kill lanes   'fastwam_eval/worker.sh'
    show_and_kill clients 'eval_policy_client.py --port 6033'
    show_and_kill server  'fastwam_model_server.py --port 6033'
    ;;
  *) echo "usage: $0 lanes|clients|server|all [--force]"; exit 2 ;;
esac

echo
