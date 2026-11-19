#!/usr/bin/env bash
# Model-check the TLA+ models of the runtime with TLC.
#
#   spec/check.sh              run every entry of the manifest below
#   spec/check.sh tasks        run the entries whose config name starts with "tasks"
#
# Each manifest line is: <module> <config> <expectation>. The expectation is "pass", or the
# name of the property TLC must report violated: mutations (a mechanism switched off) and
# limits that SPEC.md states. See spec/README.md.
set -u
cd "$(dirname "$0")"

TLC_VERSION=1.7.4
JAR=.tools/tla2tools-$TLC_VERSION.jar
if [ ! -f "$JAR" ]; then
  mkdir -p .tools
  curl -fsSL -o "$JAR" \
    "https://github.com/tlaplus/tlaplus/releases/download/v$TLC_VERSION/tla2tools.jar" || {
    echo "could not download TLC $TLC_VERSION" >&2; exit 2; }
fi

MANIFEST='
MCTasks      tasks                        pass
MCTasks      tasks_live                   pass
MCTasks      tasks_mut_noreap             ParentOutlivesChildren
MCTasks      tasks_mut_nopropagate        CancelReachesChildren
MCTasks      tasks_mut_nofair             CancelTerminates
MCOwnership  ownership                    pass
MCOwnership  ownership_takers             pass
MCOwnership  ownership_child              pass
MCOwnership  ownership_live               pass
MCOwnership  ownership_child_live         pass
MCOwnership  ownership_takers_live        pass
MCOwnership  ownership_mut_noreserve      NoOvertake
MCOwnership  ownership_mut_nofair         CancelTerminates
MCOwnership  ownership_mut_noconfirmstop  CancelledAtRest
MCOwnership  ownership_mut_failowned      FailNeverWaits
MCOwnership  ownership_mut_takefront      TakeDirect
MCOwnership  ownership_mut_childcommands  NoCommandWhenCancelled
MCOwnership  ownership_mut_childmoving    CancelledAtRest
MCOwnership  ownership_limit_take_starves_wait  NoStarvation
'

filter="${1:-}"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
failed=0
while read -r module cfg expect; do
  [ -z "$module" ] && continue
  case "$cfg" in "$filter"*) ;; *) continue ;; esac
  log="$out/$cfg.log"
  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC "$module" -config "cfg/$cfg.cfg" \
    -workers auto -metadir "$out/$cfg" >"$log" 2>&1
  code=$?
  states=$(grep -o '[0-9]* distinct states found' "$log" | tail -1)
  if [ "$expect" = pass ]; then
    if [ $code -eq 0 ]; then verdict=ok; else verdict=FAILED; fi
    what="holds"
  else
    # 12: a safety property is violated; 13: a liveness property is.
    if { [ $code -eq 12 ] || [ $code -eq 13 ]; } &&
       { grep -q "$expect is violated" "$log" || grep -q "Temporal properties were violated" "$log"; }
    then verdict=ok; else verdict=FAILED; fi
    what="violates $expect, as expected"
  fi
  printf '%-7s %-34s %-44s %s\n' "$verdict" "$cfg" "$what" "$states"
  if [ $verdict = FAILED ]; then
    failed=1
    echo "---- $cfg (exit $code) ----"; tail -n 60 "$log"; echo "----"
  fi
done <<<"$MANIFEST"
exit $failed
