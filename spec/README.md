# Formal models

TLA+ models of the parts of [SPEC.md](../SPEC.md) that are discrete and owned by the runtime,
checked with TLC. They describe the design, not the Python code, and say nothing about
the physical world.

```sh
spec/check.sh            # every model; needs Java 11+, fetches TLC once into spec/.tools
spec/check.sh tasks      # only the configs whose name starts with "tasks"
```

Each line of the output is one config in [`cfg/`](cfg): `pass` configs must hold; the others
must report the named property violated. Everything takes about three minutes on 16 cores, and about
half an hour in CI, where it runs only when `spec/` or `SPEC.md` change.

| Module | Covers | Left out |
|---|---|---|
| [`Tasks.tla`](Tasks.tla) | States and transitions (§1.1, §1.2), the task tree (§1.5) | Resources, timeouts, replacement (§6) |
| [`Ownership.tla`](Ownership.tla) | Ownership, arbitration, `take`, revocation, a child using its parent's token (§3.3, §1.6), with the Gate in-process | Deeper trees, `hold` as a request of a running task, checks, locks, the process boundary |

## Properties

| Property | SPEC | Holds |
|---|---|---|
| `LegalTransitions` | §1.2: no other transition; a terminal state never changes | yes |
| `ParentOutlivesChildren`, `NoLateChild` | §1.5 rules 2 and 3 | yes |
| `CancelReachesChildren` | §1.5 rule 1 | yes |
| `CancelTerminates` (both modules) | every cancellation ends | yes, if a cancelled body ends and a stop comes to rest |
| `OwnershipConsistent`, `QueueConsistent`, `RevokedIsCancelling` | §3.3: one owner; nothing leaked; a cancelled task has no valid token | yes |
| `NoOvertake` | §3.3 `wait`: a later request never overtakes an earlier one it overlaps | yes |
| `CancelledAtRest` | §1.1: `cancelled` means the robot is stopped | yes |
| `NoCommandWhenCancelled` | a cancelled task cannot move the robot, even through its parent's token | yes |
| `FailNeverWaits` | §3.3 `fail`: fail at once | yes |
| `TakeDirect`, `OneTaker` | §3.3 `take`: ownership goes directly to the taker; at most one take waits for an actuator | yes |
| `TakeResolves` | a waiting take gets its actuators, or is cancelled or refused | yes |
| `NoStarvation` | §3.3 `wait`: no request starves | among `wait` requests; tasks that keep taking starve them, as §3.3 says |

"Yes" means TLC found no violation in every reachable state of small models (up to seven
tasks, two actuators, two sessions), not a proof for every size.

## What the models found

The models showed five differences between SPEC.md and the implementation; each was reproduced
against `LocalGate`, then fixed, with tests in
[`tests/test_ownership_model.py`](../tests/test_ownership_model.py).

1. A task could become `cancelled` while its actuator moved: stopping was left to the
   skill's own cleanup. Revoking a token now stops its actuators, and a cancelling task
   ends, and its actuators are handed on, only at rest.
2. A `fail` request waited when its actuator was free but reserved for an earlier waiter.
3. A second `take` went ahead of a waiting one, so a supervised session could keep the arm
   from its supervisor. No order of two waiting takes satisfies "directly to the taker"
   for both, so a `take` now cancels the takes already waiting, with authority over them.
4. Tasks that keep taking starve a `wait` request. This stays; §3.3 now says so.
5. A cancelled child that used its parent's token (a task started inside `hold`) could
   still command the arm, and became `cancelled` while the motion it had started went on.
   This one was seen while fixing the first, then modelled. The Gate now rejects its
   commands and stops that motion.

Not modelled yet: trees deeper than one level together with ownership, `hold` as a
request made by a running task, checks, locks and monitors, the process boundary, and
lockstep.

## Mutations

A property that cannot fail proves nothing. Each `*_mut_*` config switches one mechanism off
(reaping, propagation, reservations, the stop on revocation, cancelling waiting takes, the
two rules for a cancelled child, the fairness assumptions) and must make TLC report the property that depends on it.
