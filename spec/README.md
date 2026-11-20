# Formal models

TLA+ models of the parts of [SPEC.md](../SPEC.md) that are discrete and owned by the runtime,
checked with TLC. They describe the design, not the Python code, and say nothing about
the physical world.

```sh
spec/check.sh            # every model; needs Java 11+, fetches TLC once into spec/.tools
spec/check.sh tasks      # only the configs whose name starts with "tasks"
```

Each line of the output is one config in [`cfg/`](cfg): `pass` configs must hold; the others
must report the named property violated. Everything takes about five minutes on 16 cores, and about
half an hour in CI, where it runs only when `spec/` or `SPEC.md` change.

| Module | Covers | Left out |
|---|---|---|
| [`Tasks.tla`](Tasks.tla) | States and transitions (§1.1, §1.2), the task tree (§1.5) | Resources, timeouts, replacement (§6) |
| [`Ownership.tla`](Ownership.tla) | Ownership, arbitration, `take`, revocation, a child using its parent's token (§3.3, §1.6), with the Gate in-process | Deeper trees, `hold` as a request of a running task |
| [`Gate.tla`](Gate.tla) | Checks that take time, locks (monitor, operator stop, fault), `allow`, fallback, reset, the setpoint watchdog (§3.4, §3.5), for one actuator | Arbitration (in `Ownership.tla`), what a check decides, budgets as times |
| [`Protocol.tla`](Protocol.tla) | The Gate in its own process: requests, replies and events in flight, the runtime's mirror of its grants, disconnect (§3.1, §3.8) | Several runtimes, message loss (the transport is reliable and ordered) |
| [`Lockstep.tla`](Lockstep.tla) | The tick/ack handshake (§2.1) | What a step computes |

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
| `ExecutedForTheOwner` (`Gate`, `Protocol`) | §3.4 rule 1, §3.3: a command reaches the backend only for the task that holds the actuator when the checks have answered, and never while it is locked | yes |
| `LockedHasNoOwner`, `NoWaiterUntilReset`, `OnlyResetUnlocks`, `MonitorLockFollowsMonitor` | §3.5: locks | yes |
| `FallbackRan`, `UnauthorizedChangesNothing` | §3.5: the fallback runs after a rejection or a safety event, and not for a command without a token | yes |
| `StopNeverWaits`, `NextRequestNeverWaits` | §3.4 rule 5: a stop never waits for a check, also across the process boundary | yes |
| `MirrorIsSound` | §3.1: the runtime believes in no token the Gate has not granted; a revocation it has not heard of is on its way | yes |
| `NoOrphanGrant` | no grant stays with a task that gave up its request | yes |
| `DisconnectStops`, `ReleasedAtRest` | §3.1: a closed connection loses its tokens; a revoked grant is released at rest | yes |
| `SimulatorWaits`, `OneTimeBase`, `RepliesInStep`, `TickIsComplete`, no deadlock | §2.1: lockstep | yes |
| `MotionHasOwner`, `GrantedAtRest`, `ReleaseSettles` | §3.3: whatever moves has an owner; an actuator is granted only at rest; one given up while moving is stopped, and passed on | yes |

"Yes" means TLC found no violation in every reachable state of small models (up to seven
tasks, two actuators, two sessions; three messages in flight each way), not a proof for
every size.

## From the models to the code

A model is not the code. Three things tie them:

- every difference a model showed is a test that failed before its fix:
  [`test_ownership_model.py`](../tests/test_ownership_model.py),
  [`test_gate_model.py`](../tests/test_gate_model.py),
  [`test_protocol_model.py`](../tests/test_protocol_model.py);
- [`test_gate_properties.py`](../tests/test_gate_properties.py) runs random programs (tasks
  three deep, `hold`, wait / fail / take from two sessions, cancellations, operator stops)
  against the real Gate, in-process and in its own process, and evaluates the invariants
  above on the Gate's state after every step, under the same names;
- `roborun trace check run.jsonl` replays over a recorded run the invariants that can be
  read from a trace: `ParentOutlivesChildren`, `OwnershipConsistent`,
  `NoCommandWhenCancelled`, and no command accepted while locked until reset.

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

`Gate.tla` and `Protocol.tla` showed four more (v0.14):

6. A command whose grant was released while its checks ran was still executed: a child
   that sends through its parent's `hold`, the parent leaves the `hold`, the arm goes to
   another task, and the command then moves it. The Gate now looks at the grant again.
7. A fault left the requests waiting for the actuator in place until `reset`; an operator
   stop ends them. Both now end them.
8. With the Gate in its own process, a request cancelled while its grant was on the way
   kept the grant: the runtime had dropped the answer, and the actuator stayed with a
   cancelled task. Withdrawing a request now releases what it was granted meanwhile.
9. The Gate process served one request at a time, so a `stop` waited behind the checks of
   an earlier command, against §3.4 rule 5. Commands are now checked beside the other
   requests; those for one actuator keep their order.

The random tests then showed something the models had stated only as a limit (v0.15):

10. **A task that ended normally could leave its actuator moving.** It sent a goal and did
    not wait. Nobody owned that motion, the next owner was granted a moving actuator, and
    a task granted it and cancelled before it ran was `cancelled` while it moved (8 of 4000
    seeds). The Gate now stops an actuator given up while it moves and passes it on at
    rest; `MotionHasOwner` and `GrantedAtRest` hold, and `gate_mut_nostoponrelease` is the
    Gate before the change.

## Known limits

- Tasks that keep taking starve a `wait` request (4 above; `ownership_limit_take_starves_wait`).

Not modelled: trees deeper than one level together with ownership, and `hold` as a request
made by a running task. The random tests cover both on the code.

## Mutations

A property that cannot fail proves nothing. Each `*_mut_*` config switches one mechanism off
(reaping, propagation, reservations, the stop on revocation, cancelling waiting takes, the
two rules for a cancelled child, the look at the grant after the checks, telling the
runtime of a revocation, acknowledging only when settled, the fairness assumptions) and must make TLC report the property that depends on it.
