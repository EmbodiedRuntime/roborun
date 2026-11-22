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
| [`Ownership.tla`](Ownership.tla) | Ownership, arbitration, `take`, revocation (§3.3, §1.6) over a task tree of any depth: tasks that use a token held above them or ask for actuators themselves, and `hold` as a request of a running task; with the Gate in-process | The stop of an actuator given up while it moves (in `Gate.tla`) |
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
| `NoCommandWhenCancelled` | a cancelled task cannot move the robot, even through an ancestor's token or its own `hold` | yes |
| `CancelReachesSubtree` (`Ownership`) | §1.5 rule 1 with ownership: below a cancelled task nothing runs, waits or holds | yes |
| `FailNeverWaits` | §3.3 `fail`: fail at once | yes |
| `TakeDirect`, `OneTaker` | §3.3 `take`: ownership goes directly to the taker; at most one take waits for an actuator | yes |
| `TakeResolves` | a waiting take gets its actuators, or is cancelled or refused | yes |
| `NoStarvation` | §3.3 `wait`: no request starves | among `wait` requests made at start; tasks that keep taking starve them, as §3.3 says, and two tasks that each `hold` what the other owns wait for ever (§1.6) |
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

## The core against the models

The three ties above sample the code. A fourth is exhaustive, for the Gate's rules. They are
in [`roborun/core.py`](../src/roborun/core.py), a state machine with no clock, no backend and
no `await`; `LocalGate` is a shell around it. So the rules can be run through every order of
events, and [`spec/compare`](compare/) does that and compares the result with the state graph
TLC dumps for the same instance:

```sh
python spec/compare            # the quick set, about two minutes
python spec/compare --all      # with the full-size instances: over an hour, 10 GB
python spec/compare -v gate2   # one comparison, with the differences it finds
```

An environment stands for what the model describes besides the Gate: where each task is,
and whether the actuators move. It does to the core what the runtime does (a cancelled task
is revoked, a task that ends gives up what it owns, a cancelled task ends once the Gate says
its actuators are at rest), and maps the result to a state of the model. Two things are
checked:

- **the core does nothing the model forbids**: every step of the core, seen as model
  states, is the model action of the same name, possibly followed by the model serving a
  waiting request (`Gate.tla` does that in a step of its own, the core at once). So every
  invariant TLC proved holds for the core in this instance;
- **the model allows nothing the core cannot do**: every state of the core can take every
  step the model has from the state it is seen as, and every state of the model is met.
  This is asked of each state of the core, so two histories the model cannot tell apart
  must both be able to do what the model allows.

| Comparison | Core: states, steps | Model: states | Time |
|---|---|---|---|
| `gate` (3 tasks) | 213,992, 2.8 million | 53,143 | 98 s |
| `gate_allow` (3 tasks) | 940,217, 13.4 million | 161,759 | 8 min |
| `gate2`, `gate2_allow` (2 tasks) | 12,253 and 33,163 | 5,461 and 12,079 | 5 s, 13 s |
| `ownership` | 372,882, 2.3 million | 117,064 | 3 min |
| `ownership_takers` | 2,618,416, 16.3 million | 660,928 | 54 min |
| `ownership_child`, `ownership_deep` | 49,178 and 199,040 | 16,993 and 69,974 | 19 s, 94 s |
| `ownership_hold`, `ownership_hold_take` | 47,870 and 204,608 | 17,111 and 62,973 | 21 s, 97 s |

All agree. As a control, each of five mechanisms of `Gate.tla` is switched off in the model
only (its `gate_mut_*` config), and in the core only
([`mutations.py`](compare/mutations.py)): all ten show as differences. CI runs the quick
set: the two-task Gate instances, the controls, `ownership_child` and `ownership_hold`.

What this does not cover:

- `Ownership.tla` releases an actuator at once, moving or not, and the stop of an actuator
  given up while it moves is `Gate.tla`'s. So in the Ownership comparisons both sides give
  actuators up only at rest. The two mechanisms are compared separately, and their
  combination is in no model.
- The environments are a restatement of the runtime's side in Python, not the runtime:
  `task.py` is tied to `Tasks.tla` by the tests only.
- `Protocol.tla` and `Lockstep.tla` are about two processes, which the core is not.
  `RemoteGate`, the runtime's side of the Gate process, keeps a copy of two of the core's
  rules (what a cancelled child commanded is stopped, and it ends at rest); the tests cover
  it in both layouts.
- The shell (`LocalGate`), the checks, and the backends.

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

Comparing the core with `Ownership.tla` (above) found two more (v0.16), both about a child that
commands an actuator with its parent's token:

11. **A cancelled child could end while the arm it commanded was still braking.** A `stop`
    made the Gate forget whose command the arm was carrying out. A child cancelled a moment
    later had nothing to wait for and was `cancelled` with the arm moving, against
    `CancelledAtRest`. A `stop` no longer clears that record.
12. **A cancelled child waited for a motion its parent started later.** The Gate stops what
    a cancelled child commanded and lets it end at rest. If the parent sent a new goal while
    the child was still in its cleanup, the child waited for that motion too, and ended
    `failed` ("still moving") when it lasted more than 5 s. The model lets the child end as
    soon as the arm carries out someone else's command; so does the Gate now. The first
    form of the comparison missed this: it asked that each step of the model be taken from
    some state of the core, and this state was reachable by another history. It now asks it
    of every state.

`Ownership.tla` was then widened to trees of any depth and to `hold` (seven more configs).
The fourteen earlier configs reach exactly the same states as before, and the new ones found
no difference between the model and the Gate; they state one limit of `hold` (below).

## Known limits

- Tasks that keep taking starve a `wait` request (4 above; `ownership_limit_take_starves_wait`).
- `hold` asks while its task owns, so two tasks that each `hold` what the other owns wait
  for each other for ever (`ownership_limit_hold_deadlock`). A `wait_timeout` on the `hold`,
  a cancellation, or asking for everything at `start` ends or avoids it (§1.6).

Not modelled: a `take` or `fail` request that meets an actuator still stopping after it was
given up (`Gate.tla` has the stop, `Ownership.tla` the arbitration, neither both). Tests and
the random tests cover it on the code.

## Mutations

A property that cannot fail proves nothing. Each `*_mut_*` config switches one mechanism off
(reaping, propagation, reservations, the stop on revocation, cancelling waiting takes, the
two rules for a cancelled child, revoking the tokens of a cancelled task's `hold`s, refusing
a take that would cancel its own ancestor, the look at the grant after the checks, telling the
runtime of a revocation, acknowledging only when settled, the fairness assumptions) and must make TLC report the property that depends on it.
