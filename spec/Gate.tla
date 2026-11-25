-------------------------------- MODULE Gate --------------------------------
(* What stands between a command and the backend: SPEC.md 3.4 and 3.5, for one actuator.

   Checks take time: a command is admitted in two steps, and anything may happen between
   them. Around that: locks (a monitor, an operator stop, a fault), the fallback, reset, and
   the watchdog of a setpoint stream. Arbitration between requests is in Ownership.tla;
   here a waiting request is served in any order.

   A grant goes  none -> valid -> revoked -> released,  or  valid -> released  when its task
   ends or leaves a `hold` at rest without being cancelled. Given up while the actuator
   moves, it is revoked instead: the Gate stops the actuator and keeps the grant, with no
   task behind it, until it is at rest. So the actuator changes hands only at rest. A task
   whose grant was released may still have a command in flight: a child that sends through
   its parent's `hold`, or a `send` the body did not wait for.

   Switches, each turning off one mechanism (see Tasks.tla):
     Recheck         after the checks, the grant must still be valid and no lock new
     RecheckRelease  ... and must not have been released
     FaultRejects    a fault ends the requests waiting for the actuator, as an operator
                     stop does
     UnauthFallback  (a mutation when TRUE) a command without a valid token runs the fallback
     StopOnRelease   a grant given up while the actuator moves is stopped and kept until rest
   Allow: the monitor has an `allow` check, so its lock lets through the commands that
   check accepts (e.g. retreating from a person), and the actuator can be acquired. *)
EXTENDS FiniteSets, Naturals

CONSTANTS Tasks, None, Allow, Recheck, RecheckRelease, FaultRejects, UnauthFallback,
          StopOnRelease

ASSUME /\ None \notin Tasks
       /\ {Allow, Recheck, RecheckRelease, FaultRejects, UnauthFallback, StopOnRelease}
            \subseteq BOOLEAN

VARIABLES ts,       \* ts[t]: idle, waiting, owning, cancelling, done
          grant,    \* grant[t]: none, valid, revoked, released
          pending,  \* pending[t]: None, or the command whose checks are running
          locks,    \* the causes for which the actuator is locked
          failing,  \* the monitor rejects
          motion,   \* rest, moving, stopping
          watch,    \* the watchdog of a setpoint stream is armed
          step      \* the last thing that happened, and for which task

vars == <<ts, grant, pending, locks, failing, motion, watch, step>>

Causes == {"monitor", "operator", "fault"}
Manual == {"operator", "fault"}   \* unlocked only by reset
Kinds  == {"goal", "setpoint"}

(* Lock causes that no command passes. *)
Hard == IF Allow THEN locks \ {"monitor"} ELSE locks

Holders == {t \in Tasks : grant[t] \in {"valid", "revoked"}}
Free    == Holders = {} /\ Hard = {}

Did(a, t) == step' = [a |-> a, t |-> t]

(* The fallback, and any stop: the actuator brakes, and no stream is expected. *)
Braked == IF motion = "moving" THEN "stopping" ELSE motion

(* Cancel the tasks in S: tokens are revoked first (3.1), and the Gate stops the actuator
   itself. A waiting request is withdrawn. *)
CancelledTs(S) == [t \in Tasks |->
                     IF t \in S /\ ts[t] = "owning" THEN "cancelling"
                     ELSE IF t \in S /\ ts[t] = "waiting" THEN "done"
                     ELSE ts[t]]
RevokedGrants(S) == [t \in Tasks |-> IF t \in S /\ grant[t] = "valid" THEN "revoked" ELSE grant[t]]

Owners  == {t \in Tasks : grant[t] = "valid"}
Waiters == {t \in Tasks : ts[t] = "waiting"}

-----------------------------------------------------------------------------

Init == /\ ts = [t \in Tasks |-> "idle"]
        /\ grant = [t \in Tasks |-> "none"]
        /\ pending = [t \in Tasks |-> None]
        /\ locks = {} /\ failing = FALSE
        /\ motion = "rest" /\ watch = FALSE
        /\ step = [a |-> "init", t |-> None]

(* A request for the actuator (3.5): fails at once while it is locked until reset. *)
Acquire(t) == /\ ts[t] = "idle"
              /\ IF locks \cap Manual # {}
                 THEN /\ ts' = [ts EXCEPT ![t] = "done"]
                      /\ UNCHANGED grant /\ Did("busy", t)
                 ELSE IF Free /\ Waiters = {}
                 THEN /\ ts' = [ts EXCEPT ![t] = "owning"]
                      /\ grant' = [grant EXCEPT ![t] = "valid"] /\ Did("grant", t)
                 ELSE /\ ts' = [ts EXCEPT ![t] = "waiting"]
                      /\ UNCHANGED grant /\ Did("wait", t)
              /\ UNCHANGED <<pending, locks, failing, motion, watch>>

Dispatch(t) == /\ ts[t] = "waiting" /\ Free
               /\ ts' = [ts EXCEPT ![t] = "owning"]
               /\ grant' = [grant EXCEPT ![t] = "valid"]
               /\ Did("grant", t)
               /\ UNCHANGED <<pending, locks, failing, motion, watch>>

(* A command arrives. Without a valid token it is refused and nothing else happens (3.5):
   otherwise a task could stop another task's motion. *)
Unauthorized(t) == /\ grant[t] # "valid"
                   /\ Did("unauthorized", t)
                   /\ motion' = IF UnauthFallback THEN Braked ELSE motion
                   /\ watch' = IF UnauthFallback THEN FALSE ELSE watch
                   /\ UNCHANGED <<ts, grant, pending, locks, failing>>

(* With a valid token: refused while locked; under a monitor's lock its `allow` check is
   asked, and may refuse; then the checks start. *)
Send(t, k) == /\ grant[t] = "valid" /\ pending[t] = None
              /\ \/ /\ Hard = {}
                    /\ pending' = [pending EXCEPT ![t] = [kind |-> k, passed |-> locks]]
                    /\ Did("checking", t)
                 \/ /\ Hard # {} \/ "monitor" \in locks
                    /\ UNCHANGED pending /\ Did("locked", t)
              /\ UNCHANGED <<ts, grant, locks, failing, motion, watch>>

(* What the Gate looks at again when the checks have answered. *)
StillAllowed(t) == /\ Recheck => grant[t] # "revoked" /\ locks \subseteq pending[t].passed
                   /\ RecheckRelease => grant[t] # "released"

(* Every check accepted. The command reaches the backend, which may report a fault. *)
Accepted(t) == /\ pending[t] # None
               /\ pending' = [pending EXCEPT ![t] = None]
               /\ IF ~StillAllowed(t)
                  THEN Did("unauthorized", t) /\ UNCHANGED <<ts, grant, locks, failing, motion, watch>>
                  ELSE \/ /\ motion' = "moving"
                          /\ watch' = (pending[t].kind = "setpoint")
                          /\ Did("execute", t)
                          /\ UNCHANGED <<ts, grant, locks, failing>>
                       \/ /\ locks' = locks \cup {"fault"}
                          /\ LET out == {t} \cup (IF FaultRejects THEN Waiters ELSE {})
                             IN  ts' = CancelledTs(out) /\ grant' = RevokedGrants(out)
                          /\ motion' = Braked /\ watch' = FALSE
                          /\ Did("fault", t)
                          /\ UNCHANGED failing

(* The command is over without an answer from the checks: its sender went away while they
   ran. Nothing reaches the backend and nobody is cancelled. *)
Drop(t) == /\ pending[t] # None
           /\ pending' = [pending EXCEPT ![t] = None]
           /\ Did("drop", t)
           /\ UNCHANGED <<ts, grant, locks, failing, motion, watch>>

(* A check rejected, timed out, raised, or had a stale input: the same thing (fail closed).
   The fallback runs, and the task that sent the command is cancelled. *)
Rejected(t) == /\ pending[t] # None
               /\ pending' = [pending EXCEPT ![t] = None]
               /\ ts' = CancelledTs({t}) /\ grant' = RevokedGrants({t})
               /\ motion' = Braked /\ watch' = FALSE
               /\ Did("reject", t)
               /\ UNCHANGED <<locks, failing>>

(* stop is never checked and needs no token. *)
Stop == /\ motion' = Braked /\ watch' = FALSE
        /\ Did("stop", None)
        /\ UNCHANGED <<ts, grant, pending, locks, failing>>

Rest == /\ motion = "stopping" /\ motion' = "rest"
        /\ Did("rest", None)
        /\ UNCHANGED <<ts, grant, pending, locks, failing, watch>>

(* The motion reaches its goal by itself. A stream may still be fed at rest. *)
Arrive == /\ motion = "moving" /\ motion' = "rest"
          /\ Did("arrive", None)
          /\ UNCHANGED <<ts, grant, pending, locks, failing, watch>>

(* A cancellation from the runtime. *)
Cancel(t) == /\ ts[t] \in {"owning", "waiting"}
             /\ ts' = CancelledTs({t}) /\ grant' = RevokedGrants({t})
             /\ motion' = IF ts[t] = "owning" THEN Braked ELSE motion
             /\ watch' = IF ts[t] = "owning" THEN FALSE ELSE watch
             /\ Did("cancel", t)
             /\ UNCHANGED <<pending, locks, failing>>

(* The task ends, or leaves its `hold`, with a valid grant. If the actuator is not at
   rest (a goal it did not wait for), the Gate stops it and keeps the grant until it is. *)
Release(t) == /\ ts[t] = "owning"
              /\ ts' = [ts EXCEPT ![t] = "done"]
              /\ watch' = FALSE
              /\ IF StopOnRelease /\ motion # "rest"
                 THEN /\ grant' = [grant EXCEPT ![t] = "revoked"]
                      /\ motion' = Braked /\ Did("release-stop", t)
                 ELSE /\ grant' = [grant EXCEPT ![t] = "released"]
                      /\ UNCHANGED motion /\ Did("release", t)
              /\ UNCHANGED <<pending, locks, failing>>

(* The grant kept by Release is given up at rest. *)
Settle(t) == /\ ts[t] = "done" /\ grant[t] = "revoked" /\ motion = "rest"
             /\ grant' = [grant EXCEPT ![t] = "released"]
             /\ Did("settle", t)
             /\ UNCHANGED <<ts, pending, locks, failing, motion, watch>>

(* A cancelled task ends, and its grant is released, only at rest (1.1). *)
End(t) == /\ ts[t] = "cancelling" /\ motion = "rest"
          /\ ts' = [ts EXCEPT ![t] = "done"]
          /\ grant' = [grant EXCEPT ![t] = "released"]
          /\ Did("end", t)
          /\ UNCHANGED <<pending, locks, failing, motion, watch>>

(* A safety event: lock, cancel the owners, run the fallback (3.5). *)
MonitorFires == /\ ~failing /\ failing' = TRUE
                /\ locks' = locks \cup {"monitor"}
                /\ ts' = CancelledTs(Owners) /\ grant' = RevokedGrants(Owners)
                /\ motion' = Braked /\ watch' = FALSE
                /\ Did("monitor", None)
                /\ UNCHANGED pending

MonitorAccepts == /\ failing /\ failing' = FALSE
                  /\ locks' = locks \ {"monitor"}
                  /\ Did("unlock", None)
                  /\ UNCHANGED <<ts, grant, pending, motion, watch>>

OperatorStop == /\ locks' = locks \cup {"operator"}
                /\ ts' = CancelledTs(Owners \cup Waiters)
                /\ grant' = RevokedGrants(Owners)
                /\ motion' = Braked /\ watch' = FALSE
                /\ Did("operator", None)
                /\ UNCHANGED <<pending, failing>>

Reset == /\ locks \cap Manual # {}
         /\ locks' = locks \ Manual
         /\ Did("reset", None)
         /\ UNCHANGED <<ts, grant, pending, failing, motion, watch>>

(* The stream stopped: its owner is cancelled and the fallback runs. *)
Watchdog == /\ watch /\ watch' = FALSE
            /\ ts' = CancelledTs(Owners) /\ grant' = RevokedGrants(Owners)
            /\ motion' = Braked
            /\ Did("watchdog", None)
            /\ UNCHANGED <<pending, locks, failing>>

Next == \/ \E t \in Tasks : \/ Acquire(t) \/ Dispatch(t) \/ Unauthorized(t) \/ Accepted(t)
                            \/ Rejected(t) \/ Drop(t) \/ Cancel(t) \/ Release(t) \/ End(t) \/ Settle(t)
                            \/ \E k \in Kinds : Send(t, k)
        \/ Stop \/ Rest \/ Arrive \/ MonitorFires \/ MonitorAccepts \/ OperatorStop \/ Reset \/ Watchdog

Spec == Init /\ [][Next]_vars

(* For liveness: the actuator comes to rest, checks answer within their budget, and the
   runtime ends a task that can end. *)
Fairness == /\ WF_vars(Rest)
            /\ \A t \in Tasks : /\ WF_vars(Accepted(t) \/ Rejected(t))
                                /\ WF_vars(End(t)) /\ WF_vars(Settle(t))

FairSpec == Spec /\ Fairness

-----------------------------------------------------------------------------
(* Properties *)

TypeOK == /\ ts \in [Tasks -> {"idle", "waiting", "owning", "cancelling", "done"}]
          /\ grant \in [Tasks -> {"none", "valid", "revoked", "released"}]
          /\ pending \in [Tasks -> {None} \cup [kind : Kinds, passed : SUBSET Causes]]
          /\ locks \subseteq Causes /\ failing \in BOOLEAN
          /\ motion \in {"rest", "moving", "stopping"} /\ watch \in BOOLEAN

OneHolder == Cardinality(Holders) <= 1

(* 3.4 rule 1 and 3.3, across the time checks take: a command reaches the backend only
   for the task that holds the actuator now, and never while it is locked. *)
ExecutedForTheOwner == step.a = "execute" => grant[step.t] = "valid" /\ Hard = {}

(* 3.5: after a safety event nobody holds a valid token for the actuator. *)
LockedHasNoOwner == Hard # {} => Owners = {}

(* 3.5: nothing waits for an actuator that is locked until reset. *)
NoWaiterUntilReset == locks \cap Manual # {} => Waiters = {}

(* 3.5: the monitor's lock is there exactly while the monitor rejects. *)
MonitorLockFollowsMonitor == ("monitor" \in locks) <=> failing

(* 3.5: after a rejection, a monitor, an operator stop, a watchdog or a fault, the
   actuator is not left moving, and no stream is expected. *)
FallbackRan == step.a \in {"reject", "monitor", "operator", "watchdog", "fault"} =>
                 motion # "moving" /\ ~watch

(* The watchdog watches a stream of the current owner. *)
WatchHasOwner == watch => Owners # {}

(* 3.5: a command without a valid token changes nothing. *)
UnauthorizedChangesNothing ==
  [][step'.a = "unauthorized" => motion' = motion /\ watch' = watch]_vars

(* 3.5: only reset removes an operator's or a fault's lock. *)
OnlyResetUnlocks ==
  [][(locks \cap Manual) \ locks' # {} => step'.a = "reset"]_vars

(* 3.4 rule 5: a stop is possible whatever the checks are doing. *)
StopNeverWaits == ENABLED Stop

(* 3.3: whatever moves has an owner, and the actuator is granted only at rest. Without
   StopOnRelease a task that ends with a goal on its way leaves a motion nobody owns. *)
MotionHasOwner == motion = "moving" => Owners # {}
GrantedAtRest  == step.a = "grant" => motion = "rest"

(* Every grant kept until rest is given up. *)
ReleaseSettles == \A t \in Tasks : (grant[t] = "revoked") ~> (grant[t] = "released")

(* Every check answers, and every cancellation ends. *)
ChecksAnswer     == \A t \in Tasks : (pending[t] # None) ~> (pending[t] = None)
CancelTerminates == \A t \in Tasks : (ts[t] = "cancelling") ~> (ts[t] = "done")

=============================================================================
