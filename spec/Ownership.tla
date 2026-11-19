----------------------------- MODULE Ownership -----------------------------
(* Control authority over actuators: SPEC.md 3.3 and 1.6, with the Gate in the same
   process as the tasks (every step below is atomic). Of the task tree (Tasks.tla) it keeps
   one level: a child that uses its parent's token, as a task started inside `hold` does.
   Checks, locks and monitors come with Gate.tla.

   A task asks for all its actuators when it starts. Requests wait in one queue, served
   first come, first served; a `take` cancels the owners and the takes already waiting,
   and goes to the front. A child asks for nothing: it commands its parent's actuators.

   Switches. Each switches one mechanism off, to show that a property is not vacuous; the
   implementation has them all on.
     TakeFront      a `take` request goes to the front of the queue
     TakeCancels    a `take` also cancels the takes already waiting for one of its
                    actuators, and needs authority over them as over the owners
     ConfirmStop    revoking a task's token also stops its actuators, and a cancelling
                    task releases them only at rest (SPEC 1.1)
     FailOnReserved a `fail` request also fails when an earlier request waits for one of
                    its actuators, not only when one is owned
     Reserve        a waiter reserves its actuators against later requests
     ChildReject    a command from a cancelled child is rejected, though its parent's
                    token is still valid
     ChildStop      cancelling a child stops the motion it commanded, and the child ends
                    only once that motion is at rest                                   *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Tasks, Actuators, None,
          Needs,        \* Needs[t]: the actuators t declares
          Mode,         \* Mode[t]: on_busy, one of "wait", "fail", "take"
          Session,      \* Session[t]: the principal for authority
          Supervises,   \* pairs <<s1, s2>>: session s1 may take from session s2
          Parent,       \* Parent[t]: the task whose token t uses, or None
          Repeating,    \* tasks that are started again after they end
          Impatient,    \* tasks that wait with a wait_timeout
          TakeFront, TakeCancels, ConfirmStop, FailOnReserved, Reserve, ChildReject, ChildStop

ASSUME /\ \A t \in Tasks : Needs[t] \subseteq Actuators /\ Needs[t] # {}
       /\ \A t \in Tasks : Mode[t] \in {"wait", "fail", "take"}
       /\ Repeating \subseteq Tasks /\ Impatient \subseteq Tasks
       /\ TakeFront \in BOOLEAN /\ TakeCancels \in BOOLEAN /\ ConfirmStop \in BOOLEAN
       /\ FailOnReserved \in BOOLEAN /\ Reserve \in BOOLEAN
       /\ ChildReject \in BOOLEAN /\ ChildStop \in BOOLEAN
       /\ \A t \in Tasks : Parent[t] = None \/ (Parent[t] \in Tasks /\ Parent[Parent[t]] = None)

VARIABLES st,       \* st[t]: "idle", "waiting" (pending, in the queue), "running",
                    \*        "cancelling", "terminal"
          out,      \* out[t]: how t ended: "none", "succeeded", "failed", "cancelled"
          owner,    \* owner[a]: the task that owns a, or None
          revoked,  \* the tasks whose token is revoked
          queue,    \* the waiting requests, in order
          motion,   \* motion[a]: "rest", "moving", "braking"
          last      \* last[a]: the child whose command a is carrying out; None at rest
                    \*          or when its owner commanded it

vars == <<st, out, owner, revoked, queue, motion, last>>

IsChild(t) == Parent[t] # None
Kids(P) == {c \in Tasks : Parent[c] \in P}

InQueue(t) == \E i \in 1..Len(queue) : queue[i] = t
Pos(t) == CHOOSE i \in 1..Len(queue) : queue[i] = t
Without(q, t) == SelectSeq(q, LAMBDA x : x # t)
Owned(t) == {a \in Actuators : owner[a] = t}

(* Authority to take (3.3): the same session, or a supervisor of the owner's session. *)
MayTake(x, o) == Session[x] = Session[o] \/ <<Session[x], Session[o]>> \in Supervises

(* Serve the queue in order. A request is granted when all its actuators are free and no
   earlier request that is still waiting needs one of them. Returns <<owner, granted>>. *)
RECURSIVE Serve(_, _, _, _)
Serve(q, own, reserved, granted) ==
  IF q = <<>> THEN <<own, granted>>
  ELSE LET x == Head(q)
           free == \A a \in Needs[x] : own[a] = None
       IN IF free /\ Needs[x] \cap reserved = {}
          THEN Serve(Tail(q), [a \in Actuators |-> IF a \in Needs[x] THEN x ELSE own[a]],
                     reserved, granted \cup {x})
          ELSE Serve(Tail(q), own, IF Reserve THEN reserved \cup Needs[x] ELSE reserved,
                     granted)

(* The state after serving queue q0 with ownership own0 and task states st0. *)
Settle(q0, own0, st0) ==
  LET r == Serve(q0, own0, {}, {})
  IN /\ owner' = r[1]
     /\ queue' = SelectSeq(q0, LAMBDA x : x \notin r[2])
     /\ st' = [t \in Tasks |-> IF t \in r[2] THEN "running" ELSE st0[t]]

(* With ConfirmStop, revoking the tokens of the tasks in P stops what they were moving. *)
Stopped(P) == [a \in Actuators |->
                 IF ConfirmStop /\ owner[a] \in P /\ motion[a] = "moving"
                 THEN "braking" ELSE motion[a]]

-----------------------------------------------------------------------------

Init == /\ st = [t \in Tasks |-> "idle"]
        /\ out = [t \in Tasks |-> "none"]
        /\ owner = [a \in Actuators |-> None]
        /\ revoked = {}
        /\ queue = <<>>
        /\ motion = [a \in Actuators |-> "rest"]
        /\ last = [a \in Actuators |-> None]

Refuse(t) == /\ st' = [st EXCEPT ![t] = "terminal"]
             /\ out' = [out EXCEPT ![t] = "failed"]
             /\ UNCHANGED <<owner, revoked, queue, motion, last>>

(* start(): the task asks for its actuators (3.3). *)
Request(t) ==
  /\ st[t] = "idle" /\ ~IsChild(t)
  /\ LET owners == {owner[a] : a \in Needs[t]} \ {None}
         ahead == {x \in Tasks : InQueue(x) /\ Needs[x] \cap Needs[t] # {}}
         takers == IF TakeCancels THEN {x \in ahead : Mode[x] = "take"} ELSE {}
     IN IF Mode[t] = "fail" /\ (owners # {} \/ (FailOnReserved /\ ahead # {}))
        THEN Refuse(t)                                  \* ResourceBusy
        ELSE IF Mode[t] = "take" /\ \E o \in owners \cup takers : ~MayTake(t, o)
        THEN Refuse(t)                                  \* no authority; nothing is cancelled
        ELSE LET pre == IF Mode[t] = "take" THEN owners ELSE {}
                 out0 == IF Mode[t] = "take" THEN takers ELSE {}
                 st0 == [x \in Tasks |->
                           IF x = t THEN "waiting"
                           ELSE IF x \in out0 THEN "terminal"   \* it has had no effect
                           ELSE IF x \in pre \cup Kids(pre) /\ st[x] = "running"
                                THEN "cancelling"       \* cancellation reaches the children
                           ELSE st[x]]
                 rest == SelectSeq(queue, LAMBDA x : x \notin out0)
                 q0 == IF Mode[t] = "take" /\ TakeFront
                       THEN <<t>> \o rest ELSE Append(rest, t)
             IN /\ Settle(q0, owner, st0)
                /\ revoked' = revoked \cup pre          \* revoke first, then cancel (3.1)
                /\ motion' = Stopped(pre)
                /\ out' = [x \in Tasks |-> IF x \in out0 THEN "cancelled" ELSE out[x]]
                /\ UNCHANGED last

(* A running task starts a child, which uses the parent's token. *)
Spawn(c) ==
  /\ IsChild(c) /\ st[c] = "idle" /\ st[Parent[c]] = "running"
  /\ st' = [st EXCEPT ![c] = "running"]
  /\ UNCHANGED <<out, owner, revoked, queue, motion, last>>

(* cancel(): a waiting task has had no effect and ends at once; a running one starts
   cancelling, and its token is revoked. *)
Cancel(t) ==
  \/ /\ st[t] = "waiting"
     /\ Settle(Without(queue, t), owner, [st EXCEPT ![t] = "terminal"])
     /\ out' = [out EXCEPT ![t] = "cancelled"]
     /\ UNCHANGED <<revoked, motion, last>>
  \/ /\ st[t] = "running" /\ ~IsChild(t)
     /\ st' = [x \in Tasks |-> IF x \in {t} \cup Kids({t}) /\ st[x] = "running"
                               THEN "cancelling" ELSE st[x]]
     /\ revoked' = revoked \cup {t}
     /\ motion' = Stopped({t})
     /\ UNCHANGED <<out, owner, queue, last>>
  \/ /\ st[t] = "running" /\ IsChild(t)       \* it has no token of its own to revoke
     /\ st' = [st EXCEPT ![t] = "cancelling"]
     /\ motion' = [a \in Actuators |->
                     IF ChildStop /\ last[a] = t /\ motion[a] = "moving"
                     THEN "braking" ELSE motion[a]]
     /\ UNCHANGED <<out, owner, revoked, queue, last>>

(* wait_timeout expires: ResourceBusy. *)
Timeout(t) ==
  /\ t \in Impatient /\ st[t] = "waiting"
  /\ Settle(Without(queue, t), owner, [st EXCEPT ![t] = "terminal"])
  /\ out' = [out EXCEPT ![t] = "failed"]
  /\ UNCHANGED <<revoked, motion, last>>

(* A goal or setpoint from t for a: forwarded only with a valid token. A child uses its
   parent's; once it is cancelled, the Gate no longer forwards what it sends. *)
Command(t, a) ==
  /\ st[t] \in {"running", "cancelling"}
  /\ IF IsChild(t)
     THEN /\ a \in Needs[t]
          /\ owner[a] = Parent[t] /\ Parent[t] \notin revoked
          /\ ChildReject => st[t] = "running"
     ELSE owner[a] = t /\ t \notin revoked
  /\ motion' = [motion EXCEPT ![a] = "moving"]
  /\ last' = [last EXCEPT ![a] = IF IsChild(t) THEN t ELSE None]
  /\ UNCHANGED <<st, out, owner, revoked, queue>>

(* stop: always accepted, also from a task whose token is revoked. *)
TaskStop(t, a) ==
  /\ st[t] \in {"running", "cancelling"}
  /\ a \in Needs[t] /\ motion[a] = "moving"
  /\ motion' = [motion EXCEPT ![a] = "braking"]
  /\ UNCHANGED <<st, out, owner, revoked, queue, last>>

(* The backend: a motion reaches its goal, or a stop comes to rest. *)
Arrive(a) == /\ motion[a] = "moving"
             /\ motion' = [motion EXCEPT ![a] = "rest"]
             /\ last' = [last EXCEPT ![a] = None]
             /\ UNCHANGED <<st, out, owner, revoked, queue>>

Brake(a) == /\ motion[a] = "braking"
            /\ motion' = [motion EXCEPT ![a] = "rest"]
            /\ last' = [last EXCEPT ![a] = None]
            /\ UNCHANGED <<st, out, owner, revoked, queue>>

(* The task ends and its actuators are released. *)
End(t, o) ==
  /\ st[t] \in {"running", "cancelling"}
  /\ o \in (IF st[t] = "running" THEN {"succeeded", "failed"}
                                 ELSE {"cancelled", "succeeded", "failed"})
  /\ \A c \in Kids({t}) : st[c] \in {"idle", "terminal"}   \* a parent outlives its children
  /\ (ConfirmStop /\ st[t] = "cancelling") => \A a \in Owned(t) : motion[a] = "rest"
  /\ (ChildStop /\ IsChild(t) /\ st[t] = "cancelling") => \A a \in Actuators : last[a] # t
  /\ Settle(queue, [a \in Actuators |-> IF owner[a] = t THEN None ELSE owner[a]],
            [st EXCEPT ![t] = "terminal"])
  /\ out' = [out EXCEPT ![t] = o]
  /\ revoked' = revoked \ {t}
  /\ UNCHANGED <<motion, last>>

Again(t) == /\ t \in Repeating /\ st[t] = "terminal"
            /\ st' = [st EXCEPT ![t] = "idle"]
            /\ out' = [out EXCEPT ![t] = "none"]
            /\ UNCHANGED <<owner, revoked, queue, motion, last>>

Ends(t) == \E o \in {"cancelled", "succeeded", "failed"} : End(t, o)

Next == \/ \E t \in Tasks : Request(t) \/ Spawn(t) \/ Cancel(t) \/ Timeout(t) \/ Ends(t) \/ Again(t)
        \/ \E t \in Tasks, a \in Actuators : Command(t, a) \/ TaskStop(t, a)
        \/ \E a \in Actuators : Arrive(a) \/ Brake(a)

Spec == Init /\ [][Next]_vars

(* Assumptions for liveness: every task that holds actuators ends, and a stop comes to
   rest. *)
Fairness == /\ \A t \in Tasks : WF_vars(Ends(t))
            /\ \A a \in Actuators : WF_vars(Brake(a))

FairSpec == Spec /\ Fairness

-----------------------------------------------------------------------------
(* Properties *)

TypeOK == /\ st \in [Tasks -> {"idle", "waiting", "running", "cancelling", "terminal"}]
          /\ out \in [Tasks -> {"none", "succeeded", "failed", "cancelled"}]
          /\ owner \in [Actuators -> Tasks \cup {None}]
          /\ revoked \subseteq Tasks
          /\ queue \in Seq(Tasks)
          /\ motion \in [Actuators -> {"rest", "moving", "braking"}]
          /\ last \in [Actuators -> Tasks \cup {None}]

(* 3.3: an owner is a task that is running or cancelling and declared the actuator; a
   running or cancelling task owns everything it declared; nothing is leaked. *)
OwnershipConsistent ==
  /\ \A a \in Actuators : owner[a] # None =>
       /\ st[owner[a]] \in {"running", "cancelling"}
       /\ a \in Needs[owner[a]]
  /\ \A t \in Tasks : (st[t] \in {"running", "cancelling"} /\ ~IsChild(t)) => Owned(t) = Needs[t]
  /\ \A c \in Tasks : (IsChild(c) /\ st[c] \in {"running", "cancelling"})
                         => st[Parent[c]] \in {"running", "cancelling"}

QueueConsistent ==
  /\ \A t \in Tasks : st[t] = "waiting" <=> InQueue(t)
  /\ \A i, j \in 1..Len(queue) : queue[i] = queue[j] => i = j

(* A cancelled, preempted or rejected task has no valid token, and no other task loses
   its token. *)
RevokedIsCancelling == \A t \in Tasks : t \in revoked <=> (st[t] = "cancelling" /\ ~IsChild(t))

(* 3.3, wait: a later request never overtakes an earlier one that overlaps it. A `take`
   is exempt: it is an explicit preemption. *)
NoOvertake ==
  [][\A x, y \in Tasks :
       (/\ x # y
        /\ st[x] # "running" /\ st'[x] = "running" /\ Mode[x] # "take"
        /\ ~IsChild(x)                 \* a child acquires nothing: it uses its parent's
        /\ InQueue(y) /\ st'[y] = "waiting"
        /\ Needs[x] \cap Needs[y] # {})
       => (InQueue(x) /\ Pos(x) < Pos(y))]_vars

(* 3.3, fail: fail at once; such a request never waits. *)
FailNeverWaits == \A t \in Tasks : Mode[t] = "fail" => st[t] # "waiting"

(* 3.3, take: ownership goes directly to the taker; no other task acquires the actuators
   in between. *)
TakeDirect ==
  [][\A x \in Tasks :
       (Mode[x] = "take" /\ st[x] = "waiting" /\ st'[x] = "waiting")
       => \A a \in Needs[x] : owner'[a] \in {owner[a], None}]_vars

(* 3.3, take: at most one take waits for an actuator. *)
OneTaker == \A x, y \in Tasks :
              (x # y /\ Mode[x] = "take" /\ Mode[y] = "take" /\ InQueue(x) /\ InQueue(y))
              => Needs[x] \cap Needs[y] = {}

(* A waiting take is resolved: it gets its actuators, or it is cancelled or refused. *)
TakeResolves == \A t \in Tasks : (Mode[t] = "take" /\ st[t] = "waiting") ~> (st[t] # "waiting")

(* 1.1: when a task that holds actuators becomes `cancelled`, the robot is stopped. *)
CancelledAtRest ==
  [][\A t \in Tasks :
       (st[t] = "cancelling" /\ st'[t] = "terminal" /\ out'[t] = "cancelled")
       => /\ \A a \in Owned(t) : motion[a] = "rest"
          /\ \A a \in Actuators : last[a] # t]_vars

(* Derived invariant of SPEC: a cancelled task cannot move the robot, even if its code
   keeps running. A new motion is always commanded by a running task. *)
NoCommandWhenCancelled ==
  [][\A a \in Actuators :
       (motion'[a] = "moving" /\ (motion[a] # "moving" \/ last'[a] # last[a]))
       => st[IF last'[a] # None THEN last'[a] ELSE owner[a]] = "running"]_vars

(* Every cancellation ends. *)
CancelTerminates == \A t \in Tasks : (st[t] = "cancelling") ~> (st[t] = "terminal")

(* 3.3, wait: no request starves, as long as tasks do not keep taking. *)
NoStarvation == \A t \in Tasks : (Mode[t] = "wait" /\ st[t] = "waiting") ~> (st[t] # "waiting")

=============================================================================
