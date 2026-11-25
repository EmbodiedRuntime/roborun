----------------------------- MODULE Ownership -----------------------------
(* Control authority over actuators: SPEC.md 3.3 and 1.6, with the Gate in the same
   process as the tasks (every step below is atomic). Checks, locks and monitors come
   with Gate.tla.

   Giving up. A task that ends, or leaves a `hold`, while an actuator of that token is
   not at rest gives up nothing yet: the Gate stops the actuator and keeps the token, with
   no task behind it, until everything the token covers is at rest (`Kept`, `Pass`). A
   kept token is revoked, so nobody commands with it; requests wait for it, a `fail` fails,
   and a `take` cancels nobody for it.

   Who asks. A task asks for its actuators when it starts; a running task asks for more
   with `hold`, here a scope: a request of its own that the task waits for, owns, and
   leaves. Tasks form a tree of any depth. A task or scope does not ask for what its own
   task or an ancestor already holds: it uses that token. A task that asks for nothing
   starts at once.

   Requests wait in one queue, served first come, first served; a `take` cancels the
   tasks that own what it asks for and the takes already waiting, and goes to the front.
   Cancelling a task cancels its whole subtree: every token in it is revoked, every
   request in it withdrawn.

   Switches. Each switches one mechanism off, to show that a property is not vacuous; the
   implementation has them all on.
     TakeFront      a `take` request goes to the front of the queue
     TakeCancels    a `take` also cancels the takes already waiting for one of its
                    actuators, and needs authority over them as over the owners
     ConfirmStop    revoking a token also stops its actuators, and a cancelling task
                    releases them only at rest (SPEC 1.1)
     FailOnReserved a `fail` request also fails when an earlier request waits for one of
                    its actuators, not only when one is owned
     Reserve        a waiter reserves its actuators against later requests
     ChildReject    a command from a cancelled task is rejected, though the ancestor's
                    token it uses is still valid
     ChildStop      cancelling a task stops the motion it commanded with an ancestor's
                    token, and it ends only once that motion is at rest
     HoldRevoke     cancelling a task revokes the tokens of its `hold` scopes too
     OwnLine        a `take` is refused when a take of the same task or of an ancestor
                    already waits for the actuator: it would cancel itself
     StopOnRelease  a token given up while one of its actuators moves is kept, and the
                    actuator stopped, until rest                                       *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Tasks, Scopes, Actuators, None,
          Needs,        \* Needs[x]: the actuators a task declares, or a scope holds
          Mode,         \* Mode[x]: on_busy, one of "wait", "fail", "take"
          Session,      \* Session[t]: the principal for authority
          Supervises,   \* pairs <<s1, s2>>: session s1 may take from session s2
          Parent,       \* Parent[t]: the task that started t, or None
          Of,           \* Of[h]: the task that enters scope h
          Repeating,    \* tasks that are started again after they end
          Impatient,    \* requests made with a wait_timeout
          TakeFront, TakeCancels, ConfirmStop, FailOnReserved, Reserve, ChildReject, ChildStop,
          HoldRevoke, OwnLine, StopOnRelease

Reqs == Tasks \cup Scopes       \* whatever can ask for actuators, and so hold a token

RECURSIVE Anc(_)
Anc(t) == IF Parent[t] = None THEN {} ELSE {Parent[t]} \cup Anc(Parent[t])

ASSUME /\ Tasks \cap Scopes = {}
       /\ \A x \in Reqs : Needs[x] \subseteq Actuators /\ Needs[x] # {}
       /\ \A x \in Reqs : Mode[x] \in {"wait", "fail", "take"}
       /\ Repeating \subseteq Tasks /\ Impatient \subseteq Reqs
       /\ {TakeFront, TakeCancels, ConfirmStop, FailOnReserved, Reserve, ChildReject,
           ChildStop, HoldRevoke, OwnLine, StopOnRelease} \subseteq BOOLEAN
       /\ \A t \in Tasks : Parent[t] \in Tasks \cup {None} /\ t \notin Anc(t)
       /\ \A h \in Scopes : Of[h] \in Tasks

VARIABLES st,       \* st[x]: "idle", "waiting" (in the queue), "running" (a scope: held),
                    \*        "cancelling", "terminal" (a scope: left)
          out,      \* out[t]: how task t ended: "none", "succeeded", "failed", "cancelled"
          asks,     \* asks[x]: what x asked for itself; empty while idle, and once
                    \*          terminal with nothing kept
          owner,    \* owner[a]: the task or scope whose token covers a, or None
          revoked,  \* the tasks and scopes whose token is revoked or kept
          queue,    \* the waiting requests, in order
          motion,   \* motion[a]: "rest", "moving", "braking"
          last      \* last[a]: the task whose command a is carrying out with an
                    \*          ancestor's token; None at rest or when the owner commanded it

vars == <<st, out, asks, owner, revoked, queue, motion, last>>

Tk(x) == IF x \in Scopes THEN Of[x] ELSE x      \* the task behind a request
Line(t) == {t} \cup Anc(t)
Sub(P) == {d \in Tasks : Line(d) \cap P # {}}   \* the tasks in P and everything below them
ScopesOf(P) == {h \in Scopes : Of[h] \in P}
Kids(t) == {c \in Tasks : Parent[c] = t}

InQueue(x) == \E i \in 1..Len(queue) : queue[i] = x
Pos(x) == CHOOSE i \in 1..Len(queue) : queue[i] = x
Without(q, S) == SelectSeq(q, LAMBDA x : x \notin S)
Owned(x) == {a \in Actuators : owner[a] = x}
Uses(t) == Needs[t] \cup UNION {Needs[h] : h \in ScopesOf({t})}

(* The tokens the Gate keeps until rest: their task has ended, or has left the `hold`. *)
Kept == {x \in Reqs : st[x] = "terminal" /\ Owned(x) # {}}

(* A task that waits for its `hold` does nothing else. *)
Blocked(t) == \E h \in ScopesOf({t}) : st[h] = "waiting"

(* Authority to take (3.3): the same session, or a supervisor of the owner's session. *)
MayTake(x, o) == Session[Tk(x)] = Session[Tk(o)]
                 \/ <<Session[Tk(x)], Session[Tk(o)]>> \in Supervises

(* Serve the queue in order. A request is granted when all its actuators are free and no
   earlier request that is still waiting asks for one of them. Returns <<owner, granted>>. *)
RECURSIVE Serve(_, _, _, _, _)
Serve(q, own, reserved, granted, ak) ==
  IF q = <<>> THEN <<own, granted>>
  ELSE LET x == Head(q)
           free == \A a \in ak[x] : own[a] = None
       IN IF free /\ ak[x] \cap reserved = {}
          THEN Serve(Tail(q), [a \in Actuators |-> IF a \in ak[x] THEN x ELSE own[a]],
                     reserved, granted \cup {x}, ak)
          ELSE Serve(Tail(q), own, IF Reserve THEN reserved \cup ak[x] ELSE reserved,
                     granted, ak)

(* The state after serving queue q0 with ownership own0, states st0 and requests ak. *)
Settle(q0, own0, st0, ak) ==
  LET r == Serve(q0, own0, {}, {}, ak)
  IN /\ owner' = r[1]
     /\ queue' = Without(q0, r[2])
     /\ st' = [x \in Reqs |-> IF x \in r[2] THEN "running" ELSE st0[x]]
     /\ asks' = [x \in Reqs |-> IF st'[x] \in {"idle", "terminal"}
                                     /\ \A a \in Actuators : owner'[a] # x
                                  THEN {} ELSE ak[x]]

(* Cancelling the tasks in C, a set closed under Sub: with them their scopes. What runs
   starts cancelling and its token is revoked first (3.1); what waits has had no effect and
   ends at once; the Gate stops what their tokens cover and what they commanded. *)
Hit(C) == C \cup (IF HoldRevoke THEN ScopesOf(C) ELSE {h \in ScopesOf(C) : st[h] = "waiting"})
StAfter(H) == [x \in Reqs |-> IF x \in H /\ st[x] = "running" THEN "cancelling"
                              ELSE IF x \in H /\ st[x] = "waiting" THEN "terminal"
                              ELSE st[x]]
OutAfter(H) == [t \in Tasks |-> IF t \in H /\ st[t] = "waiting" THEN "cancelled" ELSE out[t]]
RevAfter(H) == revoked \cup {x \in H : st[x] \in {"running", "cancelling"} /\ asks[x] # {}}
MotionAfter(H) == [a \in Actuators |->
                     IF /\ motion[a] = "moving"
                        /\ \/ ConfirmStop /\ owner[a] \in H
                           \/ ChildStop /\ last[a] \in H /\ st[last[a]] = "running"
                     THEN "braking" ELSE motion[a]]

-----------------------------------------------------------------------------

Init == /\ st = [x \in Reqs |-> "idle"]
        /\ out = [t \in Tasks |-> "none"]
        /\ asks = [x \in Reqs |-> {}]
        /\ owner = [a \in Actuators |-> None]
        /\ revoked = {}
        /\ queue = <<>>
        /\ motion = [a \in Actuators |-> "rest"]
        /\ last = [a \in Actuators |-> None]

(* ResourceBusy: a task that asked at its start fails; a `hold` raises in its task. *)
Refuse(x) == /\ st' = [st EXCEPT ![x] = "terminal"]
             /\ out' = IF x \in Tasks THEN [out EXCEPT ![x] = "failed"] ELSE out
             /\ UNCHANGED <<asks, owner, revoked, queue, motion, last>>

(* start(), or entering `hold`: x asks for the actuators that neither its task nor an
   ancestor holds with a valid token (3.3, 1.6). *)
Request(x) ==
  /\ st[x] = "idle"
  /\ LET by == IF x \in Scopes THEN Of[x] ELSE Parent[x]     \* the task that asks
     IN  IF by = None THEN TRUE ELSE st[by] = "running" /\ ~Blocked(by)
  /\ LET line == Line(Tk(x))
         ask == {a \in Needs[x] : ~(owner[a] # None /\ owner[a] \notin revoked
                                      /\ Tk(owner[a]) \in line)}
         owners == {owner[a] : a \in ask} \ {None}
         live == owners \ Kept        \* behind a kept token there is nobody to cancel
         ahead == {y \in Reqs : InQueue(y) /\ asks[y] \cap ask # {}}
         takers == IF TakeCancels THEN {y \in ahead : Mode[y] = "take"} ELSE {}
     IN IF ask = {}
        THEN /\ st' = [st EXCEPT ![x] = "running"]       \* it uses what is held above it
             /\ UNCHANGED <<out, asks, owner, revoked, queue, motion, last>>
        ELSE IF Mode[x] = "fail" /\ (owners # {} \/ (FailOnReserved /\ ahead # {}))
        THEN Refuse(x)                                  \* ResourceBusy
        ELSE IF Mode[x] = "take" /\ OwnLine /\ \E y \in takers : Tk(y) \in line
        THEN Refuse(x)                                  \* it would cancel itself
        ELSE IF Mode[x] = "take" /\ \E o \in live \cup takers : ~MayTake(x, o)
        THEN Refuse(x)                                  \* no authority; nothing is cancelled
        ELSE LET H == IF Mode[x] = "take"
                      THEN Hit(Sub({Tk(y) : y \in live \cup takers})) ELSE {}
                 st0 == [StAfter(H) EXCEPT ![x] = "waiting"]
                 rest == Without(queue, H)
                 q0 == IF Mode[x] = "take" /\ TakeFront
                       THEN <<x>> \o rest ELSE Append(rest, x)
             IN /\ Settle(q0, owner, st0, [asks EXCEPT ![x] = ask])
                /\ revoked' = RevAfter(H)
                /\ motion' = MotionAfter(H)
                /\ out' = OutAfter(H)
                /\ UNCHANGED last

(* cancel(): it reaches the whole subtree. *)
Cancel(t) ==
  /\ st[t] \in {"waiting", "running"}
  /\ LET H == Hit(Sub({t}))
     IN /\ Settle(Without(queue, H), owner, StAfter(H), asks)
        /\ revoked' = RevAfter(H)
        /\ motion' = MotionAfter(H)
        /\ out' = OutAfter(H)
        /\ UNCHANGED last

(* wait_timeout expires: ResourceBusy. *)
Timeout(x) ==
  /\ x \in Impatient /\ st[x] = "waiting"
  /\ Settle(Without(queue, {x}), owner, [st EXCEPT ![x] = "terminal"], asks)
  /\ out' = IF x \in Tasks THEN [out EXCEPT ![x] = "failed"] ELSE out
  /\ UNCHANGED <<revoked, motion, last>>

(* A goal or setpoint from t for a: forwarded only with a valid token of t or of an
   ancestor. Once t is cancelled, the Gate no longer forwards what it sends, though an
   ancestor's token is still valid. *)
Command(t, a) ==
  /\ st[t] \in {"running", "cancelling"} /\ ~Blocked(t)
  /\ a \in Uses(t)
  /\ owner[a] # None /\ owner[a] \notin revoked /\ Tk(owner[a]) \in Line(t)
  /\ (ChildReject /\ Tk(owner[a]) # t) => st[t] = "running"
  /\ motion' = [motion EXCEPT ![a] = "moving"]
  /\ last' = [last EXCEPT ![a] = IF Tk(owner[a]) = t THEN None ELSE t]
  /\ UNCHANGED <<st, out, asks, owner, revoked, queue>>

(* stop: always accepted, also from a task whose token is revoked. *)
TaskStop(t, a) ==
  /\ st[t] \in {"running", "cancelling"} /\ ~Blocked(t)
  /\ a \in Uses(t) /\ motion[a] = "moving"
  /\ motion' = [motion EXCEPT ![a] = "braking"]
  /\ UNCHANGED <<st, out, asks, owner, revoked, queue, last>>

(* The backend: a motion reaches its goal, or a stop comes to rest. *)
Arrive(a) == /\ motion[a] = "moving"
             /\ motion' = [motion EXCEPT ![a] = "rest"]
             /\ last' = [last EXCEPT ![a] = None]
             /\ UNCHANGED <<st, out, asks, owner, revoked, queue>>

Brake(a) == /\ motion[a] = "braking"
            /\ motion' = [motion EXCEPT ![a] = "rest"]
            /\ last' = [last EXCEPT ![a] = None]
            /\ UNCHANGED <<st, out, asks, owner, revoked, queue>>

(* Give up what the tasks and scopes in S own. *)
Released(S) == [a \in Actuators |-> IF owner[a] \in S THEN None ELSE owner[a]]

(* Of the tokens in S, those the Gate keeps: one of their actuators is not at rest. *)
Keep(S) == IF StopOnRelease
           THEN {x \in S : \E a \in Owned(x) : motion[a] # "rest"} ELSE {}
Stopped(K) == [a \in Actuators |->
                 IF owner[a] \in K /\ motion[a] = "moving" THEN "braking" ELSE motion[a]]

(* The task ends: it has left its scopes, or was cancelled in them, and what it and they
   own is released, or kept until rest. A cancelled task waits for what was revoked from
   it, not for what it gave up before: that is the Gate's already. *)
End(t, o) ==
  /\ st[t] \in {"running", "cancelling"} /\ ~Blocked(t)
  /\ o \in (IF st[t] = "running" THEN {"succeeded", "failed"}
                                 ELSE {"cancelled", "succeeded", "failed"})
  /\ \A c \in Kids(t) : st[c] \in {"idle", "terminal"}   \* a parent outlives its children
  /\ \A h \in ScopesOf({t}) : st[h] \in {"idle", "terminal", "cancelling"}
  /\ LET mine == {t} \cup ScopesOf({t})
         keep == Keep(mine)
     IN /\ (ConfirmStop /\ st[t] = "cancelling") =>
             \A a \in Actuators : owner[a] \in mine \ Kept => motion[a] = "rest"
        /\ (ChildStop /\ st[t] = "cancelling") => \A a \in Actuators : last[a] # t
        /\ Settle(queue, Released(mine \ keep),
                  [x \in Reqs |-> IF x \in mine /\ st[x] # "idle" THEN "terminal" ELSE st[x]],
                  asks)
        /\ revoked' = (revoked \ mine) \cup keep
        /\ motion' = Stopped(keep)
  /\ out' = [out EXCEPT ![t] = o]
  /\ UNCHANGED last

(* The task leaves its `hold`. *)
Leave(h) ==
  /\ st[h] = "running" /\ st[Of[h]] = "running"
  /\ LET keep == Keep({h})
     IN /\ Settle(queue, Released({h} \ keep), [st EXCEPT ![h] = "terminal"], asks)
        /\ revoked' = revoked \cup keep
        /\ motion' = Stopped(keep)
  /\ UNCHANGED <<out, last>>

(* A cancelled task unwinds out of its `hold`: the revoked token is given up at rest,
   possibly before the task itself has ended. *)
Drop(h) ==
  /\ st[h] = "cancelling"
  /\ ConfirmStop => \A a \in Owned(h) : motion[a] = "rest"
  /\ Settle(queue, Released({h}), [st EXCEPT ![h] = "terminal"], asks)
  /\ revoked' = revoked \ {h}
  /\ UNCHANGED <<out, motion, last>>

(* What the Gate kept is at rest: it is given up, and the next request is served. *)
Pass(x) ==
  /\ x \in Kept
  /\ \A a \in Owned(x) : motion[a] = "rest"
  /\ Settle(queue, Released({x}), st, asks)
  /\ revoked' = revoked \ {x}
  /\ UNCHANGED <<out, motion, last>>

(* A task is started again. Its name stands for the token as well, so here it starts
   again only once nothing is kept under it. *)
Again(t) == /\ t \in Repeating /\ st[t] = "terminal"
            /\ ({t} \cup ScopesOf({t})) \cap Kept = {}
            /\ st' = [x \in Reqs |-> IF x \in {t} \cup ScopesOf({t}) THEN "idle" ELSE st[x]]
            /\ out' = [out EXCEPT ![t] = "none"]
            /\ UNCHANGED <<asks, owner, revoked, queue, motion, last>>

Ends(t) == \E o \in {"cancelled", "succeeded", "failed"} : End(t, o)

Next == \/ \E x \in Reqs : Request(x) \/ Timeout(x) \/ Pass(x)
        \/ \E t \in Tasks : Cancel(t) \/ Ends(t) \/ Again(t)
        \/ \E h \in Scopes : Leave(h) \/ Drop(h)
        \/ \E t \in Tasks, a \in Actuators : Command(t, a) \/ TaskStop(t, a)
        \/ \E a \in Actuators : Arrive(a) \/ Brake(a)

Spec == Init /\ [][Next]_vars

(* Assumptions for liveness: every task that holds actuators ends, every `hold` is left,
   and a stop comes to rest. *)
Fairness == /\ \A t \in Tasks : WF_vars(Ends(t))
            /\ \A h \in Scopes : WF_vars(Leave(h)) /\ WF_vars(Drop(h))
            /\ \A x \in Reqs : WF_vars(Pass(x))
            /\ \A a \in Actuators : WF_vars(Brake(a))

FairSpec == Spec /\ Fairness

-----------------------------------------------------------------------------
(* Properties *)

TypeOK == /\ st \in [Reqs -> {"idle", "waiting", "running", "cancelling", "terminal"}]
          /\ out \in [Tasks -> {"none", "succeeded", "failed", "cancelled"}]
          /\ asks \in [Reqs -> SUBSET Actuators]
          /\ owner \in [Actuators -> Reqs \cup {None}]
          /\ revoked \subseteq Reqs
          /\ queue \in Seq(Reqs)
          /\ motion \in [Actuators -> {"rest", "moving", "braking"}]
          /\ last \in [Actuators -> Tasks \cup {None}]

(* 3.3: an owner is running or cancelling, or its token is kept, and it asked for the
   actuator; whatever does not wait owns exactly what it asked for; nothing is leaked. A
   task runs only under a parent that has not ended, and a scope is held only by such a
   task. *)
OwnershipConsistent ==
  /\ \A a \in Actuators : owner[a] # None =>
       /\ st[owner[a]] \in {"running", "cancelling"} \/ owner[a] \in Kept
       /\ a \in asks[owner[a]]
  /\ \A x \in Reqs : st[x] # "waiting" => Owned(x) = asks[x]
  /\ \A x \in Reqs : asks[x] \subseteq Needs[x]
  /\ \A c \in Tasks : (Parent[c] # None /\ st[c] \in {"waiting", "running", "cancelling"})
                         => st[Parent[c]] \in {"running", "cancelling"}
  /\ \A h \in Scopes : st[h] \in {"waiting", "running", "cancelling"}
                         => st[Of[h]] \in {"running", "cancelling"}

QueueConsistent ==
  /\ \A x \in Reqs : st[x] = "waiting" <=> InQueue(x)
  /\ \A i, j \in 1..Len(queue) : queue[i] = queue[j] => i = j

(* A cancelled, preempted or rejected task has no valid token, and no other task loses
   its token. *)
RevokedIsCancelling ==
  \A x \in Reqs : x \in revoked <=> ((st[x] = "cancelling" /\ asks[x] # {}) \/ x \in Kept)

(* 3.3: whatever is not at rest has an owner, and a request is granted only actuators at
   rest. Without StopOnRelease a task that ends with a goal on its way leaves a motion
   nobody owns, and the next owner starts with a moving actuator. *)
MotionHasOwner == \A a \in Actuators : motion[a] # "rest" => owner[a] # None
GrantedAtRest ==
  [][\A a \in Actuators : (owner'[a] # owner[a] /\ owner'[a] # None) => motion[a] = "rest"]_vars

(* Nothing new starts on a kept token: its actuators are at rest or coming to rest. *)
KeptIsStopping == \A x \in Kept : \A a \in Owned(x) : motion[a] # "moving"

(* Every kept token is given up. *)
ReleaseSettles == \A x \in Reqs : (x \in Kept) ~> (x \notin Kept)

(* 1.5: a cancellation reaches everything below the task: nothing there still runs, waits
   or holds. *)
CancelReachesSubtree ==
  \A t \in Tasks : st[t] = "cancelling" =>
    \A x \in (Sub({t}) \ {t}) \cup ScopesOf(Sub({t})) : st[x] \notin {"running", "waiting"}

(* 3.3, wait: a later request never overtakes an earlier one that overlaps it. A `take`
   is exempt: it is an explicit preemption. *)
NoOvertake ==
  [][\A x, y \in Reqs :
       (/\ x # y
        /\ st[x] # "running" /\ st'[x] = "running" /\ Mode[x] # "take"
        /\ InQueue(y) /\ st'[y] = "waiting"
        /\ asks'[x] \cap asks[y] # {})  \* empty for one that only uses what is held above it
       => (InQueue(x) /\ Pos(x) < Pos(y))]_vars

(* 3.3, fail: fail at once; such a request never waits. *)
FailNeverWaits == \A x \in Reqs : Mode[x] = "fail" => st[x] # "waiting"

(* 3.3, take: ownership goes directly to the taker; no other task acquires the actuators
   in between. *)
TakeDirect ==
  [][\A x \in Reqs :
       (Mode[x] = "take" /\ st[x] = "waiting" /\ st'[x] = "waiting")
       => \A a \in asks[x] : owner'[a] \in {owner[a], None}]_vars

(* 3.3, take: at most one take waits for an actuator. *)
OneTaker == \A x, y \in Reqs :
              (x # y /\ Mode[x] = "take" /\ Mode[y] = "take" /\ InQueue(x) /\ InQueue(y))
              => asks[x] \cap asks[y] = {}

(* A waiting take is resolved: it gets its actuators, or it is cancelled or refused. *)
TakeResolves == \A x \in Reqs : (Mode[x] = "take" /\ st[x] = "waiting") ~> (st[x] # "waiting")

(* 1.1: when a task that holds actuators becomes `cancelled`, the robot is stopped. *)
CancelledAtRest ==
  [][\A t \in Tasks :
       (st[t] = "cancelling" /\ st'[t] = "terminal" /\ out'[t] = "cancelled")
       => /\ \A a \in Actuators : owner[a] \in ({t} \cup ScopesOf({t})) \ Kept
                                      => motion[a] = "rest"
          /\ \A a \in Actuators : last[a] # t]_vars

(* Derived invariant of SPEC: a cancelled task cannot move the robot, even if its code
   keeps running. A new motion is always commanded by a running task. *)
NoCommandWhenCancelled ==
  [][\A a \in Actuators :
       (motion'[a] = "moving" /\ (motion[a] # "moving" \/ last'[a] # last[a]))
       => st[IF last'[a] # None THEN last'[a] ELSE Tk(owner[a])] = "running"]_vars

(* Every cancellation ends. *)
CancelTerminates == \A t \in Tasks : (st[t] = "cancelling") ~> (st[t] = "terminal")

(* 3.3, wait: no request starves, as long as tasks do not keep taking, and no two tasks
   each `hold` what the other owns. *)
NoStarvation == \A x \in Reqs : (Mode[x] = "wait" /\ st[x] = "waiting") ~> (st[x] # "waiting")

=============================================================================
