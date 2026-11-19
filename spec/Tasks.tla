------------------------------- MODULE Tasks -------------------------------
(* The task state machine and the task tree: SPEC.md 1.1, 1.2 and 1.5.

   A task is created by its parent's body, runs a body, and becomes terminal only after
   its children have. Resources and the Gate are left out; they are in Ownership.tla.

   Two switches turn off one mechanism each. They exist to show that the properties below
   are not vacuous: with a switch off, TLC must report the matching property violated.
     Propagate   cancelling a task also cancels its descendants   (1.5 rule 1)
     Reap        a body that ends cancels its children and waits  (1.5 rules 2, 3) *)
EXTENDS FiniteSets

CONSTANTS Tasks,      \* the tasks that may ever exist
          Root,       \* the root of the tree; always running
          Parent,     \* Parent[t]: the task whose body creates t, or Root
          Reap, Propagate

ASSUME /\ Root \notin Tasks
       /\ Parent \in [Tasks -> Tasks \cup {Root}]
       /\ Reap \in BOOLEAN
       /\ Propagate \in BOOLEAN

VARIABLES state,   \* state[t]: "absent" before t is created, then a state of SPEC 1.1
          body     \* body[t]: "none", "active" while the code runs, then how it ended

vars == <<state, body>>

Terminal == {"succeeded", "failed", "cancelled"}
Live     == {"running", "cancelling"}
States   == {"absent", "pending"} \cup Live \cup Terminal
Bodies   == {"none", "active", "ok", "error", "cancelled"}

(* SPEC 1.2, plus creation. *)
Allowed == {<<"absent", "pending">>,
            <<"pending", "running">>, <<"pending", "cancelled">>, <<"pending", "failed">>,
            <<"running", "succeeded">>, <<"running", "failed">>, <<"running", "cancelling">>,
            <<"cancelling", "cancelled">>, <<"cancelling", "succeeded">>,
            <<"cancelling", "failed">>}

Children(p) == {c \in Tasks : Parent[c] = p}

BodyActive(p) == IF p = Root THEN TRUE ELSE body[p] = "active"

(* The descendants of p that a cancellation of p reaches: it goes down through running
   tasks, and ends at pending ones. A child that is already cancelling was swept before. *)
RECURSIVE Hit(_)
Hit(p) == UNION {IF state[c] = "running" THEN {c} \cup Hit(c)
                 ELSE IF state[c] = "pending" THEN {c}
                 ELSE {} : c \in Children(p)}

(* Request cancellation of every task in S: a running task starts cancelling; a pending
   task has had no effect, so it is cancelled at once. *)
Swept(S) == [x \in Tasks |->
               IF x \in S
               THEN IF state[x] = "running" THEN "cancelling" ELSE "cancelled"
               ELSE state[x]]

Outcome(o) == IF o = "ok" THEN "succeeded" ELSE IF o = "error" THEN "failed" ELSE "cancelled"

-----------------------------------------------------------------------------

Init == /\ state = [t \in Tasks |-> "absent"]
        /\ body = [t \in Tasks |-> "none"]

(* A running body starts a child (a cancelling body may too, in its cleanup). *)
Create(t) == /\ state[t] = "absent"
             /\ BodyActive(Parent[t])
             /\ state' = [state EXCEPT ![t] = "pending"]
             /\ UNCHANGED body

Start(t) == /\ state[t] = "pending"
            /\ state' = [state EXCEPT ![t] = "running"]
            /\ body' = [body EXCEPT ![t] = "active"]

(* It could not start, e.g. its actuators stayed busy. *)
FailToStart(t) == /\ state[t] = "pending"
                  /\ state' = [state EXCEPT ![t] = "failed"]
                  /\ UNCHANGED body

(* cancel(): a request. *)
Cancel(t) == /\ state[t] \in {"pending", "running"}
             /\ state' = Swept({t} \cup (IF Propagate /\ state[t] = "running" THEN Hit(t) ELSE {}))
             /\ UNCHANGED body

(* The body returns, raises, or ends on its cancellation. Its children are cancelled. *)
BodyEnds(t, o) == /\ body[t] = "active"
                  /\ o \in {"ok", "error", "cancelled"}
                  /\ o = "cancelled" => state[t] = "cancelling"
                  /\ body' = [body EXCEPT ![t] = o]
                  /\ state' = IF Reap THEN Swept(Hit(t)) ELSE state

(* The task becomes terminal once its children are. *)
Finish(t) == /\ body[t] \in {"ok", "error", "cancelled"}
             /\ state[t] \in Live
             /\ Reap => \A c \in Children(t) : state[c] \in Terminal \cup {"absent"}
             /\ state' = [state EXCEPT ![t] = Outcome(body[t])]
             /\ UNCHANGED body

Next == \E t \in Tasks :
          \/ Create(t) \/ Start(t) \/ FailToStart(t) \/ Cancel(t) \/ Finish(t)
          \/ \E o \in {"ok", "error", "cancelled"} : BodyEnds(t, o)

Spec == Init /\ [][Next]_vars

(* Assumptions for liveness: a body whose task is cancelling ends (code does not ignore
   its cancellation for ever), and the runtime finishes a task that can be finished. *)
Fairness == \A t \in Tasks :
              /\ WF_vars(state[t] = "cancelling" /\ \E o \in {"ok", "error", "cancelled"} : BodyEnds(t, o))
              /\ WF_vars(Finish(t))

FairSpec == Spec /\ Fairness

-----------------------------------------------------------------------------
(* Properties *)

TypeOK == /\ state \in [Tasks -> States]
          /\ body \in [Tasks -> Bodies]

BodyMatchesState == \A t \in Tasks :
                      /\ body[t] = "active" => state[t] \in Live
                      /\ state[t] = "absent" => body[t] = "none"

(* 1.2: no other transition is allowed; a terminal state never changes. *)
LegalTransitions == [][\A t \in Tasks :
                         state'[t] # state[t] => <<state[t], state'[t]>> \in Allowed]_vars

(* 1.5 rules 2 and 3: a task is not terminal before its children are; no orphans. *)
ParentOutlivesChildren == \A t \in Tasks :
                            state[t] \in Terminal =>
                              \A c \in Children(t) : state[c] \in Terminal \cup {"absent"}

(* A task that has ended starts no child. *)
NoLateChild == [][\A t \in Tasks :
                    (state[t] = "absent" /\ state'[t] = "pending") =>
                      (IF Parent[t] = Root THEN TRUE ELSE state[Parent[t]] \in Live)]_vars

(* 1.5 rule 1: a cancel request reaches the children in the same step. *)
CancelReachesChildren == [][\A t \in Tasks :
                              (state[t] = "running" /\ state'[t] = "cancelling") =>
                                \A c \in Children(t) : state'[c] \notin {"pending", "running"}]_vars

(* Every cancellation ends. *)
CancelTerminates == \A t \in Tasks : (state[t] = "cancelling") ~> (state[t] \in Terminal)

=============================================================================
