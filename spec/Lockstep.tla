------------------------------ MODULE Lockstep ------------------------------
(* The lockstep handshake: SPEC.md 2.1.

   The Gate process owns simulated time. It sends tick(t); the runtime sets its clock,
   lets every task react, and sends ack once it is settled: no request is waiting for a
   reply and the tasks have nothing more to ask. Requests made meanwhile are answered at
   once; their checks may take (real) time. Only then does the simulator advance.

   Switches (see Tasks.tla):
     AckWhenSettled  the runtime acknowledges only when no request waits for a reply
     TickWhenQuiet   the Gate sends a tick only when its monitors have judged the new step *)
EXTENDS Naturals, Sequences

CONSTANTS MaxT,     \* steps to run
          MaxReq,   \* requests the tasks may make in one step
          AckWhenSettled, TickWhenQuiet

VARIABLES t,           \* simulated time, in steps (the Gate's clock)
          gate,        \* tick: about to send one; wait: waiting for the ack; done
          checking,    \* requests whose answer the Gate is working on
          monitoring,  \* the Gate's monitors are judging the step just made
          clock,       \* the runtime's clock
          rt,          \* idle: waiting for a tick; react: its tasks are reacting
          asked,       \* requests made in this step
          outstanding, \* requests waiting for a reply
          toGate, toRt

vars == <<t, gate, checking, monitoring, clock, rt, asked, outstanding, toGate, toRt>>

Init == /\ t = 0 /\ gate = "tick" /\ checking = 0 /\ monitoring = FALSE
        /\ clock = 0 /\ rt = "idle" /\ asked = 0 /\ outstanding = 0
        /\ toGate = <<>> /\ toRt = <<>>

(* 1. The Gate sends a tick, once its monitors have judged the step, so that the tick
   carries what they decided. *)
Tick == /\ gate = "tick"
        /\ TickWhenQuiet => ~monitoring
        /\ toRt' = Append(toRt, [op |-> "tick", t |-> t])
        /\ gate' = "wait"
        /\ UNCHANGED <<t, checking, monitoring, clock, rt, asked, outstanding, toGate>>

(* 2. The runtime hears the Gate: a tick sets its clock and wakes its tasks; a reply
   answers one request. *)
Hear == /\ toRt # <<>>
        /\ toRt' = Tail(toRt)
        /\ IF Head(toRt).op = "tick"
           THEN /\ clock' = Head(toRt).t /\ rt' = "react" /\ asked' = 0
                /\ UNCHANGED outstanding
           ELSE /\ outstanding' = outstanding - 1
                /\ UNCHANGED <<clock, rt, asked>>
        /\ UNCHANGED <<t, gate, checking, monitoring, toGate>>

(* A task sends a command or an ownership request. *)
Ask == /\ rt = "react" /\ asked < MaxReq
       /\ asked' = asked + 1 /\ outstanding' = outstanding + 1
       /\ toGate' = Append(toGate, "request")
       /\ UNCHANGED <<t, gate, checking, monitoring, clock, rt, toRt>>

(* 3. Settled: the runtime acknowledges. *)
Ack == /\ rt = "react"
       /\ AckWhenSettled => outstanding = 0
       /\ rt' = "idle"
       /\ toGate' = Append(toGate, "ack")
       /\ UNCHANGED <<t, gate, checking, monitoring, clock, asked, outstanding, toRt>>

(* The Gate takes a request and starts on its answer; on an ack it advances the simulator
   by one step, or ends the run. *)
Take == /\ toGate # <<>>
        /\ toGate' = Tail(toGate)
        /\ IF Head(toGate) = "ack"
           THEN /\ gate = "wait"
                /\ IF t + 1 = MaxT THEN gate' = "done" /\ t' = t + 1
                                   ELSE gate' = "tick" /\ t' = t + 1
                /\ monitoring' = TRUE  \* they run at the new time
                /\ UNCHANGED checking
           ELSE /\ checking' = checking + 1
                /\ UNCHANGED <<t, gate, monitoring>>
        /\ UNCHANGED <<clock, rt, asked, outstanding, toRt>>

Answer == /\ checking > 0
          /\ checking' = checking - 1
          /\ toRt' = Append(toRt, [op |-> "reply", t |-> t])
          /\ UNCHANGED <<t, gate, monitoring, clock, rt, asked, outstanding, toGate>>

Judged == /\ monitoring /\ monitoring' = FALSE
          /\ UNCHANGED <<t, gate, checking, clock, rt, asked, outstanding, toGate, toRt>>

Done == gate = "done" /\ toRt = <<>> /\ checking = 0 /\ UNCHANGED vars

Next == Tick \/ Hear \/ Ask \/ Ack \/ Take \/ Answer \/ Judged \/ Done

Spec     == Init /\ [][Next]_vars
FairSpec == Spec /\ WF_vars(Tick) /\ WF_vars(Hear) /\ WF_vars(Ack) /\ WF_vars(Take) /\ WF_vars(Answer)
                 /\ WF_vars(Judged)

-----------------------------------------------------------------------------
(* Properties. That the handshake cannot deadlock is checked by TLC itself. *)

TypeOK == /\ t \in 0..MaxT /\ clock \in 0..MaxT
          /\ gate \in {"tick", "wait", "done"} /\ rt \in {"idle", "react"}
          /\ checking \in 0..MaxReq /\ asked \in 0..MaxReq /\ outstanding \in 0..MaxReq

(* The simulator never advances while user code is still handling a step: when time
   moves, no request is in flight, being answered, or waiting for its reply. *)
SimulatorWaits == [][t' # t => /\ outstanding = 0 /\ checking = 0
                               /\ \A i \in 1..Len(toGate) : toGate[i] = "ack"]_vars

(* Tasks react at the time of the tick: the two clocks agree whenever user code runs. *)
OneTimeBase == rt = "react" => clock = t

(* A reply is heard in the step its request was made in: no answer crosses a tick. *)
RepliesInStep == \A i \in 1..Len(toRt) : toRt[i].op = "reply" => toRt[i].t = t

(* A tick carries everything the Gate decided about its step: the monitors have judged. *)
TickIsComplete == [][(gate = "tick" /\ gate' = "wait") => ~monitoring]_vars

(* The run goes on: every step is reached. *)
TimeAdvances == <>(gate = "done")

=============================================================================
