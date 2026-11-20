------------------------------ MODULE Protocol ------------------------------
(* The Gate in its own process: SPEC.md 3.1 and 3.8, for one runtime and one actuator.

   The runtime sends requests; every request gets a reply; what the Gate decides on its
   own (a cancellation order, a grant to a request that waited) travels as an event with
   its next reply or tick. Both directions are FIFO, and anything can be in flight. The
   runtime keeps a mirror of the grants its tasks hold; the Gate is authoritative.

   Switches (see Tasks.tla):
     Concurrent        the Gate takes the next request while the checks of a command run
     WithdrawReleases  withdrawing a request that was granted meanwhile releases the grant
     Tell              when the Gate revokes a token on its own, it tells the runtime *)
EXTENDS FiniteSets, Naturals, Sequences

CONSTANTS Tasks, None, MaxInFlight, Concurrent, WithdrawReleases, Tell

VARIABLES mirror,     \* runtime: none, asked, valid, revoked, gone
          toGate,     \* requests in flight
          toRt,       \* replies and ticks in flight, each with events
          grant,      \* Gate: none, waiting, valid, revoked, released
          releasing,  \* Gate: revoked grants to release once the actuator is at rest
          events,     \* Gate: events for the runtime, not yet sent
          inflight,   \* Gate: tasks with a command whose checks are running
          motion,     \* rest, moving, stopping
          connected,
          step

vars == <<mirror, toGate, toRt, grant, releasing, events, inflight, motion, connected, step>>

Ops    == {"acquire", "withdraw", "release", "revoke", "send", "stop"}
Event  == [ev : {"cancel", "grant"}, t : Tasks]
Holders == {t \in Tasks : grant[t] \in {"valid", "revoked"}}
Braked == IF motion = "moving" THEN "stopping" ELSE motion
Did(a, t) == step' = [a |-> a, t |-> t]

Request(op, t) == toGate' = Append(toGate, [op |-> op, t |-> t])
(* A reply or a tick: it carries every event the Gate has for the runtime. *)
Reply(op, t, more) == /\ toRt' = Append(toRt, [op |-> op, t |-> t, evs |-> events \cup more])
                      /\ events' = {}

-----------------------------------------------------------------------------
(* The runtime *)

Init == /\ mirror = [t \in Tasks |-> "none"] /\ grant = [t \in Tasks |-> "none"]
        /\ toGate = <<>> /\ toRt = <<>> /\ events = {} /\ releasing = {} /\ inflight = {}
        /\ motion = "rest" /\ connected = TRUE
        /\ step = [a |-> "init", t |-> None]

RAcquire(t) == /\ mirror[t] = "none"
               /\ mirror' = [mirror EXCEPT ![t] = "asked"]
               /\ Request("acquire", t) /\ Did("ask", t)
               /\ UNCHANGED <<toRt, grant, releasing, events, inflight, motion, connected>>

(* The task is cancelled: while it waits, the request is withdrawn; while it owns, its
   token is revoked here first and the Gate is told. *)
RCancel(t) == /\ mirror[t] \in {"asked", "valid"}
              /\ mirror' = [mirror EXCEPT ![t] = IF mirror[t] = "asked" THEN "gone" ELSE "revoked"]
              /\ Request(IF mirror[t] = "asked" THEN "withdraw" ELSE "revoke", t)
              /\ Did("cancel", t)
              /\ UNCHANGED <<toRt, grant, releasing, events, inflight, motion, connected>>

RSend(t) == /\ mirror[t] = "valid"
            /\ Request("send", t) /\ Did("send", t)
            /\ UNCHANGED <<mirror, toRt, grant, releasing, events, inflight, motion, connected>>

RStop == /\ Request("stop", None) /\ Did("askstop", None)
         /\ UNCHANGED <<mirror, toRt, grant, releasing, events, inflight, motion, connected>>

RRelease(t) == /\ mirror[t] \in {"valid", "revoked"}
               /\ mirror' = [mirror EXCEPT ![t] = "gone"]
               /\ Request("release", t) /\ Did("giveup", t)
               /\ UNCHANGED <<toRt, grant, releasing, events, inflight, motion, connected>>

(* A reply or a tick arrives. A grant for a request that is no longer waiting is dropped;
   a cancellation order marks the token revoked (events keep their order: a grant and its
   revocation may arrive together). *)
RReceive == /\ toRt # <<>>
            /\ LET m == Head(toRt)
                   granted == {e.t : e \in {e \in m.evs : e.ev = "grant"}}
                   ordered == {e.t : e \in {e \in m.evs : e.ev = "cancel"}}
               IN  mirror' = [t \in Tasks |->
                                IF t \in granted /\ mirror[t] = "asked"
                                THEN (IF t \in ordered THEN "revoked" ELSE "valid")
                                ELSE IF t \in ordered /\ mirror[t] = "valid" THEN "revoked"
                                ELSE IF m.op = "busy" /\ m.t = t /\ mirror[t] = "asked" THEN "gone"
                                ELSE mirror[t]]
            /\ toRt' = Tail(toRt) /\ Did("hear", None)
            /\ UNCHANGED <<toGate, grant, releasing, events, inflight, motion, connected>>

-----------------------------------------------------------------------------
(* The Gate *)

Free == Holders = {}

(* The Gate takes the next request. Without Concurrent it does so only when no check is
   running, so a stop waits behind the checks of an earlier command. *)
GReceive ==
  /\ connected /\ toGate # <<>>
  /\ Concurrent \/ inflight = {}
  /\ LET m == Head(toGate)  t == m.t  IN
     /\ toGate' = Tail(toGate)
     /\ Did(m.op, t)
     /\ CASE m.op = "acquire" ->
               IF Free /\ grant[t] = "none"
               THEN /\ grant' = [grant EXCEPT ![t] = "valid"]
                    /\ Reply("granted", t, {[ev |-> "grant", t |-> t]})
                    /\ UNCHANGED <<releasing, inflight, motion>>
               ELSE /\ grant' = [grant EXCEPT ![t] = IF @ = "none" THEN "waiting" ELSE @]
                    /\ Reply("queued", t, {})
                    /\ UNCHANGED <<releasing, inflight, motion>>
          [] m.op = "withdraw" ->
               /\ grant' = [grant EXCEPT ![t] =
                              IF @ = "waiting" \/ (@ = "valid" /\ WithdrawReleases)
                              THEN "released" ELSE @]
               /\ Reply("done", t, {}) /\ UNCHANGED <<releasing, inflight, motion>>
          [] m.op = "release" ->
               /\ grant' = [grant EXCEPT ![t] = IF @ = "valid" THEN "released" ELSE @]
               /\ releasing' = IF grant[t] = "revoked" THEN releasing \cup {t} ELSE releasing
               /\ Reply("done", t, {}) /\ UNCHANGED <<inflight, motion>>
          [] m.op = "revoke" ->
               /\ grant' = [grant EXCEPT ![t] = IF @ = "valid" THEN "revoked" ELSE @]
               /\ motion' = IF grant[t] = "valid" THEN Braked ELSE motion
               /\ Reply("done", t, {}) /\ UNCHANGED <<releasing, inflight>>
          [] m.op = "stop" ->
               /\ motion' = Braked
               /\ Reply("done", t, {}) /\ UNCHANGED <<grant, releasing, inflight>>
          [] m.op = "send" ->
               IF grant[t] = "valid" /\ t \notin inflight
               THEN /\ inflight' = inflight \cup {t}
                    /\ UNCHANGED <<grant, releasing, motion, toRt, events>>
               ELSE /\ Reply("done", t, {}) /\ UNCHANGED <<grant, releasing, inflight, motion>>
  /\ UNCHANGED <<mirror, connected>>

(* The checks of a command answer (Gate.tla): it is executed for a token still valid. *)
GChecked(t) == /\ t \in inflight
               /\ inflight' = inflight \ {t}
               /\ \/ /\ grant[t] = "valid" /\ motion' = "moving" /\ Did("execute", t)
                  \/ /\ motion' = Braked /\ Did("reject", t)
               /\ IF connected THEN Reply("done", t, {}) ELSE UNCHANGED <<toRt, events>>
               /\ UNCHANGED <<mirror, toGate, grant, releasing, connected>>

GDispatch(t) == /\ grant[t] = "waiting" /\ Free /\ connected
                /\ grant' = [grant EXCEPT ![t] = "valid"]
                /\ events' = events \cup {[ev |-> "grant", t |-> t]}
                /\ Did("grant", t)
                /\ UNCHANGED <<mirror, toGate, toRt, releasing, inflight, motion, connected>>

(* The Gate cancels an owner itself (a monitor, an operator stop, a take from another
   runtime): the token is revoked first, then the runtime is told (3.1). *)
GRevokes(t) == /\ grant[t] = "valid" /\ connected
               /\ grant' = [grant EXCEPT ![t] = "revoked"]
               /\ motion' = Braked
               /\ events' = IF Tell THEN events \cup {[ev |-> "cancel", t |-> t]} ELSE events
               /\ Did("revoked", t)
               /\ UNCHANGED <<mirror, toGate, toRt, releasing, inflight, connected>>

GTick == /\ connected /\ events # {}
         /\ Reply("tick", None, {}) /\ Did("tick", None)
         /\ UNCHANGED <<mirror, toGate, grant, releasing, inflight, motion, connected>>

GRest == /\ motion = "stopping" /\ motion' = "rest" /\ Did("rest", None)
         /\ UNCHANGED <<mirror, toGate, toRt, grant, releasing, events, inflight, connected>>

(* A revoked grant is released only at rest (1.1). *)
GReleases(t) == /\ t \in releasing /\ motion = "rest"
                /\ grant' = [grant EXCEPT ![t] = "released"]
                /\ releasing' = releasing \ {t}
                /\ Did("released", t)
                /\ UNCHANGED <<mirror, toGate, toRt, events, inflight, motion, connected>>

(* The connection closes: every token it held is revoked and the robot stops (3.1). *)
Disconnect == /\ connected /\ connected' = FALSE
              /\ grant' = [t \in Tasks |-> IF grant[t] = "valid" THEN "revoked"
                                           ELSE IF grant[t] = "waiting" THEN "released"
                                           ELSE grant[t]]
              /\ releasing' = releasing \cup Holders
              /\ motion' = Braked
              /\ toGate' = <<>> /\ toRt' = <<>> /\ events' = {}
              /\ mirror' = [t \in Tasks |-> "gone"]
              /\ Did("disconnect", None)
              /\ UNCHANGED inflight

Next == \/ connected /\ \E t \in Tasks : RAcquire(t) \/ RCancel(t) \/ RSend(t) \/ RRelease(t)
        \/ connected /\ (RStop \/ RReceive)
        \/ GReceive \/ GTick \/ GRest \/ Disconnect
        \/ \E t \in Tasks : GChecked(t) \/ GDispatch(t) \/ GRevokes(t) \/ GReleases(t)

Spec == Init /\ [][Next]_vars

(* For TLC: at most MaxInFlight messages in flight each way. *)
Bounded == Len(toGate) <= MaxInFlight /\ Len(toRt) <= MaxInFlight

-----------------------------------------------------------------------------
(* Properties *)

TypeOK == /\ mirror \in [Tasks -> {"none", "asked", "valid", "revoked", "gone"}]
          /\ grant \in [Tasks -> {"none", "waiting", "valid", "revoked", "released"}]
          /\ releasing \subseteq Tasks /\ inflight \subseteq Tasks /\ events \subseteq Event
          /\ motion \in {"rest", "moving", "stopping"} /\ connected \in BOOLEAN

OneHolder == Cardinality(Holders) <= 1

(* A command is executed only for the task that holds the actuator at the Gate. *)
ExecutedForTheOwner == step.a = "execute" => grant[step.t] = "valid"

InFlightToRt(e) == e \in events \/ \E i \in 1..Len(toRt) : e \in toRt[i].evs
InFlightToGate(op, t) == \E i \in 1..Len(toGate) : toGate[i] = [op |-> op, t |-> t]

(* 3.1: the runtime never believes in a token the Gate has not granted; and when the Gate
   has revoked one the runtime still believes in, the order to cancel is on its way. *)
MirrorIsSound == \A t \in Tasks :
                   mirror[t] = "valid" =>
                     \/ grant[t] = "valid"
                     \/ grant[t] = "revoked" /\ InFlightToRt([ev |-> "cancel", t |-> t])

(* No grant is left with a task that no longer wants it: the request that gives it up is
   on its way. *)
NoOrphanGrant == \A t \in Tasks :
                   (connected /\ grant[t] = "valid" /\ mirror[t] = "gone") =>
                     InFlightToGate("withdraw", t) \/ InFlightToGate("release", t)

(* 3.4 rule 5, across the boundary: the Gate takes the next request, a stop for one,
   whatever the checks are doing. *)
NextRequestNeverWaits == (connected /\ toGate # <<>>) => (Concurrent \/ inflight = {})

(* 3.1: after a disconnect no token is valid, and nothing is left moving. *)
DisconnectStops == ~connected => \A t \in Tasks : grant[t] # "valid"

(* 1.1: a revoked grant is released only at rest. *)
ReleasedAtRest == [][\A t \in Tasks : (grant[t] = "revoked" /\ grant'[t] = "released")
                                        => motion = "rest"]_vars

=============================================================================
