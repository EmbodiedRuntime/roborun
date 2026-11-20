---------------------------- MODULE MCOwnership ----------------------------
(* Models of Ownership for TLC. A config picks a subset of these tasks and scopes.

     w1  wait  {arm, grip}   session s1      k1  take  {arm}   session s1
     w2  wait  {arm}         session s1      k2  take  {arm}   session s2
     g   wait  {grip}        session s1      k3  take  {arm}   session s1
     f   fail  {arm}         session s1      kb  take  {arm, grip}  session s1

   Below w2:
     c   a child of w2, using its arm
     gc  a child of c, using w2's arm two levels up
     cg  a child of w2 that uses the arm and asks for the grip itself (wait)

   Scopes (`hold`):
     hg  w2 holds the grip (wait)      ha  g holds the arm (wait)
     hk  w2 holds the grip (take)      ck  a child of c that takes the grip

   Session s2 supervises s1: k2 may take from the others, and none may take from k2. *)
EXTENDS Ownership, TLC

CONSTANTS w1, w2, g, f, k1, k2, k3, kb, c, gc, cg, ck, hg, ha, hk, arm, grip, s1, s2

MCNeeds == (w1 :> {arm, grip}) @@ (w2 :> {arm}) @@ (g :> {grip})
           @@ (f :> {arm}) @@ (k1 :> {arm}) @@ (k2 :> {arm}) @@ (k3 :> {arm})
           @@ (kb :> {arm, grip}) @@ (c :> {arm}) @@ (gc :> {arm}) @@ (cg :> {arm, grip})
           @@ (ck :> {grip}) @@ (hg :> {grip}) @@ (ha :> {arm}) @@ (hk :> {grip})

MCMode == (w1 :> "wait") @@ (w2 :> "wait") @@ (g :> "wait")
          @@ (f :> "fail") @@ (k1 :> "take") @@ (k2 :> "take") @@ (k3 :> "take") @@ (kb :> "take")
          @@ (c :> "wait") @@ (gc :> "wait") @@ (cg :> "wait") @@ (ck :> "take")
          @@ (hg :> "wait") @@ (ha :> "wait") @@ (hk :> "take")

MCSession == [t \in {w1, w2, g, f, k1, k2, k3, kb, c, gc, cg, ck} |-> IF t = k2 THEN s2 ELSE s1]

MCParent == [t \in {w1, w2, g, f, k1, k2, k3, kb, c, gc, cg, ck} |->
               IF t \in {c, cg} THEN w2 ELSE IF t \in {gc, ck} THEN c ELSE None]

MCOf == (hg :> w2) @@ (ha :> g) @@ (hk :> w2)

MCSupervises == {<<s2, s1>>}

=============================================================================
