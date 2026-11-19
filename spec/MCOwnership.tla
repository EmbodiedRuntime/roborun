---------------------------- MODULE MCOwnership ----------------------------
(* Models of Ownership for TLC. A config picks a subset of these tasks.

     w1  wait  {arm, grip}   session s1      k1  take  {arm}   session s1
     w2  wait  {arm}         session s1      k2  take  {arm}   session s2
     g   wait  {grip}        session s1      k3  take  {arm}   session s1
     f   fail  {arm}         session s1      kb  take  {arm, grip}  session s1

     c   a child of w2, using its arm

   Session s2 supervises s1: k2 may take from the others, and none may take from k2. *)
EXTENDS Ownership, TLC

CONSTANTS w1, w2, g, f, k1, k2, k3, kb, c, arm, grip, s1, s2

MCNeeds == (w1 :> {arm, grip}) @@ (w2 :> {arm}) @@ (g :> {grip})
           @@ (f :> {arm}) @@ (k1 :> {arm}) @@ (k2 :> {arm}) @@ (k3 :> {arm})
           @@ (kb :> {arm, grip}) @@ (c :> {arm})

MCMode == (w1 :> "wait") @@ (w2 :> "wait") @@ (g :> "wait")
          @@ (f :> "fail") @@ (k1 :> "take") @@ (k2 :> "take") @@ (k3 :> "take") @@ (kb :> "take") @@ (c :> "wait")

MCSession == (w1 :> s1) @@ (w2 :> s1) @@ (g :> s1) @@ (f :> s1) @@ (k1 :> s1) @@ (k2 :> s2)
             @@ (k3 :> s1) @@ (kb :> s1) @@ (c :> s1)

MCParent == [t \in {w1, w2, g, f, k1, k2, k3, kb, c} |-> IF t = c THEN w2 ELSE None]

MCSupervises == {<<s2, s1>>}

=============================================================================
