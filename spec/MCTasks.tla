------------------------------ MODULE MCTasks ------------------------------
(* A model of Tasks for TLC: a tree three deep with one sibling.

       Root - a - b - c
                \ d                                                          *)
EXTENDS Tasks, TLC

CONSTANTS a, b, c, d

MCParent == (a :> Root) @@ (b :> a) @@ (c :> b) @@ (d :> a)

=============================================================================
