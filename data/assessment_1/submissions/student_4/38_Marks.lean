
-- Scenario 1: Advanced Course Eligibility Decision

-- Declare propositional variables
variable (x1 x2 y1 y2 z : Prop) -- x1 and x2 are pre prerequisites, y1 and y2 are approval departments, z is condition to entry
variable (provable : Prop → Prop) -- The meta level prdicate takes the proposions to check provable.

-- Given rules

variable (r1 : provable x1 → provable z) --If x1 is provable, then z is provable.
variable (r2 : provable x2 → provable z) --If x2 is provable, then z is provable.
variable (r3 : provable y1 → provable z) --If y1 is provable, then z is provable.
variable (r4 : provable y2 → provable z) --If y2 is provable, then z is provable.
variable (hx2 : provable x2)

-- Rule to conculde eligibility z is provable
example : provable z := r2 hx2


-- Test with another rule with the below assumption
-- Assumptions
variable (ha : x1) -- Passing prerequisites x1
variable (hy1 : y1) -- explicitly from the primary department y1
variable (cond : C → y2) -- conditionally from a secondary committee
variable (rule : (x1 ∨ x2) ∧ (y1 ∨ y2) → z) --If passing least one of two prerequisites and
-- conditionally from a secondary committee

example : z :=
  rule (⟨(Or.inl ha : x1 ∨ x2), (Or.inl hy1 : y1 ∨ y2)⟩)

-- Decalarion for Part 2
-- Part 2 for for Example 1
opaque conj : Prop -> Prop -> Prop
opaque Provablerovable : Prop -> Prop
axiom AxConjElimRight : ∀ x y, provable (conj x y) -> provable y
axiom AxConjElimLeft : ∀ x y, provable (conj x y) -> provable x
axiom AxConjIntro : ∀ x y, provable x -> provable y -> provable (conj x y)
axiom AxPrTrue : provable True
axiom AxNotPrFalse : provable False -> False
-- To be re-visited
example : ∀ x, ¬ provable (conj x False) ∧ (provable x -> ¬ provable False) := sorry

--======================================================================================
--======================================================================================

-- Scenario 2: Complex Transaction Verifi cation System
-- Declare propositions
variable (a b c d : Prop)

-- Assumptions
variable (ha : a)       -- a:The transaction originates from a device previously registered by the user.
variable (hb : b)       -- b:The transaction amount exceeds the user's typical spending pattern.
variable (hd : d)       -- d:The transaction has passed secondary biometric verifi cation.
variable (hnegc : ¬c)   -- c:The user's recent location history is consistent with the transaction's current location.

-- Given Rule to apply
variable (rule : ((b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) → (a ∧ d))
-- Declare the example. Note, Unused variables can be ignored.
example
  (a b c d : Prop) (ha : a) (hb : b) (hd : d) (hnegc : ¬c):
  ((b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) → (a ∧ d) := by

  intro h
  rcases h with ⟨h1, h2, h3, h4, h5⟩
  exact And.intro ha hd

-- Part 2 for for Example 2
-- To be re-visited
example : ∀ x y, provable (conj x False) -> provable y := sorry

--======================================================================================
--======================================================================================


--Scenario 3: Advanced Autonomous Vehicle Decision System
-- Declare propositions
variable (A B C D: Prop)
-- A: Heavy congestion is detected on the usual route.
-- B: Weather conditions on the alternate route are unsafe.
-- C: The car has enough fuel for the alternate route.
-- D: The car successfully arrives at the destination.
variable (AlternateRoute UsualRoute : Prop)

-- Declare the rules
variable (rule1 : A ∧ C ∧ ¬B → AlternateRoute) -- If heavy congestion is detected, the car can only reach the destination if the alternate route has safe weather (¬) and enough fuel
variable (rule2 : B ∧ ¬A → UsualRoute) -- If weather conditions are unsafe on the alternate route, the car must take the usual route, which requires no congestion (¬).

-- Apply rules
example (ha : A) (hc : C) (hnb : ¬B) : AlternateRoute := by
  apply rule1
  exact ⟨ha, hc, hnb⟩

example (hb : B) (hna : ¬A) : UsualRoute := by
  apply rule2
  exact ⟨hb, hna⟩

--Part 2 for Example 3
-- To be re-visited
example : ∀ x y z, provable (conj x (conj y z)) -> provable (conj (conj y x) z) := sorry


--======================================================================================
--======================================================================================

-- Scenario 4: An Advanced Cybersecurity System for threats
-- An advanced AI cybersecurity system evaluates and responds to potential cyber threats based on propositional logic
-- Declare propositions
variable (t r f m : Prop)
-- t: A cybersecurity threat has been detected.
-- r: An automatic response action (like isolating a system) is activated.
-- f: The detected threat is confi rmed as a false positive after initial detection.
-- m : Manual intervention by cybersecurity personnel is initiate

-- Theorem to prove
-- If a threat (t) is detected, the system must initiate an automatic response (r).
-- If a threat is determined to be a false positive (f), automatic response actions must be immediately halted (¬r).
-- Any automatic response (r) must be accompanied by a manual intervention follow-up (m).
-- Either manual intervention (m) does not occur, or the threat is not a false positive (¬f), ensuring consistency in response.
example :
  ((t → r) ∧ (f → ¬r) ∧ (r → m) ∧ (¬m ∨ ¬f) ∧ t) → (r ∧ m ∧ ¬f) := by
  -- Assumption h for the above 5 facts. Note, n denotes negation.
  intro h
  rcases h with ⟨htr, hfrnr, hrm, nm_or_nf, ht⟩

  -- derive r an automatic response action
  have hr : r := htr ht

  -- derive m manual intervention
  have hm : m := hrm hr

  -- Use (¬m ∨ ¬f) and known D to deduce ¬r
  -- Since ¬m is false, only ¬r can be true
  -- ¬m contradicts
  have hnf : ¬f := by
    cases nm_or_nf with
    | inl hnm => contradiction
    | inr hnf => exact hnf

  -- Result
  exact ⟨hr, hm, hnf⟩

  --Part 2 for Example 4
  -- To be re-visited
  example : ¬ provable (conj x y)→ provable x  → ¬ provable y := sorry
