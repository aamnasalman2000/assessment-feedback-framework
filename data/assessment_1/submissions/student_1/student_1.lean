
-- Assessment 1 - Part 1: Representation in First Order Logic
/-
----------------------------------------------------
Part 1: Logical Scenarios
----------------------------------------------------
-/

-- Scenario 1: Course Eligibility Decision
example (x1 x2 y1 y2 z : Prop)
  (h1 : x1 ∨ x2)
  (h2 : y1 ∨ y2)
  (hx : (x1 → z) ∧ (x2 → z))
  (hy : (y1 → z) ∧ (y2 → z)) : z :=
  by
    cases h1 with
    | inl hx1 => exact hx.left hx1
    | inr hx2 => exact hx.right hx2

/-
Explanation:
We split the disjunction x1 ∨ x2. In both cases, we apply the corresponding implication to derive z.
The assumption about y1 ∨ y2 is not needed in this case to establish z.
-/

-- Scenario 2: Complex Transaction Verification System
example (a b c d : Prop)
  (h : (b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) : a ∧ d :=
  by
    rcases h with ⟨hbd, hand, haord, hb, hnc⟩
    have hd := hbd hb
    cases haord with
    | inl ha => exact ⟨ha, hd⟩
    | inr hd' => exact ⟨by_contradiction (λ hna => hnc ((hand hna).left)), hd'⟩

/-
Explanation:
From b and b → d we get d. Then from a ∨ d and ¬c, we infer a must be true via contradiction.
-/

-- Scenario 3: Advanced Autonomous Vehicle Decision System
example (cg ws f s : Prop)
  (h4 : cg → (ws ∧ f → s))
  (h5 : ws → ¬cg) :
  (cg → ws → f → s) ∧ (ws → ¬cg) :=
  by
    constructor
    · intros hcg hws hf
      apply h4 hcg
      exact ⟨hws, hf⟩
    · exact h5

/-
Explanation:
Direct application of implications using intro and conjunction construction.
-/

-- Scenario 4: Cybersecurity System for Threats
example (t r f m : Prop)
  (h6 : (t → r) ∧ (f → ¬r) ∧ (r → m) ∧ ((¬m ∨ ¬f) → t)) :
  r ∧ m ∧ ¬f :=
  by
    rcases h6 with ⟨htr, hfr, hrm, hnmft⟩
    by_contra h
    push_neg at h
    rcases h with ⟨hr, hm, hf⟩
    have contra := hfr hf
    exact contra hr

/-
Explanation:
Assume the negation and push it inside using `push_neg`.
From hf and hfr we get ¬r, contradicting hr. Hence, original triple holds.
-/

/-
----------------------------------------------------
Part 2: Formal Proofs in Lean
----------------------------------------------------
-/

opaque conj : Prop → Prop → Prop
opaque provable : Prop → Prop

axiom AxConjElimRight : ∀ x y, provable (conj x y) → provable y
axiom AxConjElimLeft : ∀ x y, provable (conj x y) → provable x
axiom AxConjIntro : ∀ x y, provable x → provable y → provable (conj x y)
axiom AxPrTrue : provable True
axiom AxNotPrFalse : provable False → False

-- Theorem 1
theorem ex1 : ∀ x, ¬ provable (conj x False) ∧ (provable x → ¬ provable False) :=
  by
    intro x
    constructor
    · intro h
      have pf : provable False := AxConjElimRight x False h
      exact AxNotPrFalse pf
    · intros px pf
      exact AxNotPrFalse pf

-- Theorem 2
theorem ex2 : ∀ x y, provable (conj x False) → provable y :=
  by
    intros x y h
    have pf : provable False := AxConjElimRight x False h
    exact False.elim (AxNotPrFalse pf)

-- Theorem 3
theorem ex3 : ∀ x y z, provable (conj x (conj y z)) → provable (conj (conj y x) z) :=
  by
    intros x y z h
    have px : provable x := AxConjElimLeft x (conj y z) h
    have pyz : provable (conj y z) := AxConjElimRight x (conj y z) h
    have py : provable y := AxConjElimLeft y z pyz
    have pz : provable z := AxConjElimRight y z pyz
    have pxy : provable (conj y x) := AxConjIntro y x py px
    exact AxConjIntro (conj y x) z pxy pz

-- Theorem 4
theorem ex4 : ∀ x y : Prop,
  ¬ provable (conj x y) →
  provable x →
  ¬ provable y :=
  by
    intros x y h px py
    apply h
    exact AxConjIntro x y px py

/-
Explanation for ex1:
This theorem shows two ideas:
1. The conjunction (x ∧ False) is not provable, because it would lead to provable False.
2. Even if x is provable, that does not mean False is provable.

The proof uses:
- Elimination on conj x False to get provable False.
- Then applies AxNotPrFalse to derive a contradiction.
- So the whole conjunction is unprovable, and provable x does not allow proving False.
-*/

/-
Explanation for ex2:
From a contradiction, anything follows (principle of explosion).
If conj x False is provable, we can extract provable False.
From that, we can derive any provable y using False.elim and AxNotPrFalse.
-*/

/-
Explanation for ex3:
This theorem restructures nested conjunctions:
From provable (x ∧ (y ∧ z)), we extract:
- provable x
- provable y and z from the inner conjunction

Then we reconstruct:
- provable (y ∧ x)
- followed by provable ((y ∧ x) ∧ z)
This shows how associativity of conjunction is preserved under provability.
-*/

/-
Explanation for ex4:
We assume that provable x and provable y.
Then by AxConjIntro, we would get provable (x ∧ y), which contradicts our assumption ¬ provable (x ∧ y).
Therefore, y cannot be provable.
This is a proof by contradiction.
-
