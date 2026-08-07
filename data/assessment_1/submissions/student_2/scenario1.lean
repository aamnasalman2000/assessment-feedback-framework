
/-
PART 1 – Propositional Logic Proofs
-/

-- Scenario 1: Advanced Course Eligibility Decision
-- Given:
-- provable (x1 ∨ x2), provable (y1 ∨ y2)
-- and: provable x1 → provable z
--      provable x2 → provable z
--      provable y1 → provable z
--      provable y2 → provable z
-- Prove: provable z
section Scenario1

-- Declare variables for propositions and provability predicate
section Scenario1

variable (x1 x2 y1 y2 z : Prop)
variable (provable : Prop → Prop)

axiom Hx : ∀ (x1 x2 : Prop) (provable : Prop → Prop), provable x1 ∨ provable x2
axiom Hy : ∀ (y1 y2 : Prop) (provable : Prop → Prop), provable y1 ∨ provable y2
axiom R1 : ∀ (x1 z : Prop) (provable : Prop → Prop), provable x1 → provable z
axiom R2 : ∀ (x2 z : Prop) (provable : Prop → Prop), provable x2 → provable z
axiom R3 : ∀ (y1 z : Prop) (provable : Prop → Prop), provable y1 → provable z
axiom R4 : ∀ (y2 z : Prop) (provable : Prop → Prop), provable y2 → provable z

example : provable z := by
  have h : provable x1 ∨ provable x2 := Hx x1 x2 provable
  cases h with
  | inl px1 =>
      have r1 : provable x1 → provable z := R1 x1 z provable
      exact r1 px1
  | inr px2 =>
      have r2 : provable x2 → provable z := R2 x2 z provable
      exact r2 px2

/- Explanation:
   - Hx is a function that, given x1, x2, and provable, yields provable x1 ∨ provable x2.
   - We instantiate Hx with the current x1, x2, provable to obtain the disjunction.
   - Using disjunction elimination (cases tactic), we handle two cases:
     - Case provable x1: Instantiate R1 to get provable x1 → provable z, then apply it.
     - Case provable x2: Instantiate R2 to get provable x2 → provable z, then apply it.
   - R1 and R2 are similarly instantiated to match the specific x1, x2, z, provable.
   - Alternatively, we could use Hy with R3 and R4, but Hx is sufficient and chosen for simplicity.
   - No additional assumptions are made beyond the given axioms.
-/

end Scenario1
