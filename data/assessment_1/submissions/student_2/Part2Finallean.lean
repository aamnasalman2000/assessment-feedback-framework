
--Part 2:
-- Definitions
opaque conj : Prop → Prop → Prop
opaque provable : Prop → Prop

-- Axioms
axiom AxConjElimRight : ∀ x y, provable (conj x y) → provable y
axiom AxConjElimLeft : ∀ x y, provable (conj x y) → provable x
axiom AxConjIntro : ∀ x y, provable x → provable y → provable (conj x y)
axiom AxPrTrue : provable True
axiom AxNotPrFalse : provable False → False

-- Theorem ex1: Prove that conjunction with False is not provable,
-- and provability of x does not imply provability of False
theorem ex1 : ∀ x, ¬provable (conj x False) ∧ (provable x → ¬provable False) := by
  intro x
  constructor
  -- Part 1: Prove ¬provable (conj x False)
  · intro h
    -- Assume provable (conj x False)
    -- Use AxConjElimRight to get provable False
    have h_false : provable False := AxConjElimRight x False h
    -- Apply AxNotPrFalse: provable False → False
    exact AxNotPrFalse h_false
  -- Part 2: Prove provable x → ¬provable False
  · intro h_px
    -- Assume provable x, need ¬provable False
    intro h_false
    -- provable False leads to contradiction via AxNotPrFalse
    exact AxNotPrFalse h_false

-- Explanation for ex1:
-- Assumptions: Use axioms AxConjElimRight and AxNotPrFalse.
-- Strategy:
-- - For ¬provable (conj x False):
--   - Assume provable (conj x False).
--   - Extract provable False using AxConjElimRight.
--   - Use AxNotPrFalse to derive a contradiction, proving the negation.
-- - For provable x → ¬provable False:
--   - Assume provable x and provable False.
--   - AxNotPrFalse directly gives a contradiction, so ¬provable False holds.
-- The proof uses contradiction to handle negations.

-- Theorem ex2: Principle of explosion
-- From provable (conj x False), anything follows
theorem ex2 : ∀ x y, provable (conj x False) → provable y := by
  intro x y h
  -- Assume provable (conj x False)
  -- Use AxConjElimRight to get provable False
  have h_false : provable False := AxConjElimRight x False h
  -- Use AxNotPrFalse to derive False
  have contra : False := AxNotPrFalse h_false
  -- From False, anything follows (ex falso quodlibet)
  exact False.elim contra

-- Explanation for ex2:
-- Assumptions: Use AxConjElimRight and AxNotPrFalse.
-- Strategy:
-- - Assume provable (conj x False).
-- - Extract provable False using AxConjElimRight.
-- - Apply AxNotPrFalse to get False.
-- - Use False.elim to derive any proposition y (principle of explosion).
-- The proof exploits the contradiction in provable False to prove any y.

-- Theorem ex3: Rearranging nested conjunctions
-- Prove that provable (conj x (conj y z)) implies provable (conj (conj y x) z)
theorem ex3 : ∀ x y z, provable (conj x (conj y z)) → provable (conj (conj y x) z) := by
  intro x y z h
  -- Assume provable (conj x (conj y z))
  -- Extract components using elimination axioms
  have h_x : provable x := AxConjElimLeft x (conj y z) h
  have h_yz : provable (conj y z) := AxConjElimRight x (conj y z) h
  have h_y : provable y := AxConjElimLeft y z h_yz
  have h_z : provable z := AxConjElimRight y z h_yz
  -- Construct provable (conj y x) using AxConjIntro
  have h_yx : provable (conj y x) := AxConjIntro y x h_y h_x
  -- Construct the goal: provable (conj (conj y x) z)
  exact AxConjIntro (conj y x) z h_yx h_z

-- Explanation for ex3:
-- Assumptions: Use AxConjElimLeft, AxConjElimRight, and AxConjIntro.
-- Strategy:
-- - From provable (conj x (conj y z)), extract provable x and provable (conj y z).
-- - From provable (conj y z), extract provable y and provable z.
-- - Build provable (conj y x) using provable y and provable x.
-- - Build the goal provable (conj (conj y x) z) using provable (conj y x) and provable z.
-- The proof rearranges conjunctions by extracting and recombining components.

-- Theorem ex4: Using negation to infer missing provability
-- If conj x y is not provable, and x is provable, then y is not provable
theorem ex4 : ∀ x y, ¬provable (conj x y) → provable x → ¬provable y := by
  intro x y h_nconj h_px
  -- Assume ¬provable (conj x y) and provable x
  -- Need to prove ¬provable y
  intro h_py
  -- Assume provable y, derive contradiction
  -- Since provable x and provable y, use AxConjIntro to get provable (conj x y)
  have h_conj : provable (conj x y) := AxConjIntro x y h_px h_py
  -- This contradicts h_nconj: ¬provable (conj x y)
  exact h_nconj h_conj

-- Explanation for ex4:
-- Assumptions: Use AxConjIntro.
-- Strategy:
-- - Assume ¬provable (conj x y), provable x, and (for contradiction) provable y.
-- - Use provable x and provable y to construct provable (conj x y) via AxConjIntro.
-- - This contradicts the assumption ¬provable (conj x y).
-- - Thus, provable y cannot hold, so ¬provable y.
-- The proof uses proof by contradiction to show y is not provable.
