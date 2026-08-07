
section Part1Scenario1
/-
There are four possibilities to fulfill the condition for z:
a) x1, y1 are both TRUE
b) x1, y2 are both TRUE
c) x2, y1 are both TRUE
d) x2, y2 are both TRUE
-/

variable (x1 x2 y1 y2 z : Prop)
variable (h1 : x1 → y1 → z)
variable (h2 : x1 → y2 → z)
variable (h3 : x2 → y1 → z)
variable (h4 : x2 → y2 → z)


example : ((x1 ∨ x2) ∧ (y1 ∨ y2)) → z :=
  fun h =>
    match h with
    | And.intro (Or.inl hx1) (Or.inl hy1) => h1 hx1 hy1 -- case a)
    | And.intro (Or.inl hx1) (Or.inr hy2) => h2 hx1 hy2 -- case b)
    | And.intro (Or.inr hx2) (Or.inl hy1) => h3 hx2 hy1 -- case c)
    | And.intro (Or.inr hx2) (Or.inr hy2) => h4 hx2 hy2 -- case d)
end Part1Scenario1

section Part1Scenario2
/- The logical statement appears to express a contradiction.
  I do not see how this can be proven.
-/
variable (a b c d : Prop)
example : (((b → d)) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) → (a ∧ d) :=
fun h =>
-- decomposition of the proposition:
  let hbd     := h.left   -- b → d
  let h2      := h.right
  let hnacd   := h2.left   -- ¬ a → (c ∧ d)
  let h3      := h2.right
  let had     := h3.left    -- a ∨ d
  let h4      := h3.right
  let hb      := h4.left    -- b
  let hnc     := h4.right   -- ¬ c
end Part1Scenario2

section Part1Scenario3
/-Naming the propositions:
a: Heavy congestion is detected on the usual route
b: Weather conditions on the alternate route are unsafe
c: The car has enough fuel for the alternate route
d: The car successfully arrives at the destination
The conditions stated as hypotheses h1 to h4 below are what we know
what can be proven is therefore limited to the disjunctions h1 ∨ h2 and h3 ∨ h4
-/
variable (a b c d : Prop)

/- h1: without congestion on usual route (¬a), the car arrives at the destination -/
variable (h1 : ¬a → d)

/- h2: if weather conditions on the alternate route are not unsafe (¬b) and there
    is enough fuel, the car also arrives at the destination-/
variable (h2 : (a ∧ ¬b ∧ c) → d)

example : (¬a ∨ (a ∧ ¬b ∧ c)) → d :=
  fun h =>
      match h with
      | Or.inl hL => h1 hL
      | Or.inr hR => h2 hR

-- h3, h4 now to show the opposite case
/- h3: the destination cannot be reached (¬d) if there is both a congestion on the
   usual route (a) and the weather on the alternate route is unsafe (b)
-/
variable (h3 : (a ∧ b) → ¬d)
/- h4: the destination can also not be reached (¬d) if the weather on the alternate
   route is not unsafe (¬b) but there is not enough fuel (¬c) while there is
   congestion on the usual route
-/
variable (h4 : (a ∧ ¬b ∧ ¬c) → ¬d)

example : ( (a ∧ b) ∨ (a ∧ ¬b ∧ ¬c) ) → ¬d :=
  fun h =>
    match h with
      | Or.inl hL => h3 hL
      | Or.inr hR => h4 hR

end Part1Scenario3

section Part1Scenario4
variable (t r f m : Prop)
example : ((t → r) ∧ (f → ¬r) ∧ (r → m) ∧ (¬m ∨ ¬f) ∧ t) → (r ∧ m ∧ ¬f) :=
  fun h =>
  -- decomposition of the proposition:
    let htr       := h.left     -- (t → r)
    let h2        := h.right
    let hfr       := h2.left    -- (f → r)
    let h3        := h2.right
    let hrm       := h3.left    -- (r → m)
    let h4        := h3.right
    let hnmnf     := h4.left    -- (¬m ∨ ¬f)
    let ht        := h4.right   -- t
    let hr        := htr ht     -- r implied by t
    let hm        := hrm hr     -- m implied by r

end Part1Scenario4
