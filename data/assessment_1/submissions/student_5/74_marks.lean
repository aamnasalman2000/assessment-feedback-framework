------ PART 1 SCENARIO 1
-- Prints a message to confirm the theorem was proved.
def main : IO Unit :=
  IO.println "You have successfully proved the eligibility theorem!"
-- -- This theorem states that if you have two prerequisites (x1 or x2) and
-- -- two approvals (y1 or y2), you can derive a conclusion (z).

-- Define 'provable' as a predicate on propositions
def provable (p : Prop) : Prop := p

-- Define the variables for the propositions involved in the eligibility theorem
variable (x1 x2 y1 y2 z : Prop)

-- Assumptions: x1 or x2 and y1 or y2 are provable
variable (hx : provable (x1 ∨ x2))
variable (hy : provable (y1 ∨ y2))

-- Rules: each of the variables x1, x2, y1 and y2 can imply z
variable (h1 : provable x1 → provable z)
variable (h2 : provable x2 → provable z)
variable (h3 : provable y1 → provable z)
variable (h4 : provable y2 → provable z)

-- Goal: Show that eligibility z is provable
example : provable z :=
  -- Start by eliminating the disjunction x1 ∨ x2
  Or.elim hx
    (fun hx1 =>
      -- Case: x1 is provable
      -- Use rule h1 to derive provable z
      h1 hx1)
    (fun _ =>
      -- Case: x2 is provable

      -- Now eliminate the disjunction y1 ∨ y2
      Or.elim hy
        (fun hy1 =>
          -- Subcase: y1 is provable
          -- Use rule h3 to derive provable z
          h3 hy1)
        (fun hy2 =>
          -- Subcase: y2 is provable
          -- Use rule h4 to derive provable z
          h4 hy2))




------PART 1 SCENARIO 2
-- Scenario 2 code to prove a ∧ d given the assumptions
set_option linter.unusedVariables false

-- Declare the propositions as variables
variable (a b c d : Prop)

-- Assumptions from system logic
variable (h1 : b → d)                     -- High amount → biometric
variable (h2 : ¬a → c ∧ d)                -- Unregistered device means that location consistency and  biometric must hold
variable (h3 : a ∨ d)                     -- At least registered OR biometric passed
variable (hb : b)                         -- High amount occurred
variable (hnc : ¬c)                       -- Location mismatch

-- Goal: Prove a ∧ d given the above assumptions
  example : a ∧ d :=
  have hd : d := h1 hb         -- If transaction amount exceeds spending, biometric must be done
  have ha : a :=               -- Prove a using proof by contradiction
    by
      by_cases ha : a               -- Case analysis on a
      · exact ha                      -- If a is true, we're done
      · let hcd := h2 ha               -- From ¬a → c ∧ d
        have hc := hcd.left            -- Extract c
        exact absurd hc hnc           -- Contradiction: have c and ¬c
  And.intro ha hd                        -- Combine the results to show a ∧ d



------ PART 1 SCENARIO 3
set_option linter.unusedVariables false

-- Define the propositions and their meanings
variable (c : Prop) -- Heavy congestion is detected on the usual route
variable (w : Prop) -- Weather conditions on the alternate route are unsafe.
variable (f : Prop) -- The car has enough fuel for the alternate route.
variable (d : Prop) -- TThe car successfully arrives at the destination.

-- Define 'provable' as a predicate on propositions.
def provable (p : Prop) : Prop := p

-- Assumptions based on the system's logical constraints
variable (h1 : (c → ((¬w ∧ f) → d)) ∧ (¬c → d))
  -- If congestion exists, then if weather is safe and fuel is available, the car reaches the destination.
  -- But If congestion does not exist, the car reaches the destination.

variable (h2 : w → ¬c)             -- If weather is unsafe, then congestion does not exist

-- Additional assumptions
variable (hf : f)          -- Fuel is available
variable (hcase : c ∨ w)   -- Either congestion exists or weather is unsafe

-- Goal is to show that the car reaches the destination successfully
example : provable d :=
  -- Eliminate the disjunction c ∨ w. We will prove d in both cases.
  Or.elim hcase
    (fun hc : c =>
      -- Case 1: Congestion exists.
      -- We need to use the first part of our revised h1.
      have h1_congestion_case : (¬w ∧ f) → d := (And.left h1) hc

      -- To use that, we need a proof of (¬w ∧ f).
      have h_no_weather_and_fuel : ¬w ∧ f :=

        -- Proof of the conjunction
        And.intro
          -- Assume w and show it leads to a contradiction.
          (fun hw : w =>
            -- If w is true, then h2 hw gives us a proof of ¬c.
            -- This contradicts hc (our assumption that c is true).
            show False from (h2 hw) hc)

          -- Proof of f: This was given to us as hf.
          hf

      -- Apply the rule to get the goal d
      show d from h1_congestion_case h_no_weather_and_fuel)

    (fun hw : w =>
      -- Case 2: Weather is unsafe.
      -- Use the second part of the revised h1.
      have h1_no_congestion_case : ¬c → d := And.right h1

      -- From h2, unsafe weather implies no congestion.
      have hnc : ¬c := h2 hw

      -- Now we can apply our rule to get the goal d.
      show d from h1_no_congestion_case hnc)





------PART 1 SCENARIO 4

variable (t r f m : Prop)

-- Assumptions based on the system's logical constraints
variable (h1 : t → r)             -- If there is a threat, then auto-response
variable (h2 : f → ¬r)            -- If false positive, no auto-response
variable (h3 : r → m)             -- Any auto-response requires manual follow-up
variable (h4 : ¬m ∨ ¬f)           -- Either no manual intervention or not a false positive
variable (ht : t)                 -- Threat is detected

-- Goal: Prove r ∧ m ∧ ¬f
example : r ∧ m ∧ ¬f :=
  have hr : r := h1 ht           -- From a threat, auto-response occurs
  have hm : m := h3 hr           -- From auto-response, manual intervention occurs
  have hf : ¬f :=                -- Need to show ¬f
    Or.elim h4
      (fun hnm : ¬m => absurd hm hnm) -- If ¬m but we have m, contradiction
      (fun hnf : ¬f => hnf)          -- If ¬f, we are done
  And.intro hr (And.intro hm hf)     -- Combine to show r ∧ m ∧ ¬f



----------------------------------------------------------------------------------------
------ PART 2

-- opaque functions
opaque conj : Prop → Prop → Prop
opaque provable : Prop → Prop

-- Definition for the axioms
-- If x and y are provable, then y is provable
axiom AxConjElimRight : ∀ x y, provable (conj x y) → provable y

-- If x and y are provable, then x is provable
axiom AxConjElimLeft : ∀ x y, provable (conj x y) → provable x

-- if x is provable them y is provable then the conjunction of x and y is provable
axiom AxConjIntro : ∀ x y, provable x → provable y → provable (conj x y)

-- True is provable
axiom AxPrTrue : provable True

-- If somehow False is provable, that causes a contradiction
axiom AxNotPrFalse : provable False → False

-- Theorem ex1 - proving that conjunction with False is not provable; and that proving x does not lead to provable False
theorem ex1 : ∀ x, ¬ provable (conj x False) ∧ (provable x → ¬ provable False) :=
λ x =>
  -- We need to prove that `¬ provable (conj x False)`.
  -- This means that if we have a proof of `conj x False`, it leads to a contradiction.
  -- We can use the axiom `AxConjElimRight` to show that if we have a proof of `conj x False`,
  -- then we can derive `provable False`, which contradicts the axiom `AxNotPrFalse`.
  let hx : ¬ provable (conj x False) :=
    λ h =>
      let pr_false := AxConjElimRight x False h
      AxNotPrFalse pr_false
  -- We need to prove that if `provable x` then `¬ provable False`
  -- This means that if we have a proof of `x`, we cannot have a proof of `False`.
  -- We can use the axiom `AxNotPrFalse` to show this.
  let hxf : provable x → ¬ provable False :=
    λ _ h_pf => AxNotPrFalse h_pf
  -- Finally, we combine both parts into a conjunction
  And.intro hx hxf


-- Theorem ex2 -  From a contradiction (x ∧ False), anything follows (principle of explosion)
theorem ex2 : ∀ x y, provable (conj x False) → provable y :=
λ x _ h =>
  -- Step 1: Extract provable False from provable (x ∧ False)
  let pr_false : provable False := AxConjElimRight x False h
  -- Step 2: Turn provable False into contradiction using the axiom
  let contradiction : False := AxNotPrFalse pr_false
  -- Step 3: Eliminate the contradiction to produce provable y
  False.elim contradiction



-- Theorem ex3 - Thhe function that takes the propositions x, y, z and a proof of the nested conjunction as its arguments.
theorem ex3 : ∀ x y z, provable (conj x (conj y z)) → provable (conj (conj y x) z) :=
λ x y z h =>
  -- Step 1: Extract x from the outer conjunction
  let px : provable x := AxConjElimLeft x (conj y z) h

  -- Step 2: Extract (y ∧ z) from the outer conjunction
  let pyz : provable (conj y z) := AxConjElimRight x (conj y z) h

  -- Step 3: Extract y from (y ∧ z)
  let py : provable y := AxConjElimLeft y z pyz

  -- Step 4: Extract z from (y ∧ z)
  let pz : provable z := AxConjElimRight y z pyz

  -- Step 5: Combine y and x to get provable (y ∧ x)
  let pyx : provable (conj y x) := AxConjIntro y x py px

  -- Step 6: Combine (y ∧ x) and z to get the final result
  AxConjIntro (conj y x) z pyx pz


----------------------------------------------------------------------------------------
-- Theorem ex4 - If conj(x, y) is not provable but x is provable, then y must not be provable
theorem ex4 : ∀ x y : Prop,
  ¬ provable (conj x y) →
  provable x →
  ¬ provable y :=
λ x y hnotConj hprovX =>
  -- Assume for contradiction that provable y
  -- If conj x y is not provable, then we cannot have both x and y provable
  -- If y is provable, then conj x y would be provable, showing a contradiction
  fun hprovY : provable y =>
    -- Then conj x y would be provable via AxConjIntro because we have both x and y provable
    let hConj : provable (conj x y) := AxConjIntro x y hprovX hprovY

    -- But that contradicts the assumption that conj x y is not provable
    hnotConj hConj
