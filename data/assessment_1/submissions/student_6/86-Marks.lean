/-

Module code: OCOM5201M
Module title: Knowledge Representation and Reasoning
Assessment title: Assessment 1

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
PART 1

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Scenario 1: Advanced Course Eligibility Decision

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

A university enrolment AI-based system is required to determine if a student is eligible (z) for an advanced module based on two conditions:

Passing at least one of two prerequisites (x1 ∨ x2).
Receiving approval either explicitly from the primary department (y1) or conditionally from a secondary committee (y2).

Formally represent and prove eligibility (z) given:

Given:
  (provable (disj x1 x2))
  (provable (disj y1 y2))

  Rules provided are:

  (provable x1 -> provable z) and (provable x2 -> provable z)
  (provable y1 -> provable z) and (provable y2 -> provable z)


I considered that both conditions need to be satisfied, i.e. Passing at least one of the two prerequisites
AND receiving approval either explicitly from the primary department or conditionally from a secondary committee.

As follows: ((x1 ∨ x2) ∧ (y1 ∨ y2)) → z

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Logical reasoning:
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

Eligibility (z) requires both:
  1-Student needs to satisfy at least one prerequisite (x1 ∨ x2).
  2-Student needs to secure approval (y1 ∨ y2).

- Given rules state that proving any of x1, x2, y1, or y2 ensures z.
- Accordingly, once both conditions are satisfied, we conclude z.
- Performed disjunctions for each condition separately, and conjunction to both,
  ensuring z is reached.

-/

-- Defining the variables, as stated in the scenario
variable (x1 x2 y1 y2 z : Prop)

-- Constructing
theorem TheEligibility
  (Passed1Preq : x1 ∨ x2) -- Defining the logic for passing at least one of the two prerequisites.
  (Got1Approval : y1 ∨ y2) -- Defining securing approval from primary dept. or secondary committee.
  (Course1Taken : x1 → z) -- Given that prerequisite 1 provable z
  (Course2Taken : x2 → z) -- Given that prerequisite 2 provable z
  (GotApproval1 : y1 → z) -- Given that approval from primary dept. provable z
  (GotApproval2 : y2 → z) -- Given that approval from secondary committee provable z
  (Eligibility : ((x1 ∨ x2) ∧ (y1 ∨ y2)) → z) -- Logic conjunction of both conditions.
  : z := by
  -- Handling at least one of the prerequisites (x1 ∨ x2)
  have Finish_Preq : z :=
    match Passed1Preq with
    | Or.inl Course1 => Course1Taken Course1
    | Or.inr Course2 => Course2Taken Course2

  -- Handling one of the approvals (y1 ∨ y2)
  have Get_approval : z :=
    match Got1Approval with
    | Or.inl Approval1 => GotApproval1 Approval1
    | Or.inr Approval2 => GotApproval2 Approval2

  -- Conjuction of both conditions, to be eligible
  let Is_eligible := And.intro Passed1Preq Got1Approval
  -- Accomplishing goal 'z', as a result
  exact Eligibility Is_eligible

-- Goal Accomplished

/-
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Scenario 2: Complex Transaction Verification System

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------


A banking AI system employs logical conditions to authorize online transactions securely.
The system evaluates multiple propositions simultaneously:

a: The transaction originates from a device previously registered by the user.
b: The transaction amount exceeds the user's typical spending pattern.
c: The user's recent location history is consistent with the transaction's current location.
d: The transaction has passed secondary biometric verification.


The system must follow these explicitly defined logical constraints to authorize a transaction successfully:

1. If the transaction amount exceeds the user's typical spending pattern (b),
then secondary biometric verification (d) must have been successfully completed.
2. If the transaction originates from an unregistered device (¬a),
then both location consistency (c) and biometric verification (d) must hold.
3. At least one primary security condition (either the registered device (a) or biometric verification (d)) must be satisfied.

Formally represent and prove the following logical statement:

((b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) → (a ∧ d)

--------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
Logic reasoning:
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Analysing the truth tree for the logic statement, results in identifying a contradiction
between 'c' and '¬c', and this will accordingly contradicts Condition2.

  - '¬c' is claims that 'c' is false, in order for '¬c' to be true.
  - Accordingly, (c∧d) should be true, if it is assumed '¬a' (where 'a' is false)
  - This contradicts with the first point, since ¬c is assumed.
  - Therefore, contradiction with Condition2 (¬a→(c∧d)).

Looking at the logic, both 'a' and 'd' are required to be true, in order for the
transaction to be authorized.

Accordingly, 'a' and 'd' are true, if we conclude the following:
  - Condition 4: 'b' is true.
  - Condition 1: (b → d) if 'b' is true, so 'd' is true.
  - Condition 5: 'c' is false, in order to statify that '¬c' becomes true
  - Condition 2: Following Condition 5, and since 'c' is false, then ('c' ∧ 'd') is false too.
    For ¬a → (c∧d) to be true, '¬a' need to be false, and therefore 'a' is true. Again, 'c'
    will need to be true, which contradicts with '¬c', which will result to be false.
  - Condition 3: to be true, either 'a' and / or 'd' are true.

Therefore, if 'a' is true,
    Condition 3 is true.
    Condition 4 is true.
    Condition 1 is true

    Condition 2 is in contradiction
-/

-- Defining the variables, as stated in the scenario
variable (a b c d : Prop)

-- Defining the conditions, as stated in the scenario

variable (Condition1 : (b→d))
variable (Condition2 : ¬a → (c ∧ d))
variable (Condition3 : a ∨ d)
variable (Condition4 : b)
variable (Condition5 : ¬c)

-- Establishing the theorem Authorized, with the proposition stated
theorem Authorized : ((b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c) → (a ∧ d) := by
  rintro ⟨Condition1, Condition2, Condition3, Condition4, Condition5⟩

  -- For Condition1: getting 'd' from Condition1 (b→d), and providing 'b' from Condition4
  have hd : d := Condition1 Condition4

  -- For Condition3: getting left 'a' or right 'd'
  have ha : a := by
    cases Condition3 with
  -- In case of 'a', returning 'a'
    | inl ha =>
      exact ha
  -- In case of 'd', this results in contradiction, since 'a' is false,
  -- as explained above.
  -- '¬a' is required for Condition2
    | inr hd =>
      apply Classical.byContradiction
      intro hna
      have hc_and_hd := Condition2 hna --Contradiction
      have hc := hc_and_hd.1
      exact Condition5 hc
      -- Contradiction, requiring '¬c' to be true, as well as (c∧d) to be true

  -- providing (a∧d)
  exact ⟨ha,hd⟩

-- Goal Accomplished!

/-
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Scenario 3: Advanced Autonomous Vehicle Decision System

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

-/
/-
A self-driving car's AI system is required to evaluate multiple logical conditions
before making critical route decisions. The system uses the following propositions:

h: Heavy congestion is detected on the usual route.
w: Weather conditions on the alternate route are unsafe.
f: The car has enough fuel for the alternate route.
d: The car successfully arrives at the destination.

The system follows these complex logical conditions:

- If heavy congestion is detected, the car can only reach the destination
  if the alternate route has safe weather (¬) and enough fuel.
- If weather conditions are unsafe on the alternate route, the car must take the usual
  route, which requires no congestion (¬).


----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------

-- Defining the rules as given

Rule1 : h→(d→(¬w∧f)) )
-- Rule1: if heavy congestion, then ONLY reaching destination IF having safe weather
  and enough fuel

Rule2 : (w → ¬h)
-- Rule2: if unsafe weather, then no congestion on usual route

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
Logic reasoning:
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

For Rule1, the word ONLY, along with IF, changes the order of d to be d→(¬w∧f) instead of
(¬w∧f)→d (if the word ONLY was not stated).

Rule2 is straight forward, w→¬h.

-/


-- Defining the theorem
theorem TrafficNavigation

-- Defining the rules
  (Rule1 : h → (d → (¬w ∧ f)))
  (Rule2 : w → ¬h) :

-- Stating the complete logic
  (h → (d → (¬w ∧ f))) ∧ (w → ¬h) :=

-- Conjunction of both Rules
⟨
  -- Rule1 : h→d→(¬w∧f)
  -- fun for h and d, using Rule1 derives ¬w∧f
  fun HeavyCongestion DestinationArrived =>
    Rule1 HeavyCongestion DestinationArrived,


  -- Rule2: w → ¬h
  -- function for w to derive ¬h, using Rule2
    fun WeatherUnsafe =>
      fun NoHeavyCongestion => Rule2 WeatherUnsafe NoHeavyCongestion
⟩

-- Goal Accomplished


----------------------------------------------------------------
-------------- ANOTHER SOLUTION --------------------------------
----------------------------------------------------------------

-- Defining the variables
variable (h w f d : Prop)

-- Defining the theorem
theorem Traffic_BasicVersion

-- Defining the rules
  (Rule1 : h → d → (¬w ∧ f))
  (Rule2 : w → ¬h) :

-- Stating the complete logic
  (h → (d → (¬w ∧ f))) ∧ (w → ¬h) :=

-- Concluding the logic, with conjunction of both Rule1 and Rule2
  ⟨Rule1, Rule2⟩

/-
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Scenario 4: An Advanced Cybersecurity System for threats

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-/
/-
An advanced AI cybersecurity system evaluates and responds to potential cyber threats
based on propositional logic

The system employs the following logical propositions:

t: A cybersecurity threat has been detected.
r: An automatic response action (like isolating a system) is activated.
f: The detected threat is confirmed as a false positive after initial detection.
m: Manual intervention by cybersecurity personnel is initiated.

The system must adhere to the following logical constraints:

1. If a threat (t) is detected, the system must initiate an automatic response (r).
2. If a threat is determined to be a false positive (f), automatic response actions
must be immediately halted (¬r).
3. Any automatic response (r) must be accompanied by a manual intervention follow-up (m).
4. Either manual intervention (m) does not occur, or the threat is not a
false positive (¬f), ensuring consistency in response.


Formally represent and prove the following logical statement:
( (t → r) ∧ (f → ¬r) ∧ (r → m) ∧ (¬m ∨ ¬f) ∧ t) → (r ∧ m ∧ ¬f)
Provide a Lean proof demonstrating this logical implication.

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
Logic reasoning:
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

Analysing the truth tree for the logic statement, results in identifying a contradiction
between 'm' and '¬m'. Once 'm' is assumed, '¬m' cannot be assumed, and accordingly '¬f' will be assumed
in Cyber4.

- t is assumed, as given
- r is derived from t, using Cyber1.
- m is derived from r, using Cyber3.
- ¬f is derived from the contradiction of 'r' and '¬r' using Cyber2

-/

-- Defining the variables
variable (t r m f : Prop)

-- Defining the rules as given
variable (Cyber1 : t → r)
variable (Cyber2 : f → ¬r)
variable (Cyber3 : r → m)
variable (Cyber4 : ¬m ∨ ¬f)
variable (Cyber5 : t)

-- Defining the theorem

theorem CyberSec : ( (t → r) ∧ (f → ¬r) ∧ (r → m) ∧ (¬m ∨ ¬f) ∧ t ) → (r ∧ m ∧ ¬f) := by
  rintro ⟨Cyber1, Cyber2, Cyber3, Cyber4, Cyber5⟩

  -- t is given using Cyber5

  -- Derive 'r' using Cyber1, and Cyber5
  have hr : r := Cyber1 Cyber5

  -- Derive 'm' using Cyber3
  have hm : m := Cyber3 hr

  -- Derive '¬f'
  -- using Cyber2 to get '¬r' from 'f'
  -- using hr, this will create contradiction, and use it to produce '¬f'
  have hnf : ¬f :=
      fun hf => (Cyber2 hf) hr

  -- Analyze Cyber4 (¬m ∨ ¬f)
  cases Cyber4 with
  | inl hnm =>
    -- in case of '¬m',
    -- Since we have already established the 'm' is true.
    -- '¬m' will not be the case, contradiction '¬m' with 'm'
    contradiction

  | inr hnf =>
  -- in case of '¬f'
    exact ⟨hr, hm, hnf⟩
  -- returning 'r' ∧ 'm' ∧ '¬f'

-- Goal Accomplished

/-
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
PART 2
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-/

-- Given
opaque conj : Prop → Prop → Prop
opaque provable : Prop → Prop
axiom AxConjElimRight : ∀ x y, provable (conj x y) → provable y
axiom AxConjElimLeft : ∀ x y, provable (conj x y) → provable x
axiom AxConjIntro : ∀ x y, provable x → provable y → provable (conj x y)
axiom AxPrTrue   : provable True
axiom AxNotPrFalse : provable False → False

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-- 1 --
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

-- Proving that conjunction with False is not provable;
-- and that proving x does not lead to provable False

theorem ex1 : ∀ x, ¬ provable (conj x False) ∧ (provable x → ¬ provable False) :=
  -- Defining function
  fun x =>
  -- Conjunction of both conditions

      ⟨
        fun FirstPart =>
        -- Constructing the FirstPart
        -- AxNotPrFalse for negation
        -- AxConjElimRight for conjuction elimination
          AxNotPrFalse (AxConjElimRight x False FirstPart),


        -- Constructing the SecondPart
        -- Prx for Provable x
        -- PrFalse for provable False
        fun _Prx PrFalse =>
        -- AxNotPrFalse for negation of provable False
         AxNotPrFalse PrFalse
      ⟩
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-- 2 --
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

-- From a contradiction (x ∧ False), anything follows (principle of explosion)

theorem ex2 : ∀ x y, provable (conj x False) → provable y :=

  -- Defining function
  fun x _y Prx =>

  -- Using absurd to derive contradiction
  -- As per principle of explosion, since contradiction is derived, anything proposition is provable
  -- AxConjElimRight to get the right side of the conjunction
   absurd (AxConjElimRight x False Prx) AxNotPrFalse

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-- 3 --
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

-- Rearranging nested conjunctions using provability
-- change from (x ∧ (y∧z)) to ((y ∧ x) ∧ z)

theorem ex3 : ∀ x y z, provable (conj x (conj y z)) → provable (conj (conj y x) z) :=

  -- Defining function
  fun x y z PrLogic =>

    -- Defining PrZ to get the left side of the conjunction i.e x
    have PrX := AxConjElimLeft x (conj y z) PrLogic

    -- Defining Pryz to get the right side of the conjunction i.e. y ∧ z
    have PrYZ := AxConjElimRight x (conj y z) PrLogic

    -- Defining Pry to get the left of the Pryz, which is y
    have PrY := AxConjElimLeft y z PrYZ

    -- Defining Prz to get the right of the Pryz, which is z
    have PrZ := AxConjElimRight y z PrYZ

    -- By this, we have the x y z individually

    -- Now combining the elements again in the order y x z

    AxConjIntro (conj y x) z (AxConjIntro y x PrY PrX) PrZ

--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
-- 4 --
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------

-- Using negation to infer missing provability

theorem ex4 : ∀ x y : Prop, ¬ provable (conj x y) → provable x → ¬ provable y :=

  -- Defining function
  fun x y PrConjFalse PrX PrY =>

    -- PrConjFalse is for conj x y is not true

    -- Conjunction PrX and PrY is true
    -- While PrConFalse is conj is not true, which is contradiction

    -- absurd to prove that y is not true, and not provable

    absurd (AxConjIntro x y PrX PrY) PrConjFalse
