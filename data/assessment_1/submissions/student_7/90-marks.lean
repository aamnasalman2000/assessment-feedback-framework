/- OCOM52012M Knowledge Representation And Reasoning Summative Assessment One -/

/-Part One -/

/- Scenario One : Advanced Course Eligibility Decision

The system evaluates multiple propositions :

       x1 - prerequisite
       x2 - prerequisite
       y1 - explicit approval from primary department
       y2 - conditional approval from secondary committee
       z  - is the student eligible for the advanced module

Two logical conditions are defined :

      x1 ∨ x2    i.e., the student must have at least one of two prerequisites
      y1 ∨ y2    i.e., the student must receive approval either explicitly from the primary department
                       or conditionally from a secondary committee

Therefore, to be eligible for the advanced course, the following logical implication must hold, i.e.,

      (x1 ∨ x2) ∧ (y1 ∨ y2) → z

Therefore the goal is to prove provable z based on one of the the following four cases (combinations) being True :

      - x1 must be True and y1 must be True, or
      - x1 must be True and y2 must be True, or
      - x2 must be True and y1 must be True, or
      - x2 must be True and y2 must be True

-/

set_option linter.unusedVariables false
opaque provable : Prop → Prop

variable (x1 x2 y1 y2 z : Prop)

--disjunctions (given)
variable (prerequisites : (x1 ∨ x2))
variable (approvals : (y1 ∨ y2))

--rules (given)
variable (prerequisite_1 : provable x1 -> provable z)
variable (prerequisite_2 : provable x2 -> provable z)
variable (approval_1 : provable y1 -> provable z)
variable (approval_2 : provable y2 -> provable z)

--additional
variable (h1 : provable y1)
variable (h2 : provable y2)

example : provable z :=
    --consider all four cases
    Or.elim prerequisites
        (
         --x1 True
         fun case1 =>
            Or.elim approvals
               --y1 True
               (fun case1_1 => approval_1 h1)
               --y2 True
               (fun case1_2 => approval_2 h2)
        )
        (
         --x2 True
         fun case2 =>
            Or.elim approvals
               --y1 True
               (fun case2_1 => approval_1 h1)
               --y2 True
               (fun case2_2 => approval_2 h2)
        )

/- Scenario Two : Complex Transaction Verification System For Authorising Online Transactions

The system evaluates multiple propositions simultaneously :

     a - the transaction originates from a device previously registered by the user
     b - the transaction amount exceeds the user's typical spending pattern
     c - the user's recent location history is consistent with the transactions current location
     d - the transaction has passed secondary biometric verification

Logical constraints are explicity defined :

     b -> d         i.e., if transaction amount exceeds the user's typical spending pattern then secondary biometric verification
                          must have been successfully completed
     ¬a -> (c ∧ d)  i.e., if the transaction originates from an unregistered device then both location consistency
                          and biometric verification must hold
     (a ∨ d)        i.e., at least one primary security condition must be satisfied

The logical statement (b → d) ∧ (¬a → (c ∧ d)) ∧ (a ∨ d) ∧ b ∧ ¬c → (a ∧ d) indicates that where the following propositions are
both true:

     b               i.e., the transaction amount has exceeded the user's typical spending pattern
     ¬c              i.e., the user's recent location history is not consistent with the transactions current location

the transaction will only be authorised where both the primary security conditions have been met, i.e.
both the propositions a and d are true

Logical Reasoning & Assumptions :

b is True (given) and b → d (b implies d). Therefore by modus ponens d is True.

a is True can be shown by contradiction i.e., :

assuming a is False, i.e. ¬a is True. Therefore ¬a → (c ∧ d) indicates that both (c ∧ d) are True. However, it is given that
¬c is True, i.e., c is False. This is a contradiction and the original assumption that a is False must be wrong.
Therefore a must also be True.

However, in Lean 4, proving ¬a is False is not the same as proving a is True. Therefore classical logic must be used.

Both d and a must be True, which also satisfies the logical constraint (a ∨ d)

-/

--define key conditions and decisions as propositions

open Classical

variable (a b c d : Prop)

example (BiometricIdentificationRequiredAsSpendingPatternExceeded : b -> d)
        (ResponseToTransactionFromUnregisteredDevice : ¬a -> (c ∧ d))
        (PrimarySecurityConditionSatisifed : (a ∨ d))
        (SpendingPatternExceeded : b)
        (InconsistentLocationHistory : ¬c) :
        (a ∧ d) :=
         --b, b → d gives d i.e. modes ponens
         have pfd : d := BiometricIdentificationRequiredAsSpendingPatternExceeded SpendingPatternExceeded
         --as both a and d are True this satisfies the condition (a ∨ d)
         have pfaord : a ∨ d := Or.intro_right a pfd
         --assume a is False, therefore from ¬a → (c ∧ d) both c and d must be True. However, it is given that ¬c is True.
         --Both c and ¬c cannot be True. This is a contradiction. Therefore the original assumption that a is False must be wrong.
         --a must therefore also be True.
         have pfa : a :=
              byContradiction
              (fun pfnota : ¬a =>
                   --modes ponens
                   have pfcd: c ∧ d := ResponseToTransactionFromUnregisteredDevice pfnota
                   --and elimination to give c
                   have pfc : c := And.left pfcd
                   --have contradiction, i.e., c and ¬c
                   show False from absurd pfc InconsistentLocationHistory)
         have pfconj : (a ∧ d) := And.intro pfa pfd
         show (a ∧ d) from pfconj

/- Scenario Three : Advanced Autonomous Vehicle Decision System

The system evaluates multiple propositions :

     v - heavy congestion is detected on the usual route
     w - weather conditions on the alternate route are unsafe
     x - the car has enough fuel for the alternate route
     y - the car successfully arrives at the destination

Logical conditions are explicity defined. These are evaluated by self-driving car's AI system prior to making
critical route decisions :

     (v ∧ (¬w ∧ x)) → y   i.e., if heavy congestion is detected, the car can only reach the destination if the alternate
                                route has safe weather (¬) and enough fuel
     (w ∧ ¬v) → y     i.e., if weather conditons on the alternate route are unsafe the car must take the usual route,
                            which requires no congestion

Therefore, to reach the destination successfully, the following logical implication must hold, i.e.,

     (v ∧ (¬w ∧ x)) ∨ (w ∧ ¬v) → y

-/

--define key conditions and decisions as propositions
variable (v w x y : Prop)
variable (CorrectRouteSelected : (v ∧ (¬w ∧ x)) ∨ (w ∧ ¬v))

variable (AlternateRoute : provable (v ∧ (¬w ∧ x)) → provable y)
variable (UsualRoute : provable (w ∧ ¬v) → provable y)

variable (h1 : provable (v ∧ (¬w ∧ x)))
variable (h2 : provable (w ∧ ¬v))

example : provable y :=
      --focus on each potential route
      Or.elim CorrectRouteSelected
      (
         --using the alternate route, to prove y v must be True and ¬w must be True and x must be True
         fun case1 => AlternateRoute h1
      )
      (
         --using the usual route, to prove y w must be True and ¬v must be True
         fun case2 => UsualRoute h2
      )

/- Scenario Four : An Advanced Cybersecurity System For Threats

The system evaluates multiple propositions :

      t - a cybersecurity threat has been detected
      r - an automatic response action is initiated
      f - the detected threat is confirmed as a false positive after initial detection
      m - manual intervention by cybersecurity personnel is initiated

The system must adhere to the following logical constraints :

      t → r       i.e., if a threat is detected the system must initiate an automatic response
      f → ¬r      i.e., if a threat is determined to be a false positive automatic actions must be terminated
      r → m       i.e., an automatic response must be accompanied by a manual intervention follow-up
      ¬m ∨ ¬f     i.e., either manual intervention does not occur, or the threat is not a false positive,
                        ensuring consistency in response.

The logical statement ((t → r) ∧ (f → ¬r) ∧ (r → m) ∧ (¬m ∨ ¬f) ∧ t) → (r ∧ m ∧ ¬f) indicates that where the following
proposition is true :

      t           i.e., a threat has been detected

then an automatic response and manual intervention follow-up must be carried out where the detected threat is not a false positive.

Logical Reasoning & Assumption :

t is True (given) and t → r (t implies r). Therefore by modus ponens r is True.

As r is True, and r → m then, by modus ponens m must also be True.

Assuming f is True, i.e. ¬f is False. Therefore (f → ¬r) indicates that both ¬r is True.
However, it has been derived by modes ponens (above) that r is true. This is a contradiction and the original assumption
that f is True must be wrong, i.e., f is False. Therefore ¬f must also be True.

Therefore (r ∧ m ∧ ¬f) is True.

-/

--define key conditions and decisions as propositions
variable (t r f m : Prop)

example (ThreatTriggersAutomaticResponse : t → r)
        (ActionsTerminatedForFalsePositive : f → ¬r)
        (AutomaticResponseAccompaniedByManualIntervention : r → m)
        (ConsistentResponse : ¬m ∨ ¬f)
        (ThreatDetected : t)
        : (r ∧ m ∧ ¬f) :=
          --modes ponens t, t → r gives r i.e. apply t to function
          have pfr : r := ThreatTriggersAutomaticResponse ThreatDetected
          --modes ponens r , r → m gives m i.e. apply r to function
          have pfm : m := AutomaticResponseAccompaniedByManualIntervention pfr
          --m, (¬m ∨ ¬f) gives ¬f, i.e, assume f is true then, using modus ponens, f and (f → ¬r) will give ¬r.
          --r is known to be True this this gives a contradiction. Therefore f must be False. ¬f must be True.
          have pfnotf : ¬f :=
            fun pff : f =>
                   --modes ponens
                   have pfnotr: ¬r := ActionsTerminatedForFalsePositive pff
                   --have contradiction. i.e., pfr and pfnotr
                   show False from absurd pfr pfnotr
          have pfconj : (r ∧ (m ∧ ¬f)) := And.intro pfr (And.intro pfm pfnotf)
          show (r ∧ m ∧ ¬f) from pfconj

/- Part Two -/

opaque conj : Prop -> Prop -> Prop
axiom AxPrTrue   : provable True
axiom AxNotPrFalse : provable False -> False

--conjunction elimination rules
axiom AxConjElimLeft : ∀ x y, provable (conj x y) -> provable x  --for all x and y, if there is a proof of x ∧ y
                                                                 --then x must be provable
axiom AxConjElimRight : ∀ x y, provable (conj x y) -> provable y --for all x and y, if there is a proof of x ∧ y
                                                                 --then y must be provable

--conjunction introduction rule
axiom AxConjIntro : ∀ x y, provable x -> provable y -> provable (conj x y) --for all x and y, if x is provable and y is
                                                                           --provable, then the conjunction x ∧ y is provable


/- Proving that conjunction with False is not provable; and that proving x does not lead to provable False -/
theorem ex1 : ∀ x, ¬provable (conj x False) ∧ provable x -> ¬provable False :=
fun x ConjArgument =>
    --Using proof by contradiction, assume provable False
    (fun h : provable False =>
        --extract each part of the conjunction
        let hr := And.right ConjArgument
        let hl := And.left ConjArgument
        --as have provable False and provable x then can prove the conjunction, i.e., provable (conj x False)
        let hconj := AxConjIntro x False hr h
        --this is a contradiction, therefore the initial assumption provable False must be incorrect,
        --therefore ¬provable False is True
        show False from absurd hconj hl
    )

/- From a contradiction (x ∧ False), anything follows (principle of explosion) -/
--'[I]n constructive logic you can deduce anything from a contradiction' within the context of the contradiction
--3.2.4 Rules for Negation and False
theorem ex2 : ∀ x y, provable (conj x False) -> provable y :=
fun x y ProvableConjArgument =>
   --extract provable x and provable False from provable (conj x False) by elimination
   let pfProvablex := AxConjElimLeft x False ProvableConjArgument
   let pfProvableFalse := AxConjElimRight x False ProvableConjArgument
   --'You can also use the function False.elim to obtain a proof of anything from a proof of False'
   --3.2.4 Rules for Negation and False
   let pfFalse := AxNotPrFalse pfProvableFalse
   False.elim pfFalse
--References : Units 1 - 3 : Additional Guidance And Examples For Lean And Logic Sample Theorem 1
--             Unit 3, Lesson 3 notes

/- Rearranging nested conjunctions using provability -/
--For all ((universal quantifier) x, y and z the following holds, i.e.,  x ∧ (y ∧ z) is equivalent to (y ∧ x) ∧ z
theorem ex3 : ∀ x y z, provable (conj x (conj y z)) -> provable (conj (conj y x) z) :=
fun x y z ProvableConjArgument =>
   --Extract provable x from provable (conj x (conj y z))
   let hx := AxConjElimLeft x (conj y z) ProvableConjArgument
   --Extract provable conj (y z) from provable (conj x (conj y z))
   let hyz := AxConjElimRight x (conj y z) ProvableConjArgument
   --Extract provable y from provable conj (y z)
   let hy := AxConjElimLeft y z hyz
   --Extract provable z from provable conj (y z)
   let hz := AxConjElimRight y z hyz
   --Create provable (conj y x)
   let hyx := AxConjIntro y x hy hx
   --Create provable (conj (conj y x) z)
   AxConjIntro (conj y x) z hyx hz
--Reference : Units 1 - 3 : Additional Guidance And Examples For Lean And Logic Sample Theorem 2

/- Using negation to infer missing provability -/
--For all x and y. if x is provable but the conjunction of x and y (x → y) cannot be proved, then y is not provable
theorem ex4 : ∀ x y : Prop, ¬provable (conj x y) -> provable x -> ¬provable y :=
fun x y NotProvableConjArgument ProvablexArgument =>
    --Using proof by contradiction, assume provable y
    (fun h : provable y =>
        --Given provable x, therefore using conjunction have provable (conj x y)
        let prConj := AxConjIntro x y ProvablexArgument h
        --Given ¬provable (conj x y). Therefore have ¬provable (conj x y) and provable (conj x y) which is a contradiction.
        --Therefore the initial assumption must be False, i.e., provable y is False, so ¬provable y is True.
        show False from absurd prConj NotProvableConjArgument
    )
