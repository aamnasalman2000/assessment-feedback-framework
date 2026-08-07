% Assessment 2 Given Prolog Code.
% Has only been tested on swish prolog (web wersion of swi-prolog)
% other prolog may well not draw the pictures well.

whitebox :- format('~s', ["\u25A1"]).
space    :- format('~s', [" "]).
blackBox :- format('~s', ["\u25A0"]).

numb(0). numb(1). numb(2). numb(3). numb(4). numb(5).
numb(6). numb(7). numb(8). numb(9). numb(10). 

state([R,C]) :- numb(R),  numb(C).

showWhere(Formula) :- showRows(Formula, 0).

showRows(Formula, N) :- numb(N), !, showCols(Formula, N, 0),
                        NPlus is N + 1, showRows(Formula, NPlus).
showRows(_, _).

showCols(Formula, R, C) :- numb(C), holds(Formula, [R,C]), !, 
                           blackBox, space, CPlus is C + 1,
                           showCols(Formula, R, CPlus).

showCols(Formula, R, C) :- numb(C), !, 
                           whitebox, space, CPlus is C + 1, 
                           showCols(Formula, R, CPlus).
showCols(_, _, _) :- nl.

%holdsList(filling, [ [2,4], [2,5], [2,6], [2,7] ]).
%holdsList(bread, [ [3,4], [3,5], [3,6], [3,7] ]). 
%holdsList(filling, [ [4,4], [4,5], [4,6], [4,7] ]). 
%holdsList(bread, [ [5,4], [5,5], [5,6], [5,7] ]). 
%holdsList(bread, [ [6,4], [6,5], [6,6], [6,7] ]). 


% holdsList is only provided as a convenience for creating 
% scenarios for testing. You should use it to test your understanding 
% of the problem and your solution.
% With no clauses for holdsList swish will complain, but we
% can get round this with the following:
% 
% holdsList(false, [ ]).
% 
% This means that the list of locations where the formula false holds 
% is the empty list.

holdsList(false, [ ]).

holds(Formula, State) :- holdsList(Formula, StateList),
                         member(State,StateList).

holds( or(P,_), S):- holds(P,S).
holds( or(_,Q), S):- holds(Q,S).

holds( and(P,Q), S) :- holds(P,S), holds(Q,S).

holds( not(P), S) :- form(P), chkstate(S), !, not(holds(P,S)).

holds( implies(P, Q), S) :- holds(or(not(P),Q),S).

holds( dia(R, F), S) :- rel(R, S, T), holds(F,T).

holds( box(R,F), S) :- foreach(rel(R,S,T),holds(F,T)).

form(bread).
form(filling).
form(or(X,Y))  :- form(X), form(Y).
form(and(X,Y)) :- form(X), form(Y).
form(implies(X,Y)) :- form(X), form(Y).
form(not(X))   :- form(X).
form(dia(_,X)) :- form(X).
form(box(_,X)) :- form(X).
form(false).

chkstate(X) :- state(X), !.
chkstate(X) :- write('unknown state : '),write(X), nl, fail. 

rel(above, [X,Y], [Z,Y]) :- numb(X), numb(Y), numb(Z), X < Z.
rel(oneabove, [X,Y], [Z,Y]) :- numb(X), numb(Y), numb(Z), Z is X + 1.

rel(below, [X,Y], [Z,Y]) :- rel(above, [Z,Y], [X,Y]).
rel(onebelow, [X,Y], [Z,Y]) :- rel(oneabove, [Z,Y], [X,Y]).

rel(isLeftOf, [X,Y], [X,Z]) :- numb(X), numb(Y), numb(Z), Y < Z.

rel(isRightOf, [X,Y], [X,Z])    :- rel(isLeftOf, [X,Z], [X,Y]).




%*********************************************************************

% Tasks and answers to 1-4
% anything logically equivalent would get full marks

%TASK 1:
%Any place that has bread cannot also have filling and no place that has filling can also have bread.

%Answers are allowed to be more than one line as long is the whole file submitted has no syntax errors.
%To answer task 1: remove the % from start of the line below and insert your answer between
%the parentheses. 
%For example, If your answer is or(bread, filling) then where it says "%answer1( )." you need
%to make it say "answer1(or(bread, filling))." and the quotes are not part of the answer.

answer1(
        and( implies(bread,   not(filling)),
             implies(filling, not(bread)) )
       ).

/* ------------------------------------------------------------------

	Explanation:
	The formula forces full pair-wise mutual exclusivity: any state containing bread cannot also
	contain filling, and vice-versa. Wikipedia (Page from Task1) defines a sandwich as
    food "between two pieces of bread" – clearly separating bread from its filling role. 
    A classic peanut-butter-and-jelly sandwich listed on that same page fits
	the rule perfectly: the bread slices occupy their own layer and the PB&J 
    spread is the distinct filling, so no single spot is both bread AND filling.

------------------------------------------------------------------ */


%TASK 2:
%Filling must have bread immediately above it and below it.

%Answers are allowed to be more than one line as long is the whole file submitted has no syntax errors.
%To answer task 2: remove the % from start of the line below and insert your answer between
%the parentheses. 


openSandwich(S, filling) :-
        holds(filling, S),
        holds( dia(oneabove, bread), S),          % toast immediately under
        holds( box(below, not(bread)), S).        % NO bread anywhere above

/* make it accessible for dia */
:- discontiguous rel/3.
rel(openSandwich, S, S) :- openSandwich(S, filling).

answer2(
        implies(
            filling,
            and( dia(above, bread),                  % bread below (always)
                 or( dia(below, bread),              % bread above  (closed)
                     dia(openSandwich, filling) ) )  % or it’s open-face
        )
       ).

/* ------------------------------------------------------------------

	Explanation:
	It guarantees bread lies immediately below every filling square 
    and demands bread immediately above – unless open sandwiches, 
    following the "filling must be enclosed by bread unless open sandwich"
    statement. 'openSandwich/2' (new, arity 2) marks a filling tile that has
    bread one row beneath and none above, matching the description of open-face sandwich.

------------------------------------------------------------------ */


%TASK 3:
%Going more than one row below filling there should be no bread.

%Answers are allowed to be more than one line as long is the whole file submitted has no syntax errors.
%To answer task 3: remove the % from start of the line below and insert your answer between
%the parentheses. 

answer3(
        implies(
            filling,
            dia( oneabove,                          % jump to the cell 1 row below
                 and( bread,                        % that cell is bread
                      box( above, not(bread) ) ) )  % every deeper cell is NOT bread
        )
       ).

/* ------------------------------------------------------------------
 * 
	Explanation:
    box p is true in every state reachable by relation r.
    Here r = above/3, so  box(above,not(bread)) forbids bread in any
	deeper row. The  problem with this is that a Double Decker Sandwich (like a club sandwich)
    breaks the rule, as described in the Wikipedia page, it "adds another
    piece of bread" below the first bread-filling pair.
    
------------------------------------------------------------------ */


%TASK 4:
%If somewhere has filling, then everywhere which is in any row anywhere above cannot have filling.

%Answers are allowed to be more than one line as long is the whole file submitted has no syntax errors.
%To answer task 4: remove the % from start of the line below and insert your answer between
%the parentheses. 


answer4(
        implies( filling,                    % if a cell contains filling
                 box( below,                 % then for *every* cell that is
                      not(filling) ) )       % reachable by going down (i.e. any row above),
       ).                                     % filling must be absent

/* ------------------------------------------------------------------

	Explanation:
	Same box logic as before, but here box(below,not(filling))
    uses rel = below/3 (which points to higher rows, i.e. cells above the
    current one).  So the implication enforces the statement "If somewhere has filling, 
	then in everywhere which is in any row anywhere above cannot have filling."
	Again, the problem is that multi-deck clubs would break this, because their
	upper fillings have more filling still higher up.
    
------------------------------------------------------------------ */


%TASK 5
%Give three different examples of ways in which the above constraints could all be satisfied but 
%it can be argued that it is not really a sandwich. Hint: think about size of sandwiches may help.

%Your answer will be English sentences describing a problematic sandwich. 
%For example: "A sandwich that is knotted" rather than listing places which are bread and filling.
%That was just a silly example. It is unclear what a knotted sandwich would be and it is nothing like correct.

%In task 5 only (not in tasks 1 to 4) you may choose to define one or more extra relations. 
%There are no extra marks for doing this - just a possibility if you find an answer that needs it.
%Just add your relations after the ones given above.

%Answers are allowed to be more than one line as long is the whole file submitted has no syntax errors.

%ANSWER 5.1
%write your answer here as a comment, as many lines as you need but start each with the % for comments

%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.
%Remove the % and insert your formula in the space in the next line.

/* 
	Needle sandwich:
	- One cell wide sandwich
	- Tasks 1-4 all vacuously true (no size requirements). 
	- A sandich that narrow is not considered to be a proper sandwich.
    
	Prevention: If a square is bread, at least one adjacent square (left or right)
    must also be bread. If a square is filling, at least one adjacent square must also  
    be filling. This forces every bread or filling run to be ≥ 2 cells wide
    
*/

answer51(
   and(  implies( bread,
                  or( dia(isLeftOf,  bread),
                      dia(isRightOf, bread) ) ),
        implies( filling,
                 or( dia(isLeftOf,  filling),
                     dia(isRightOf, filling) ) ) )
).

%ANSWER 5.2
%write your answer here as a comment, as many lines as you need but start each with the % for comments

%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.
%Remove the % and insert your formula in the space in the next line.

/*
 	All-bread sandwich:
	- Grid filled only with bread squares.
	- Tasks 1-4 all vacuously true (no filling appears).
	- A plate of plain bread is not a sandwich.

	Prevention: Each bread square must have filling directly above or
	below, forcing at least one layer of filling somewhere.
*/


answer52(
        implies( bread,
                 or( dia(above, filling),
                     dia(below, filling) ) )
).

%ANSWER 5.3
%write your answer here as a comment, as many lines as you need but start each with the % for comments

%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.
%Remove the % and insert your formula in the space in the next line.

/*
	Swiss-cheese slice
 	- Bread , gap , bread in the same row (e.g. [3,4] – [3,5]=empty – [3,6]).
 	- Tasks 1-4 still pass, but a holed slice feels wrong as a sandwich.

 	Prevention: If a bread square can ‘see’ another bread square two
 	columns to the right, the square in between must also be bread.”
*/

%ANSWER 5.3
answer53(
        implies( bread,
                 not( dia(isRightOf,                     % step 1 is there a gap?
                          and( not(bread),               %   middle cell empty
                               dia(isRightOf, bread) )   %   step 2 is it bread
                        ) )
               )
).


