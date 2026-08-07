/* Assessment 2 Given Prolog Code.
  Has only been tested on swish prolog (web wersion of swi-prolog)
  other prolog may well not draw the pictures well.
*/
% Draws the boxes, white or black if holds or not
whitebox :- format('~s', ["\u25A1"]).
space :- format('~s', [" "]).
blackBox :- format('~s', ["\u25A0"]).
% State and Numbers, which represent the position [1,1], [2,3], etc
numb(0). numb(1). numb(2). numb(3). numb(4). numb(5).
numb(6). numb(7). numb(8). numb(9). numb(10).
state([R,C]) :- numb(R), numb(C).

% Prints a visual grid showing where the formula holds.
showWhere(Formula) :- showRows(Formula, 0).

showRows(Formula, N) :- numb(N), !, showCols(Formula, N, 0),
					  NPlus is N + 1, showRows(Formula, NPlus).
showRows(_, _).

% If the formula holds in a state → ■, otherwise → □.
showCols(Formula, R, C) :- numb(C), holds(Formula, [R,C]), !,
						blackBox, space, CPlus is C + 1,
						showCols(Formula, R, CPlus).
showCols(Formula, R, C) :- numb(C), !,
						whitebox, space, CPlus is C + 1,
						showCols(Formula, R, CPlus).
showCols(_, _, _) :- nl.

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

% Logic modal evaluation holds/2
% Normal operators and, or, negation and implication
holds( or(P,_), S):- holds(P,S).
holds( or(_,Q), S):- holds(Q,S).
holds( and(P,Q), S) :- holds(P,S), holds(Q,S).
holds( not(P), S) :- form(P), chkstate(S), !, not(holds(P,S)).
holds( implies(P, Q), S) :- holds(or(not(P),Q),S).

% Modal Operators Diammond and Box states, not sure about hold statement
holds( dia(R, F), S) :- rel(R, S, T), holds(F,T).
holds( box(R,F), S) :- foreach(rel(R,S,T),holds(F,T)).

form(bread).
form(filling).
% Added relations
form(wrap).
form(pancake).

% Normal operators and, or, negation and implication with the last 2 modal operators
form(or(X,Y)) :- form(X), form(Y).
form(and(X,Y)) :- form(X), form(Y).
form(implies(X,Y)) :- form(X), form(Y).
form(not(X)) :- form(X).
% Modal operators
form(dia(_,X)) :- form(X).
form(box(_,X)) :- form(X).

form(false).
chkstate(X) :- state(X), !.
chkstate(X) :- write('unknown state : '),write(X), nl, fail.

% Spatial relationships
rel(above, [X,Y], [Z,Y]) :- numb(X), numb(Y), numb(Z), X < Z.
rel(oneabove, [X,Y], [Z,Y]) :- numb(X), numb(Y), numb(Z), Z is X + 1.
rel(below, [X,Y], [Z,Y]) :- rel(above, [Z,Y], [X,Y]).
rel(onebelow, [X,Y], [Z,Y]) :- rel(oneabove, [Z,Y], [X,Y]).
rel(isLeftOf, [X,Y], [X,Z]) :- numb(X), numb(Y), numb(Z), Y < Z. % 
rel(isRightOf, [X,Y], [X,Z]) :- rel(isLeftOf, [X,Z], [X,Y]).

%*********************************************************************

%TASK 1: Any place that has bread cannot also have filling and no place that has filling can also have bread.
answer1(and(implies(bread, not(filling)), implies(filling, not(bread)))).
 
/* This definition relates to the original definition described in wikipedia about a sandwich, a basic food consistent of a filling between 2 breads 
 * by enforcing that filling needs to be between breads and not the filling surrounding the breads.
 */
/* Food has always been complex and diverse around the world, from which in some places did not have the same resources as European culture like wheat.
 * In the case of Mexico, locals had corn and created the Tortilla, which is type of flat bread that based the entire Mexican cuisine. With colonization Spanish colonizers
 * were able to introduce the wheat, which later would be the base of the burrito toritilla, a new type of sandwich with a different kind of ensamblement non conventional,
 * which will be later discussed on in TASK 5.
*/

%TASK 2:
%Filling must have bread immediately above it and below it.

answer2( implies(filling, and(dia(oneabove, filling, bread), box(onebelow, filling, bread)))).
/* In my previous answer, what I am trying to say is that filling implies to have a bread above and below the filling. I should use box, but, there is a debate since midlle ages
 * about sandiwches without a top or bottom, in my conception a sandwich need to have a floor and a celing but in this case if we want to consider open sandwiches,
 * we will allow the DIAMOND option in the bread above and make obligatory the bottom bread.
*/

%TASK 3:
%Going more than one row below filling there should be no bread.

answer3( implies(filling, not(dia(onebelow, dia(below, bread) ))) ).
/* In my previous code basically, my formula the first DIAMOND suggest that I set the logic I level bellow the filling which is the bread, later I set anothoer DIAMONd
 * through nesting it to the previous so in the logic we from the level after onwards with bread, then we decide to negate all from the begining to state that after the bread 
 * level exists the posibility of no bread; Except for triple deckers sandiwch onwards which are mentioned in the Wikipedia article.
 * */

/* My previous code considers that the second line after the filling does not contain any kind of bread, in this case I leave the posibility of layering by allowing
 * DIAMOND statements, which can make the creation of a CLUB SANDWICH. 
 */


%TASK 4:
%If somewhere has filling, then everywhere which is in any row anywhere above cannot have filling.

answer4( implies(dia(filling), box(above,not(filling)) ) ).

/*
  Tecnically the task requieres that we cannot stack fillings, which we try to represent, of course exist filling stacking through triple deckers onwards (club sandwich +), 
  as well since we can have open-faced sandwiches, we could stack fillings due that we would not have any limit but the filling may fall apart from it.
*/


%TASK 5

%ANSWER 5.1
%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.

%Filling must have wrap immediately surrounding it.

% Cultural - A wrap? Problematic, example burritos 
answer51( implies(filling, and(
                               box(oneabove, filling, wrap), 
                               box(onebelow, filling, wrap),
                               box(isLeftOf, filling, wrap),
                               box(isRightOf, filling, wrap)
                               ))).
/* In my previous code I decided to model wraps in general where can be problematic to consider them as Sanwiches, usually a sandwich does not wrapp the filling , from which
 * wraps are debatable in their own category, like burritos, durüm kebab´s, etc.
*/


%ANSWER 5.2
%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.

% Structural - double bread sandwich? Does not specify 2 lines below that can be bread the filling.
answer52( implies(filling, not(box(onebelow, box(onebelow, box(below, bread) )))) ).
% Sandwiches should not have double layering always, would be a thick sandwich, what can be argued is that triple deckers do the same job, the difference
% is where the sandwiches do not have a double layering without a layer of filling between them, lets call double bread for the same amount of filling.


%ANSWER 5.3
%Give a formula below that prevents your example when it holds in EVERY state.
%That is: If your formula holds everywhere than this kind of problematic sandwich cannot happen.

% Ingredient-based - using pancakes as bread?
answer53( implies(filling, and(
                               box(oneabove, filling, pancake), 
                               box(onebelow, filling, pancake),                               
                               ))).

/* Pancakes are a type of bread, right? Thus we do not use them as the first layer for sandwiches, even bagels are used but pancakes? Sounds a bit crazy, but it should be 
 * considered a sandwich because is a type of bread. This type of ingredient is never used for sandwiches.
*/




