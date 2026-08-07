% Student ID: 201866539

% BASE DEFINITIONS
form(bread).
form(filling).

% Truth grid: where bread and filling exist
holdsList(bread, [ [3,4], [3,5], [3,6], [3,7] ]).
holdsList(filling, [ [4,4], [4,5], [4,6], [4,7] ]).

% Core logic
holds(F, [R,C]) :- holds1(F, [R,C]).

holds1(Form, State) :- form(Form), holdsList(Form, List), member(State, List).
holds1(neg(F), S) :- \+ holds(F, S).
holds1(and(F1, F2), S) :- holds(F1, S), holds(F2, S).
holds1(or(F1, F2), S) :- holds(F1, S); holds(F2, S).
holds1(implica(F1, F2), S) :- \+ holds(F1, S); holds(F2, S).
holds1(dia(Dir, F), [X,Y]) :- rel(Dir, [X,Y], [NX,NY]), holds(F, [NX,NY]).
holds1(box(Dir, F), [X,Y]) :- \+ rel(Dir, [X,Y], [_,_]); forall(rel(Dir, [X,Y], [NX,NY]), holds(F, [NX,NY])).
holds1(center, [X,Y]) :- num(X), num(Y).
holds1(true, _).

% Directions on the grid
rel(up, [X,Y], [NX,Y]) :- num(X), num(NX), NX is X - 1.
rel(down, [X,Y], [NX,Y]) :- num(X), num(NX), NX is X + 1.
rel(left, [X,Y], [X,NY]) :- num(Y), num(NY), NY is Y - 1.
rel(right, [X,Y], [X,NY]) :- num(Y), num(NY), NY is Y + 1.
rel(center, [X,Y], [X,Y]).

% Grid size (10x10)
num(1). num(2). num(3). num(4). num(5).
num(6). num(7). num(8). num(9). num(10).

% Visualizer
showWhere(F) :- between(1,10,R), showRow(F,R), nl, fail.
showWhere(_).

showRow(F, R) :- between(1,10,C), (holds(F, [R,C]) -> write('■ ') ; write('□ ')), fail.
showRow(_, _).


% ANSWERS

% TASK 1: Bread and filling cannot be in the same place
answer1(neg(and(bread, filling))).

% TASK 2: Filling must have bread immediately above and below
answer2(and(filling, and(dia(up, bread), dia(down, bread)))).

% TASK 3: More than one step below filling must have no bread
answer3(implica(dia(down, dia(down, filling)), neg(bread))).

% TASK 4: If there is filling somewhere, then all positions above it must have no filling
answer4(implica(filling, box(up, neg(filling)))).

% TASK 5.1: Bread on top and bottom with nothing in between
% Solution: Must have filling between bread layers
answer5_1(implica(and(dia(up, bread), dia(down, bread)), dia(center, filling))).

% TASK 5.2: Too many layers of filling
% Solution: If there's filling above and below, the current must not be filling
answer5_2(implica(and(dia(up, filling), dia(down, filling)), neg(filling))).

% TASK 5.3: Filling at the edge without support
% Solution: Filling must be supported by bread above and below
answer5_3(implica(filling, and(dia(up, bread), dia(down, bread)))).