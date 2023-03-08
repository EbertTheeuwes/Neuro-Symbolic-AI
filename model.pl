% kans van symbool
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

% idempotent
idd(X,Y) :- digit(X,Y).

% even or odd
evenOrOdd(X,0) :- digit(X,Y), 0 is mod(Y, 2).
evenOrOdd(X,1) :- digit(X,Y), 1 is mod(Y, 2).

even(X) :- digit(X,Y), 0 is mod(Y, 2).
odd(X) :- digit(X,Y), 1 is mod(Y, 2).

% addition
addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.
add3(X,Y,Z,R) :- digit(X,X2), digit(Y,Y2), digit(Z,Z2), R is X2+Y2+Z2.
add9(A1,A2,A3,B1,B2,B3,C1,C2,C3,Z) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3),
    digit(B1,BR1), digit(B2,BR2), digit(B3,BR3),
    digit(C1,CR1), digit(C2,CR2), digit(C3,CR3),
    Z is AR1 + AR2 + AR3 + BR1 + BR2 + BR3 + CR1 + CR2 + CR3.

% label number
labelnum(X,Y) :- digit(X,X1), Y is X1.

% werkt voorlopig enkel op een bord van 3X3

% definieer het bord
board([
    [-,-,-],
    [-,-,-],
    [-,-,-]
]).

% definieer de spelers
%players([x,o]).
%players([0,1,2,3,4,5,6,7,8,9]).
players([0,1]).

% een speler heeft gewonnen als er drie symbolen van die speler op een rij staan
win(Player,Board) :-
    % controleer horizontale lijnen
    Board = [ [Player,Player,Player], _, _ ];
    Board = [ _, [Player,Player,Player], _ ];
    Board = [ _, _, [Player,Player,Player] ];
    % controleer verticale lijnen
    Board = [ [Player,_,_], [Player,_,_], [Player,_,_] ];
    Board = [ [_,Player,_], [_,Player,_], [_,Player,_] ];
    Board = [ [_,_,Player], [_,_,Player], [_,_,Player] ];
    % controleer diagonale lijnen
    Board = [ [Player,_,_], [_,Player,_], [_,_,Player] ];
    Board = [ [_,_,Player], [_,Player,_], [Player,_,_] ].

% controleer of het spel is beëindigd (door te kijken of het bord vol is of als een speler heeft gewonnen)
%game_over(Board,/) :-
%    flatten(Board,FlatBoard),
%    \+ member(-, FlatBoard).

game_over(Board,Winner) :-
    players(Players),
    member(Winner,Players),
    win(Winner,Board).

% check
%check([[A1,A2,A3],[B1,B2,B3],[C1,C2,C3]],Winner) :-
check(A1,A2,A3,B1,B2,B3,C1,C2,C3,Winner) :-
    evenOrOdd(A1,AR1), evenOrOdd(A2,AR2), evenOrOdd(A3,AR3),
    evenOrOdd(B1,BR1), evenOrOdd(B2,BR2), evenOrOdd(B3,BR3),
    evenOrOdd(C1,CR1), evenOrOdd(C2,CR2), evenOrOdd(C3,CR3),
    game_over([[AR1,AR2,AR3],[BR1,BR2,BR3],[CR1,CR2,CR3]],Winner).