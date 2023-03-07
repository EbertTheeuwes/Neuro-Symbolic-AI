% Kans van symbool
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

% werkt voorlopig enkel op een bord van 3X3

% definieer het bord
board([
    [-,-,-],
    [-,-,-],
    [-,-,-]
]).

% definieer de spelers
players([x,o]).

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
game_over(Board,/) :-
    flatten(Board,FlatBoard),
    \+ member(-, FlatBoard).

game_over(Board,Winner) :-
    players(Players),
    member(Winner,Players),
    win(Winner,Board).

% check
check([[A1,A2,A3],[B1,B2,B3],[C1,C2,C3]],Winner) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3),
    digit(B1,BR1), digit(B2,BR2), digit(B3,BR3),
    digit(C1,CR1), digit(C2,CR2), digit(C3,CR3),
    game_over([[AR1,AR2,AR3],[BR1,BR2,BR3],[CR1,CR2,CR3]],Winner).