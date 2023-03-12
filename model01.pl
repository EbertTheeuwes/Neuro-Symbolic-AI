% kans van symbool
nn(mnist_net,[X],Y,[0,1]) :: digit(X,Y).

% definieer het bord
board([
    [-,-,-],
    [-,-,-],
    [-,-,-]
]).

% definieer de spelers
players([0,1]).

% kijk na of specifiek getal gewonnen heeft
win0(0,Board) :-
    win(0,Board).
win1(1,Board) :-
    \+ win(0,Board),
    win(1,Board).
no_win(2,Board) :-
    \+ win(_,Board).

% speciaal geval voor 2x2 grid
win20(0,Board) :-
    win22(0,Board).
win21(1,Board) :-
    \+ win22(0,Board),
    win22(1,Board).
no_win2(2,Board) :-
    \+ win22(_,Board).

% een speler heeft gewonnen als er drie symbolen van die speler op een rij staan
win(Player,Board) :-
    players(Players),
    member(Player,Players),
    % controleer horizontale lijnen
    (
    Board = [ [Player,Player,Player], _, _ ];
    Board = [ _, [Player,Player,Player], _ ];
    Board = [ _, _, [Player,Player,Player] ];
    % controleer verticale lijnen
    Board = [ [Player,_,_], [Player,_,_], [Player,_,_] ];
    Board = [ [_,Player,_], [_,Player,_], [_,Player,_] ];
    Board = [ [_,_,Player], [_,_,Player], [_,_,Player] ];
    % controleer diagonale lijnen
    Board = [ [Player,_,_], [_,Player,_], [_,_,Player] ];
    Board = [ [_,_,Player], [_,Player,_], [Player,_,_] ]
    ).

% speciaal geval voor 2x2 grid
win22(Player,Board) :-
    players(Players),
    member(Player,Players),
    % controleer horizontale lijnen
    (
    Board = [ [Player,Player], _];
    Board = [ _, [Player,Player]];
    % controleer verticale lijnen
    Board = [ [Player,_], [Player,_]];
    Board = [ [_,Player], [_,Player]];
    % controleer diagonale lijnen
    Board = [ [Player,_], [_,Player]];
    Board = [ [_,Player], [Player,_]]
    ).

% bepaal uitslag van het spel
game_over(Board,Winner) :-
    win0(Winner,Board);win1(Winner,Board);no_win(Winner,Board).

game_over22(Board,Winner) :-
    win20(Winner,Board);win21(Winner,Board);no_win2(Winner,Board).

% check
check1x3grid(A1,A2,A3,Winner) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3),
    game_over([[AR1,AR2,AR3],[-,-,-],[-,-,-]],Winner).

check2x3grid(A1,A2,A3,B1,B2,B3,Winner) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3),
    digit(B1,BR1), digit(B2,BR2), digit(B3,BR3),
    game_over([[AR1,AR2,AR3],[BR1,BR2,BR3],[-,-,-]],Winner).

check3x3grid(A1,A2,A3,B1,B2,B3,C1,C2,C3,Winner) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3),
    digit(B1,BR1), digit(B2,BR2), digit(B3,BR3),
    digit(C1,CR1), digit(C2,CR2), digit(C3,CR3),
    game_over([[AR1,AR2,AR3],[BR1,BR2,BR3],[CR1,CR2,CR3]],Winner).

check2x2grid(A1,A2,B1,B2,Winner) :-
    digit(A1,AR1), digit(A2,AR2),
    digit(B1,BR1), digit(B2,BR2),
    game_over22([[AR1,AR2],[BR1,BR2]],Winner).