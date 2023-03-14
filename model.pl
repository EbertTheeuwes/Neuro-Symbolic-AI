% kans van symbool
nn(mnist_net,[X],Y,[0,5,9]) :: digit(X,Y).

% definieer het bord
board([
    [-,-,-],
    [-,-,-],
    [-,-,-]
]).

% definieer de spelers
players([5,9]).

% kijk na of specifiek getal gewonnen heeft
win5(5,Board) :-
    win(5,Board).
win9(9,Board) :-
    \+ win(5,Board),
    win(9,Board).
win1(1,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    win(1,Board).
win2(2,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win1(1,Board),
    win(2,Board).
win3(3,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win(1,Board),
    \+ win(2,Board),
    win(3,Board).
win4(4,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win(1,Board),
    \+ win(2,Board),
    \+ win(3,Board),
    win(4,Board).
win6(6,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win(1,Board),
    \+ win(2,Board),
    \+ win(3,Board),
    \+ win(4,Board),
    win(6,Board).
win7(7,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win(1,Board),
    \+ win(2,Board),
    \+ win(3,Board),
    \+ win(4,Board),
    \+ win(6,Board),
    win(7,Board).
win8(8,Board) :-
    \+ win(5,Board),
    \+ win(9,Board),
    \+ win(1,Board),
    \+ win(2,Board),
    \+ win(3,Board),
    \+ win(4,Board),
    \+ win(6,Board),
    \+ win(7,Board),
    win(8,Board).
no_win(0,Board) :-
    \+ win(_,Board).

% speciaal geval voor 2x2 grid
win25(5,Board) :-
    win22(5,Board).
win29(9,Board) :-
    \+ win22(5,Board),
    win22(9,Board).
no_win2(0,Board) :-
    \+ win22(_,Board).

% speciaal geval voor 4x4 grid
win45(5,Board) :-
    win44(5,Board).
win49(9,Board) :-
    \+ win44(5,Board),
    win44(9,Board).
no_win4(0,Board) :-
    \+ win44(_,Board).

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

% speciaal geval voor 4x4 grid
win44(Player,Board) :-
    players(Players),
    member(Player,Players),
    % controleer horizontale lijnen
    (
    Board = [ [Player,Player,Player,Player], _, _, _ ];
    Board = [ _, [Player,Player,Player,Player], _, _ ];
    Board = [ _, _, [Player,Player,Player,Player], _ ];
    Board = [ _, _, _, [Player,Player,Player,Player] ];
    % controleer verticale lijnen
    Board = [ [Player,_,_,_], [Player,_,_,_], [Player,_,_,_], [Player,_,_,_] ];
    Board = [ [_,Player,_,_], [_,Player,_,_], [_,Player,_,_], [_,Player,_,_] ];
    Board = [ [_,_,Player,_], [_,_,Player,_], [_,_,Player,_], [_,_,Player,_] ];
    Board = [ [_,_,_,Player], [_,_,_,Player], [_,_,_,Player], [_,_,_,Player] ];

    % controleer diagonale lijnen
    Board = [ [Player,_,_,_], [_,Player,_,_], [_,_,Player,_], [_,_,_,Player] ];
    Board = [ [_,_,_,Player], [_,_,Player,_], [_,Player,_,_], [Player,_,_,_] ]
    ).

% bepaal uitslag van het spel
game_over(Board,Winner) :-
    win5(Winner,Board);win9(Winner,Board);no_win(Winner,Board).

% speciaal voor 2x2 grid
game_over22(Board,Winner) :-
    win25(Winner,Board);win29(Winner,Board);no_win2(Winner,Board).

% speciaal voor 4x4 grid
game_over44(Board,Winner) :-
    win45(Winner,Board);win49(Winner,Board);no_win4(Winner,Board).

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

check4x4grid(A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,D4,Winner) :-
    digit(A1,AR1), digit(A2,AR2), digit(A3,AR3), digit(A4,AR4),
    digit(B1,BR1), digit(B2,BR2), digit(B3,BR3), digit(B4,BR4),
    digit(C1,CR1), digit(C2,CR2), digit(C3,CR3), digit(C4,CR4),
    digit(D1,DR1), digit(D2,DR2), digit(D3,DR3), digit(D4,DR4),
    game_over44([[AR1,AR2,AR3,AR4],[BR