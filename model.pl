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