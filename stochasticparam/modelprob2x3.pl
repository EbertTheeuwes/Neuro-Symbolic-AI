nn(mnist_net,[S],D,[0,1])::digit(S,D).
t(0.6)::last_square_row1(0).
last_square_row1(1) :- \+ last_square_row1(0).
t(0.4)::last_square_row2(0).
last_square_row2(1) :- \+ last_square_row2(0).

win(S1,S2,S4,S5,Winner) :-
    digit(S1,D1),
    digit(S2,D2),
    digit(S4,D4),
    digit(S5,D5),
    last_square_row1(D3),
    last_square_row2(D6),
    win([D1,D2,D3,D4,D5,D6],Winner).

win([0,0,0,_,_,_],0).
win([1,1,1,_,_,_],1).
win([_,_,_,0,0,0],0).
win([1,1,1,_,_,_],1).
