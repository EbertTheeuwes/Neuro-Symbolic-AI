nn(mnist_net,[S],D,[0,1])::digit(S,D).
t(1/2)::last_square(0).
last_square(1) :- \+ last_square(0).

win(S1,S2,Winner) :-
    digit(S1,D1),
    digit(S2,D2),
    last_square(D3),
    win([D1,D2,D3],Winner).

win([0,0,0],0).
win([1,1,1],1).

