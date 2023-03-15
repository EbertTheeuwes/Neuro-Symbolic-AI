import matplotlib.pylab as plt

iterations = []
avg_losses = []

data = """Iteration:  100 	s:18.9899 	Average Loss:  0.637992275506258
Iteration:  200 	s:18.4960 	Average Loss:  0.31234880272633747
Iteration:  300 	s:16.2327 	Average Loss:  0.14125247823554674
Iteration:  400 	s:15.9413 	Average Loss:  0.08504163415081109
Iteration:  500 	s:17.4940 	Average Loss:  0.11109366131928254
Iteration:  600 	s:15.2769 	Average Loss:  0.06300733706258754
Iteration:  700 	s:16.3677 	Average Loss:  0.11024337012228443
Iteration:  800 	s:16.3232 	Average Loss:  0.04876505925242199
Iteration:  900 	s:14.4467 	Average Loss:  0.04608629698022497
Iteration:  1000 	s:15.0067 	Average Loss:  0.014772933518752529
Iteration:  1100 	s:14.3211 	Average Loss:  0.07286445549011808
Iteration:  1200 	s:14.6979 	Average Loss:  0.08235710239610895
Iteration:  1300 	s:14.5729 	Average Loss:  0.04133111746913962
Iteration:  1400 	s:14.4256 	Average Loss:  0.07328741438939801
Iteration:  1500 	s:14.2840 	Average Loss:  0.0404803459403611
Iteration:  1600 	s:15.0303 	Average Loss:  0.13600447786031908
Iteration:  1700 	s:14.8333 	Average Loss:  0.07650912263703276
Iteration:  1800 	s:14.5266 	Average Loss:  0.006156637655506874"""

for line in data.split("\n"):
    parts = line.split()
    iterations.append(int(parts[1]))
    avg_losses.append(float(parts[5]))

plt.plot(iterations, avg_losses)
plt.title("Loss as a function of training iterations")
plt.xlabel("training iterations")
plt.ylabel("cross entropy loss")
plt.show()
