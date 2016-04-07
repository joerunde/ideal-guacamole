import numpy as np
from hmmlearn import hmm

np.random.seed()

model = hmm.MultinomialHMM(n_components=5)

model.startprob_ = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
model.transmat_ = np.array([[0.80, 0.20, 0.00, 0.00, 0.00],
                            [0.00, 0.80, 0.20, 0.00, 0.00],
                            [0.00, 0.00, 0.80, 0.20, 0.00],
                            [0.00, 0.00, 0.00, 0.80, 0.20],
                            [0.00, 0.00, 0.00, 0.00, 1.00]])

model.emissionprob_ = np.array([[0.9, 0.1],
                                [0.7, 0.3],
                                [0.5, 0.5],
                                [0.3, 0.7],
                                [0.1, 0.9]])

X = None
lens = []
States = []

f = open('observations_5state.csv', 'wt')
g = open('problems_5state.csv', 'wt')

a,b = model.sample()
print a,b
c,d = model.sample(5)
print c,d

for c in range(200):
    length = 25
    x, states = model.sample(length)

    for o in x:
        f.write(str(o) + ',')
        g.write('0,')
    f.write('-1\n')
    g.write('-1\n')

    lens.append(length)
    if X is not None:
        X = np.concatenate([X, x])
    else:
        X = x
    States.append(states)
print X
f.close()
g.close()

"""
model2 = hmm.MultinomialHMM(n_components=5, verbose=True, n_iter=500)
model2.fit(X, lens)

np.set_printoptions(5)

print np.round(model2.startprob_,2)
print np.round(model2.transmat_,2)
print np.round(model2.emissionprob_,2)"""