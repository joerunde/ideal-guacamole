import numpy as np
from hmmlearn import hmm

np.random.seed()

model = hmm.MultinomialHMM(n_components=3)

model.startprob_ = np.array([0.8, 0.15, 0.05])
model.transmat_ = np.array([[0.8, 0.15, 0.05],
                            [0.0, 0.70, 0.30],
                            [0.0, 0.00, 1.00]])

model.emissionprob_ = np.array([[0.9, 0.1],
                                [0.5, 0.5],
                                [0.1, 0.9]])

X = None
lens = []
States = []

for c in range(100):
    x, states = model.sample(n_samples=18)
    lens.append(18)
    if X is not None:
        X = np.concatenate([X, x])
    else:
        X = x
    States.append(states)
print X


model2 = hmm.MultinomialHMM(n_components=3, verbose=True, n_iter=500)
model2.fit(X, lens)

print model2.startprob_
print model2.transmat_
print model2.emissionprob_