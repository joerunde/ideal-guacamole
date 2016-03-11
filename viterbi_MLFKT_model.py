""" Run an MCMC fit on the MLFKT model (maybe in BKT mode though)
    Does all the train/test splitting and calculates RMSE over post test
"""

from Metropolis.mcmc_sampler import MCMCSampler
from Metropolis.mlfkt_model import MLFKTModel
import sys, json, time, random, os, math
import numpy as np

print "usage: python viterbi_MLFKT_model.py burnin iterations _ bkt(y/n) skills num_intermediate_states"


skill = sys.argv[5]
intermediate_states = int(sys.argv[6])
total_states = intermediate_states + 2
fname = skill.replace(" ","_")
fname = fname.replace("\"","")
"""
try:
    os.system("python bkt_data_split.py " + fname + ".csv \"" + skill + "\"")
except Exception as e:
    print str(e)
    pass
"""

#time.sleep(1)

#load observations
X = np.loadtxt(open("observations_" + fname + ".csv","rb"),delimiter=",")
#load problem IDs for these observations
P = np.loadtxt(open("problems_" + fname + ".csv","rb"),delimiter=",")

# find baseline rows:
for c in range(X.shape[0]):
    l = 0
    for i in range(X.shape[1]):
        if X[c,i] > -1:
            l += 1
    X[c,-1] = l

maxl = max(X[:,-1])
print maxl

newX = []
newP = []
for c in range(X.shape[0]):

    if X[c,-1] >= maxl-1:
        newX.append(X[c,:])
        newP.append(P[c,:])
X = np.array(newX)
P = np.array(newP)



start = time.time()

if 'y' in sys.argv[4]:
    model = MLFKTModel(X, P, intermediate_states, 0)
else:
    model = MLFKTModel(X, P, intermediate_states, 0.1)

mcmc = MCMCSampler(model, 0.15)

burn = int(sys.argv[1])
for c in range(20):
    mcmc.burnin(int(math.ceil((burn+0.0) / 20)))
    print("finished burn-in #: " + str((c+1)*burn/20))

num_iterations = int(sys.argv[2])
loop = 20
per_loop = int(math.ceil((num_iterations+0.0) / loop))
for c in range(loop):
    a = time.time()
    mcmc.MH(per_loop)
    b = time.time()
    print("finished iteration: " + str((c+1)*per_loop) + " in " + str(int(b-a)) + " seconds")

end = time.time()

print("Finished burnin and " + str(num_iterations) + " iterations in " + str(int(end-start)) + " seconds.")

model.viterbi()


