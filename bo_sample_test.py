from Metropolis.mlfkt_model import MLFKTModel
from Metropolis.bo_sampler import BOSampler
import sys, json, time, random, os, math
import numpy as np
X = np.loadtxt(open("observations_y_axis.csv","rb"),delimiter=",")
P = np.loadtxt(open("problems_y_axis.csv","rb"),delimiter=",")
model = MLFKTModel(X, P, 0, 0.1)
sampler = BOSampler(model)

for c in range(1000):
    sampler.BO_sample()

