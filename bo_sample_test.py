from Metropolis.mlfkt_model import MLFKTModel
import sys, json, time, random, os, math
import numpy as np
X = np.loadtxt(open("observations_y_axis.csv","rb"),delimiter=",")
P = np.loadtxt(open("problems_y_axis.csv","rb"),delimiter=",")
model = MLFKTModel(X, P, 1, 0.1)


gg = model.get_params_for_BO()

ptest = [0,0,0,0,0,1,0.2,0.2,0.2,0.2,0.2,0,0,0]

y = model.evaluate_params_from_BO(ptest)

print y

