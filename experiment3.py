import os, sys

fname = sys.argv[1]

for c in range(4):
    os.system("python fit_MLFKT_model.py 100 300 5 y " + fname + " " + str(c) + " 5")