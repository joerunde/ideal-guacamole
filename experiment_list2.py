import os, sys

for fname in sys.argv[1:]:
    for c in range(4):
        os.system("python fit_MLFKT_model.py 200 1000 5 n " + fname + " " + str(c) + " 10")