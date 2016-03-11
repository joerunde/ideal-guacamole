import os, sys

for c in range(3):
    for fname in sys.argv[1:]:
        os.system("python fit_MLFKT_model.py 100 500 5 n " + fname + " " + str(c) + " 3")