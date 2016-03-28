import os, sys

for fname in sys.argv[1:]:
    os.system("python fit_MLFKT_transition_model.py 500 1000 5 n " + fname + " 0 3")