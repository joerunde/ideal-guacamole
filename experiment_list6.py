import os, sys

for fname in sys.argv[1:]:
    os.system("python fit_MLFKT_model.py 200 500 5 n " + fname + " 0 3")