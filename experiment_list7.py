import os, sys

for fname in sys.argv[1:]:
    os.system("python fit_MLFKT_skill_model.py 200 1000 5 n " + fname + " 0 3")