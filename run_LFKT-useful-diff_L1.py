import sys, os

for skill in sys.argv[1:]:
    os.system("python fit_MLFKT_usefulness_difficulty_model.py 100 500 5 n " + skill + " 0 4 L1")