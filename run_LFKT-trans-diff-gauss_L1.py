import sys, os

for skill in sys.argv[1:]:
    os.system("python fit_MLFKT_transition_difficulty_gaussian_model.py 100 500 5 n " + skill + " 0 4 L1")