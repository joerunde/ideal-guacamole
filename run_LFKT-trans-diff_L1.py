import sys, os

for skill in sys.argv[1:]:
    os.system("python fit_MLFKT_transition_difficulty_model.py 500 2000 5 n " + skill + " 0 4 L1")