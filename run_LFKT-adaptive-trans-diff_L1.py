import sys, os

for skill in sys.argv[1:]:
    #we're gonna need a lot of iterations for this
    os.system("python fit_MLFKT_adaptive_transition_difficulty_model.py 200 2000 5 n " + skill + " 0 3 L1")