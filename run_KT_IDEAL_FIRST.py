import sys, os

for skill in sys.argv[1:]:
    os.system("python fit_KT_IDEAL_model.py 100 1000 5 n " + skill + " 0 3 first")