import sys, os

for skill in sys.argv[1:]:
    os.system("python fit_MLFKT_model.py 100 2000 5 y " + skill + " 0 8 L2")