""" run the damn thing a bunch
Do multiple LFKT and BKT fits (different train/test split each time) over all skills
"""

import subprocess as sub
import time
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"x axis\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"x axis\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"center\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"center\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"h to d\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"h to d\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"d to h\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"d to h\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"spread\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"spread\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"histogram\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"histogram\"", shell=True)
time.sleep(300)
for c in range(15):
	sub.Popen("python fit_LFKT_model.py 500 500 5 n \"shape\"", shell=True)
	time.sleep(0.1)
	sub.Popen("python fit_LFKT_model.py 500 500 5 y \"shape\"", shell=True)
print "fired off a shitload of processes, good luck"
