""" T test for significance in predictive power
Loads two pickled numpy arrays containing two models' RMSE distribution over different train/test splits
"""

import numpy as np
import scipy.stats, cPickle, math, sys

if len(sys.argv) < 3:
	print "usage: python rmse_t_test.py skill iter"
	print "Loads <skill><iter> and <skill>_bkt<iter> files, does t test"

skill = sys.argv[1]
itr = sys.argv[2]
lfkt = cPickle.load(open(skill+itr,"r"))
bkt = cPickle.load(open(skill+"_bkt"+itr,"r"))

lfktmean = np.mean(lfkt)
bktmean = np.mean(bkt)

print "average LFKT rmse: " + str(lfktmean)
print "average BKT rmse:  " + str(bktmean)

tind, pind = scipy.stats.ttest_ind(lfkt, bkt)

print "t,p unpaired test: " + str(tind) + "\t" + str(pind)

