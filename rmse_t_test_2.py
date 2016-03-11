""" T test for significance in predictive power
"""

import numpy as np
import scipy.stats, json, sys

#usage: python rmse_t_test.py agg [skills]

def name(skill, states):
    return 'RMSE_' + skill + '_' + str(states) + 'states_1000iter.json'

agg_skill = sys.argv[1]
skills = sys.argv[2:]

rmse_dict = {}
rmse_dict[agg_skill] = [json.load(open(name(agg_skill,c),'r')) for c in range(2,6)]
for skill in skills:
    rmse_dict[skill] = [json.load(open(name(skill,c),'r')) for c in range(2,6)]

for skill in sys.argv[1:]:
    for c, a in enumerate(rmse_dict[skill]):
        for i, b in enumerate(rmse_dict[skill]):
            if i < c:
                continue
            t,p = scipy.stats.ttest_ind(a, b)
            if p < 0.1:
                if np.mean(a) < np.mean(b):
                    print skill + '\t' + str(c+2) + '\t' + str(np.mean(a)) + '\t' + skill + '\t' + str(i+2) + '\t' + str(np.mean(b)) + '\t' + str(p)
                else:
                    print skill + '\t' + str(i+2) + '\t' + str(np.mean(b)) + '\t' + skill + '\t' + str(c+2) + '\t' + str(np.mean(a)) + '\t' + str(p)

for c, a in enumerate(rmse_dict[agg_skill]):
    for skill in skills:
        for i, b in enumerate(rmse_dict[skill]):
            t,p = scipy.stats.ttest_ind(a, b)
            if p < 0.1:
                if np.mean(a) < np.mean(b):
                    print agg_skill + '\t' + str(c+2) + '\t' + str(np.mean(a)) + '\t' + skill + '\t' + str(i+2) + '\t' + str(np.mean(b)) + '\t' + str(p)
                else:
                    print skill + '\t' + str(i+2) + '\t' + str(np.mean(b)) + '\t' + agg_skill + '\t' + str(c+2) + '\t' + str(np.mean(a)) + '\t' + str(p)



"""
#test each against 2 vs 3 states
for skill in sys.argv[1:]:
    a = json.load(open(name(skill,2),'r'))
    b = json.load(open(name(skill,3),'r'))
    t,p = scipy.stats.ttest_ind(a, b)
    if p < 0.1:
        if np.mean(a) < np.mean(b):
            print skill + '\t2\t' + str(np.mean(a)) + '\t' + skill + '\t3\t' + str(np.mean(b)) + '\t' + str(p)
        else:
            print skill + '\t3\t' + str(np.mean(b)) + '\t' + skill + '\t2\t' + str(np.mean(a)) + '\t' + str(p)

#test agg against other skills- 2 states
for skill in skills:
    a = json.load(open(name(agg_skill,2),'r'))
    b = json.load(open(name(skill,2),'r'))
    t,p = scipy.stats.ttest_ind(a, b)
    if p < 0.1:
        if np.mean(a) < np.mean(b):
            print agg_skill + '\t2\t' + str(np.mean(a)) + '\t' + skill + '\t2\t' + str(np.mean(b)) + '\t' + str(p)
        else:
            print skill + '\t2\t' + str(np.mean(b)) + '\t' + agg_skill + '\t2\t' + str(np.mean(a)) + '\t' + str(p)

#test agg against other skills- 3 states
for skill in skills:
    a = json.load(open(name(agg_skill,3),'r'))
    b = json.load(open(name(skill,3),'r'))
    t,p = scipy.stats.ttest_ind(a, b)
    if p < 0.1:
        if np.mean(a) < np.mean(b):
            print agg_skill + '\t3\t' + str(np.mean(a)) + '\t' + skill + '\t3\t' + str(np.mean(b)) + '\t' + str(p)
        else:
            print skill + '\t3\t' + str(np.mean(b)) + '\t' + agg_skill + '\t3\t' + str(np.mean(a)) + '\t' + str(p)
"""

