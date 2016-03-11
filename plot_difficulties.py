import json
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

#renders a bar chart to fname as .png
def plot_bars(x, y, labels, title, xaxis, yaxis, fname):
        fig = plt.figure()
        fig.set_figwidth(12)
        plt.title(title)
        plt.bar(x, y, align='center')
        plt.xticks(x, labels, size='small', rotation='vertical')
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        if not os.path.exists("Plots"):
                        os.mkdir("Plots")
        fig.savefig(fname, bbox_inches='tight')
        plt.close()

def plot_diffs(agg_data, sep_data, fname):
    N = len(agg_data)
    num = 20
    width = 1.0 / (num+1)
    fig, ax = plt.subplots()
    rectsl = []
    legend = []
    labels = []

    #agg_dat = []
    #for skill, dat in agg_data.iteritems():
    #    agg_dat.extend(dat)
    #agg_mean = np.mean(agg_dat)
    #agg_std = np.std(agg_dat)

    off = 0
    col = 0.0
    for skill, dat in agg_data.iteritems():
        #width = 1)
        sep = sep_data[skill]# - np.mean(sep_data[skill])) #/ np.std(sep_data[skill])
        rectsl.append(ax.bar(np.arange(len(dat)) + off, sep, .5, color=(col,1-abs(.5-col),1-col)))
        legend.append(skill + ' separate')
        ax.plot([off + len(dat) + .3], [np.mean(sep)], '.', color=(col,1-abs(.5-col),1-col))

        rectsl.append(ax.bar(np.arange(len(dat)) + off + .5, dat, .5, color=(1-col,abs(.5-col),col)))
        legend.append(skill)
        ax.plot([off + len(dat) + .7], [np.mean(dat)], '.', color=(1-col,abs(.5-col),col))

        off = off + len(dat) + 1
        col += 1.0 / N

    ax.set_ylabel('Difficulty offset')
    ax.set_title('Difficulty offsets of problems learned in separate models \nvs. combined trajectories, using ' + sys.argv[1] + ' hidden states')
    ax.set_xticks(np.arange(N) + width)
    ax.set_xticklabels(tuple(labels))
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(tuple(rectsl), tuple(legend), prop = fontP)
    ax.set_ylim(-3,3)
    ax.set_xlim(0, off+5)

    fig.savefig(fname, dpi=fig.dpi*2)
    #plt.show()


#usage python plot_difficulties.py numstates agg_skill [skills]

diffs = {}

agg = sys.argv[2]
states = int(sys.argv[1])


diffs_data = {} #prob name -> difficulty

dmap = {} #id -> [diffs...] (then avg)
#agg_prob_map = {} #prob name -> id

f = open('problems_idx_' + agg + '.csv', 'r')
agg_probs = json.loads(f.readline())
#for c in range(len(agg_probs)):
#    agg_prob_map[agg_probs[c]] = c

#print agg_probs


f = open('PARAMS_' + agg + '_' + str(states) + 'states_1000iter.json', 'r')
params = json.loads(f.readline())
for p in params:
    for k, v in p.iteritems():
        if 'D_' in k:
            id = int(k[2:])
            if id not in dmap:
                dmap[id] = []
            dmap[id].append(v)
for k,v in dmap.iteritems():
    dmap[k] = np.mean(v)
    diffs_data[agg_probs[k]] = dmap[k]

#print diffs_data

skill_diffs_sep = {}
skill_diffs = {}
for skill in sys.argv[3:]:
    sep_dmap = {}
    sep_diffs_data = {}

    skill_probs = json.load(open('problems_idx_' + skill + '.csv', 'r'))
    #print skill_probs
    skill_params = json.load(open('PARAMS_' + skill + '_' + str(states) + 'states_1000iter.json', 'r'))
    for p in skill_params:
        for k, v in p.iteritems():
            if 'D_' in k:
                id = int(k[2:])
                if id not in sep_dmap:
                    sep_dmap[id] = []
                sep_dmap[id].append(v)
    for k, v in sep_dmap.iteritems():
        #print k
        sep_diffs_data[skill_probs[k]] = np.mean(v)

    skill_diffs[skill] = []
    skill_diffs_sep[skill] = []
    f = open('problems_idx_' + skill + '.csv', 'r')
    skillprobs = json.loads(f.readline())
    for prob in skillprobs:
        skill_diffs[skill].append(diffs_data[prob])
        skill_diffs_sep[skill].append(sep_diffs_data[prob])

    #print skill_diffs[skill]
    #print skill_diffs_sep[skill]

#print skill_diffs

#for k,v in skill_diffs.iteritems():
#    print k
#    print np.mean(v)

plot_diffs(skill_diffs, skill_diffs_sep, 'final_plots/Difficulty_' + agg + '_' + str(states) + 'states.png')