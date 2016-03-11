import json
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_groups(data, fname):
    N = 4
    num = len(data)
    width = 1.0 / (num+1)
    fig, ax = plt.subplots()
    rectsl = []
    legend = []
    labels = ['2 states', '3 states', '4 states', '5 states']

    off = 0
    for skill, dat in data.iteritems():
        means = dat['mean']
        stds = dat['std']

        rectsl.append(ax.bar(np.arange(N) + off, means, width, color=(1-off,abs(.5-off),off), yerr=stds))
        legend.append(skill)
        off = off + width

    ax.set_ylabel('RMSE')
    ax.set_title('Student stratified RMSE on sets of skills with varying number of hidden states')
    ax.set_xticks(np.arange(N) + width)
    ax.set_xticklabels(tuple(labels))
    ax.legend(tuple(rectsl), tuple(legend))
    ax.set_ylim(.3,.6)

    fig.savefig(fname, dpi=fig.dpi*2)
    plt.show()


#usage python plot_all_rmse.py [skills]

#look for 2-5 states, 1000 iter

data = {}

fname = 'final_plots/RMSE_plot'
for skill in sys.argv[1:]:
    fname += '_' + skill
    data[skill] = {}
    data[skill]['mean'] = []
    data[skill]['std'] = []

    for c in range(2,6):
        f = open('RMSE_' + skill + '_' + str(c) + 'states_1000iter.json', 'r')
        rmses = json.loads(f.readline())
        data[skill]['mean'].append(np.mean(rmses))
        data[skill]['std'].append(np.std(rmses))

fname += '.png'
plot_rmse_groups(data, fname)