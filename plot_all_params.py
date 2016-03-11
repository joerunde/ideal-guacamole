import json
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_param_groups(data, skill, fname):

    fig, a = plt.subplots(3,4)

    for c in range(4):
        a[0][c].pcolor(data['pi'][c+2], cmap=plt.cm.Blues)
        a[0][c].invert_yaxis()
        a[0][c].get_xaxis().set_ticks([])
        a[0][c].get_yaxis().set_ticks([])
        a[0][c].set_title('Pi:')
    for c in range(4):
        a[1][c].pcolor(data['trans'][c+2], cmap=plt.cm.Blues)
        a[1][c].invert_yaxis()
        a[1][c].get_xaxis().set_ticks([])
        a[1][c].get_yaxis().set_ticks([])
        a[1][c].set_title('Trans:')
    for c in range(4):
        a[2][c].pcolor(data['emit'][c+2], cmap=plt.cm.Blues)
        a[2][c].invert_yaxis()
        a[2][c].get_xaxis().set_ticks([])
        a[2][c].get_yaxis().set_ticks([])
        a[2][c].set_title('Emit:')

    fig.suptitle("Learned parameters for " + skill, fontsize=14)

    fig.savefig(fname, dpi=fig.dpi*2)
    #plt.show()


#usage python plot_all_rmse.py [skills]

#look for 2-5 states, 1000 iter

data = {}
data['pi'] = {}
data['trans'] = {}
data['emit'] = {}

for skill in ['xy', 'x_axis', 'y_axis', 'descrip', 'h_to_d', 'd_to_h', 'css', 'center', 'shape', 'spread', 'histogram',
              'circleall', 'circlediam', 'circlearea', 'circlecir']:

    fname = 'final_plots/PARAMS_plot_' + skill + '.png'

    for c in range(2,6):
        f = open('PARAMS_' + skill + '_' + str(c) + 'states_1000iter.json', 'r')
        params = json.loads(f.readline())
        pis = []
        for p in params:
            pis.append(p['Pi'])
        pi = np.mean(pis,0)
        #print pi
        data['pi'][c] = np.array([pi])

        transs = []
        for p in params:
            transs.append(p['Trans'])
        trans = np.mean(transs,0)
        #print trans
        data['trans'][c] = trans

        emits = []
        for p in params:
            emits.append(p['Emit'])
        emit = np.mean(emits,0)
        #print emit
        data['emit'][c] = emit

    plot_param_groups(data, skill, fname)