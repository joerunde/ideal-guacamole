#usage python make_response_matrix.py skill

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys

#read in observations and problems
fname = sys.argv[1]
X = np.loadtxt(open("dump/observations_" + fname + ".csv","rb"),delimiter=",")
#load problem IDs for these observations
P = np.loadtxt(open("dump/problems_" + fname + ".csv","rb"),delimiter=",")

rows = X.shape[0]
cols = np.max(P) + 1

response = np.zeros((rows,cols))

for c in range(rows):
    obs = X[c,:]
    probs = P[c,:]

    for i in range(len(obs)):
        if obs[i] < 0:
            #print "done at:", i, obs[i]
            break
        problem = int(probs[i])
        #print obs[i]
        if obs[i] == 0:
            #print "wrong"
            response[c, problem] = -1
        if obs[i] == 1:
            #print "right"
            response[c, problem] = 1

#plot heatmap?

def plot_responses(params, title, fname):
    fig, ax = plt.subplots(1)

    #params = params + 1

    #cmapp = LinearSegmentedColormap.from_list('mycmap', [(2 / 1.0, 'blue'),
    #                                                (1 / 1.0, 'white'),
    #                                               (0 / 1.0, 'red')]
    #                                   )


    plt.pcolor(params, cmap='RdYlGn', vmin=-1, vmax=1)

    plt.colorbar()
    ax.set_aspect('equal')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

plot_responses(response, "Student repsonse matrix", "trans_plots/responses.png")