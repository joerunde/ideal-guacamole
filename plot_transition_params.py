import json
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np



def plot_param_vector(params, title, fname, minv=-.3, maxv=.3, labels=()):
    fig, ax = plt.subplots(1)
    if minv < 0:
        plt.pcolor(params, cmap='RdBu', vmin=minv, vmax=maxv)
    else:
        plt.pcolor(params, cmap='Blues', vmin=minv, vmax=maxv)
    plt.colorbar()
    ax.set_aspect('equal')
    ax.set_title(title)
    if len(labels) == 0:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    else:
        ax.set_xticklabels(labels, rotation=60)
        ax.set_xticks(np.arange(len(labels)) + .5)
        ax.set_yticklabels(list(reversed(labels)))
        ax.set_yticks(np.arange(len(labels)) + .5)

    plt.tight_layout()
    plt.savefig(fname)


# for each skill and transition setting, get the
# transition parameters
# transition model RMSE
# LFKT (baseline) model RMSE

for sk in ['x_axis', 'y_axis', 'center', 'shape', 'spread', 'h_to_d', 'd_to_h', 'histogram', 'xy', 'descrip', 'css', 'whole_tutor']:
    for set in ['first', 'second']:
        try:
            rmse = np.mean(json.load(open("apr3_exps/RMSE_" + sk + "_L1_" + set + "_trans_2states_2000iter.json","r")))
            baseline = np.mean(json.load(open("apr3_exps/RMSE_" + sk + "_bkt_2states_2000iter.json", "r")))
            params = json.load(open("apr3_exps/PARAMS_" + sk + "_L1_" + set + "_trans_2states_2000iter.json","r"))

            print sk, rmse, baseline
        except Exception as e:
            print e
            continue

        transition_params = {}

        for k in params[0].iterkeys():
            if "D_" in k:
                transition_params[k] = []
        for p in params:
            for k,v in p.iteritems():
                if "D_" in k:
                    transition_params[k].append(v)
        vec = []
        for k,v in transition_params.iteritems():
            print k
            vec.append(np.mean(v))
        vec = np.array([vec])
        print

        fname = "trans_plots/" + sk + "_" + set + ".png"
        title = "Transition parameters learned for " + sk + "_" + set + ".\nRMSE improvement: " + str(int(1000 * (baseline - rmse))/10.0) + "e-2"
        plot_param_vector(vec, title, fname)


#add in other stuff here
# KT params?
for sk in ['xy', 'descrip', 'css', 'whole_tutor']:
    try:
        rmse = np.mean(json.load(open("apr3_exps/RMSE_useful_" + sk + "_2states_500iter.json","r")))
        #baseline = np.mean(json.load(open("apr3_exps/RMSE_" + sk + "_bkt_2states_2000iter.json", "r")))
        params = json.load(open("apr3_exps/PARAMS_uesful_" + sk + "_2states_500iter.json","r"))
        params = params[4:]

        print sk, rmse#, baseline
    except Exception as e:
        print sk, e
        continue

    transition_params = {}

    for k in params[0].iterkeys():
        if "U" in k:
            transition_params[k] = []
    for p in params:
        for k,v in p.iteritems():
            if "U" in k:
                transition_params[k].append(v)

    if sk == 'xy' or sk == 'descrip':
        dim = 2
    if sk == 'css':
        dim = 3
    if sk == 'whole_tutor':
        dim = 8

    mat = np.zeros((dim,dim))

    vec = []
    for k,v in transition_params.iteritems():
        #print k
        val = np.mean(v)
        x = int(k[2])
        y = int(k[4])

        if x == 4 and y == 3:
            print val

        if x == 2 and y == 6:
            print val

        mat[dim-1-x,y] = val
    print mat
    print

    fname = "trans_plots/KT/" + sk + ".png"
    title = "Knowledge transfer parameters\n" #+ "_" + set + ".\nRMSE improvement: " + str(int(1000 * (baseline - rmse))/10.0) + "e-2"
    if sk == 'xy':
        plot_param_vector(mat, title, fname, 0, .3, ['x axis','y axis'])
    if sk == 'css':
        plot_param_vector(mat, title, fname, 0, .1, ['center','shape','spread'])
    if sk == 'descrip':
        plot_param_vector(mat, title, fname, 0, .1, ['h to d','d to h'])
    if sk == 'whole_tutor':
        plot_param_vector(mat, title, fname, 0, .4, ['center','shape','spread','x axis','y axis','h to d','d to h','histogram'])


# KT+D params
for sk in ['xy', 'descrip', 'css', 'whole_tutor']:
    try:
        rmse = np.mean(json.load(open("apr3_exps/RMSE_usefuldiff_" + sk + "_2states_500iter.json","r")))
        #baseline = np.mean(json.load(open("apr3_exps/RMSE_" + sk + "_bkt_2states_2000iter.json", "r")))
        params = json.load(open("apr3_exps/PARAMS_usefuldiff_" + sk + "_2states_500iter.json","r"))
        params = params[4:]

        print sk, rmse#, baseline
    except Exception as e:
        print e
        continue

    transition_params = {}

    for k in params[0].iterkeys():
        if "U" in k:
            transition_params[k] = []
    for p in params:
        for k,v in p.iteritems():
            if "U" in k:
                transition_params[k].append(v)

    if sk == 'xy' or sk == 'descrip':
        dim = 2
    if sk == 'css':
        dim = 3
    if sk == 'whole_tutor':
        dim = 8

    mat = np.zeros((dim,dim))

    vec = []
    for k,v in transition_params.iteritems():
        #print k
        val = np.mean(v)
        x = int(k[2])
        y = int(k[4])

        if x == 4 and y == 3:
            print val

        if x == 2 and y == 6:
            print val

        mat[dim-1-x,y] = val
    print mat
    print

    fname = "trans_plots/KTD/" + sk + ".png"
    title = "Knowledge transfer + Diff parameters\n" #+ "_" + set + ".\nRMSE improvement: " + str(int(1000 * (baseline - rmse))/10.0) + "e-2"
    if sk == 'xy':
        plot_param_vector(mat, title, fname, 0, .3, ['x axis','y axis'])
    if sk == 'css':
        plot_param_vector(mat, title, fname, 0, .1, ['center','shape','spread'])
    if sk == 'descrip':
        plot_param_vector(mat, title, fname, 0, .1, ['h to d','d to h'])
    if sk == 'whole_tutor':
        plot_param_vector(mat, title, fname, 0, .4, ['center','shape','spread','x axis','y axis','h to d','d to h','histogram'])