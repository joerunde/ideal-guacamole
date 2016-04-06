import json
import numpy as np
import matplotlib.pyplot as plt

#settings = ['css','center','shape','spread','xy','x_axis','y_axis','descrip','d_to_h','h_to_d','histogram','whole_tutor',
#            'circleall','composition','parallel','pentagon','triangle','trap','all_geom']

def plot_bars(bars, errs, labels, colors, total_bars, fname):
    fig, ax = plt.subplots()

    plt.bar(np.arange(len(bars)), bars, yerr=errs, color=colors, ecolor='black')

    ax.set_ylim([0.35, 0.55])
    #ax.set_xlim([0,total_bars])
    ax.set_xticklabels(labels, rotation=60)
    ax.set_xticks(np.arange(len(bars))+.3)

    #ax.get_xaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

#plot_bars([0.4, 0.5, 0.6], [0.1, 0.1, 0.1], ['A','B','C'], ['red', 'green', 'orangered'], 12)

f = open("comparison.tsv", "w")
#f.write("Setting\tBKT-2\tBKT-3\tBKT-4\tC BKT-2\tC BKT-3\tC BKT-4\tLFKT-2\tLFKT-3\tLFKT-4\tC LFKT-2\tC LFKT-3\tC LFKT-4\n")

skills = ['x_axis','y_axis','xy','center','shape','spread','css','h_to_d','d_to_h','descrip','histogram','whole_tutor']
settings = ['BKT','BKT-4','LFKT','LFKT-4','LFKT (L1)', 'LFKT-transition-first (L1)', 'LFKT-transition-second (L1)', 'LFKT-transition-first-difficulty (L1)', 'LFKT-transition-second-difficulty (L1)', 'LFKT-skills (L1)', 'LFKT-knowledge-transfer (L1)', 'LFKT-knowledge-transfer-difficulty (L1)', 'LFKT-transition-first-difficulty-gauss (L1)', 'LFKT-transition-second-difficulty-gauss (L1)', 'LFKT-adaptive-transition (L1)', 'LFKT-adaptive-transition-difficulty (L1)']


labels = ['BKT','BKT-4','LFKT','LFKT-4','LFKT (L1)', 'Trans-Before', 'Trans-After', 'Trans-Before + Linear Diff', 'Trans-After + Linear Diff', 'Skill-difficulty', 'K-Transfer', 'K-Transfer + D', 'Trans-Before + Gauss Diff', 'Trans-After + Gauss Diff', 'Adaptive Transition (L1)', 'Adaptive Transition + D (L1)']

labeltable = {}

colors = ['r','orangered','b','royalblue','navy','darkorange','yellow','limegreen','darkgreen','lightgray','darkorchid','mediumvioletred', 'pink', 'crimson', 'green', 'purple']
colortable = {}
for c in range(len(settings)):
    colortable[settings[c]] = colors[c]
    labeltable[settings[c]] = labels[c]

table = {}
stdtable = {}
for sk in skills:
    table[sk] = {}
    stdtable[sk] = {}

thresh = 0.02

def get_avg(table, set):
    #average all skills weighted by number of observations in test set
    avgval = 83 * table['x_axis'][set] + 80 * table['y_axis'][set] + 83 * table['center'][set] + 168 * table['shape'][set] + 45 * table['spread'][set] + 76 * table['h_to_d'][set] + 54 * table['d_to_h'][set] + 192 * table['histogram'][set]
    avgval /= (83 + 80 + 83 + 168 + 45 + 76 + 54 + 192 + 0.0)
    return avgval

def set_stuff(sk, set, fname, table, stdtable):
    try:
        vals = (json.load(open(fname,"r")))

        if 'use' in fname:
            print sk
            print vals
            vals = vals[4:]
            print vals

        if np.max(vals) - np.min(vals) > thresh:
            print sk, np.max(vals), np.min(vals)
            vals.remove(np.max(vals))
        val = np.mean(vals)
        table[sk][set] = val
        stdtable[sk][set] = 2*np.std(vals)
    except Exception as e:
        print "Failed:",sk,set,e
        pass

for sk in skills:
    #BKT first
    set_stuff(sk, 'BKT', "apr3_exps/RMSE_" + sk + "_bkt_2states_2000iter.json", table, stdtable)

    #BKT-4
    set_stuff(sk, 'BKT-4', "apr3_exps/RMSE_" + sk + "_bkt_4states_2000iter.json", table, stdtable)

    #LFKT
    set_stuff(sk, 'LFKT', "apr3_exps/RMSE_" + sk + "_2states_2000iter.json", table, stdtable)

    #LFKT-4
    set_stuff(sk, 'LFKT-4', "apr3_exps/RMSE_" + sk + "_4states_2000iter.json", table, stdtable)

    #LFKT (L1)
    set_stuff(sk, 'LFKT (L1)', "apr3_exps/RMSE_" + sk + "_L1_2states_2000iter.json", table, stdtable)

    #LFKT-skills (L1)
    set_stuff(sk, 'LFKT-skills (L1)', "apr3_exps/RMSE_" + sk + "_L1_2states_2000iter_skillmodel.json", table, stdtable)

    #LFKT-transition-first (L1)
    set_stuff(sk, 'LFKT-transition-first (L1)', "apr3_exps/RMSE_" + sk + "_L1_first_trans_2states_2000iter.json", table, stdtable)
    
    #LFKT-transition-second (L1)
    set_stuff(sk, 'LFKT-transition-second (L1)', "apr3_exps/RMSE_" + sk + "_L1_second_trans_2states_2000iter.json", table, stdtable)

    #LFKT-transition-difficulty (L1)
    set_stuff(sk, 'LFKT-transition-first-difficulty (L1)', "apr3_exps/RMSE_" + sk + "_L1_first_transdiff_2states_2000iter.json", table, stdtable)

    #LFKT-transition-difficulty (L1)
    set_stuff(sk, 'LFKT-transition-second-difficulty (L1)', "apr3_exps/RMSE_" + sk + "_L1_second_transdiff_2states_2000iter.json", table, stdtable)

    #LFKT-transition-difficulty-gauss (L1)
    set_stuff(sk, 'LFKT-transition-first-difficulty-gauss (L1)', "apr3_exps/RMSE_" + sk + "_L1_first_transdiffgauss_2states_500iter.json", table, stdtable)

    #LFKT-transition-difficulty-gauss (L1)
    set_stuff(sk, 'LFKT-transition-second-difficulty-gauss (L1)', "apr3_exps/RMSE_" + sk + "_L1_second_transdiffgauss_2states_500iter.json", table, stdtable)

    #adaptive transitions, + diff
    set_stuff(sk, 'LFKT-adaptive-transition (L1)', "apr3_exps/RMSE_" + sk + "_L1_adapt_trans_2states_1000iter.json", table, stdtable)
    set_stuff(sk, 'LFKT-adaptive-transition-difficulty (L1)', "apr3_exps/RMSE_" + sk + "_L1_adapt_transdiff_2states_2000iter.json", table, stdtable)

    #LFKT-knowledge-transfer (L1)
    set_stuff(sk, 'LFKT-knowledge-transfer (L1)', "apr3_exps/RMSE_useful_" + sk + "_2states_500iter.json", table, stdtable)

    #LFKT-knowledge-transfer-difficulty (L1)
    set_stuff(sk, 'LFKT-knowledge-transfer-difficulty (L1)', "apr3_exps/RMSE_usefuldiff_" + sk + "_2states_500iter.json", table, stdtable)



table['average'] = {}
stdtable['average'] = {}
#calc avgs
for set in settings:
    try:
        table['average'][set] = get_avg(table, set)
        stdtable['average'][set] = get_avg(stdtable, set)
    except:
        print "No averages for: " + set
skills.append('average')
print table['average']




f.write("--\t" + "\t".join(settings) + "\n")
for sk in skills:
    line = [sk]

    bars = []
    errs = []
    colors = []
    labels = []

    for setting in settings:
        try:
            val = table[sk][setting]
            val = int(1000 * val)/1000.0
            line.append(str(val))

            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])

        except:
            line.append("--")
    f.write("\t".join(line) + "\n")

    plot_bars(bars, errs, labels, colors, len(bars), "resultplots/" + sk + ".png")
f.close()



#plot baselines
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT','LFKT','LFKT (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/baselines/" + sk + ".png")

#plot multi-state
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT','BKT-4','LFKT','LFKT-4']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/multistate/" + sk + ".png")

#plot transition
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-transition-first (L1)','LFKT-transition-second (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/trans/" + sk + ".png")

#plot adaptive transition
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-transition-first (L1)','LFKT-transition-second (L1)', 'LFKT-adaptive-transition (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, len(bars), "resultplots/transadapt/" + sk + ".png")

#plot adaptive trans and + difficulty
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-adaptive-transition (L1)', 'LFKT-adaptive-transition-difficulty (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, len(bars), "resultplots/transadaptdiff/" + sk + ".png")


#plot transition difficulty settings
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-adaptive-transition-difficulty (L1)', 'LFKT-transition-second-difficulty (L1)', 'LFKT-transition-second-difficulty-gauss (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, len(bars), "resultplots/transalldiff/" + sk + ".png")


#plot transition-diff
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-transition-first-difficulty (L1)','LFKT-transition-second-difficulty (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/transdiff/" + sk + ".png")

#plot transition-diff-gauss
for sk in skills:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)', 'LFKT-transition-first-difficulty (L1)', 'LFKT-transition-second-difficulty (L1)', 'LFKT-transition-first-difficulty-gauss (L1)', 'LFKT-transition-second-difficulty-gauss (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/transdiffgauss/" + sk + ".png")



#plot combined....?
bars = []
errs = []
colors = []
labels = []

for set in ['BKT', 'LFKT (L1)']:
    # get average over single skill models:
    allval = 83 * table['x_axis'][set] + 80 * table['y_axis'][set] + 83 * table['center'][set] + 168 * table['shape'][set] + 45 * table['spread'][set] + 76 * table['h_to_d'][set] + 54 * table['d_to_h'][set] + 192 * table['histogram'][set]
    allval /= (83 + 80 + 83 + 168 + 45 + 76 + 54 + 192 + 0.0)

    xyval = (83 * table['x_axis'][set] + 80 * table['y_axis'][set]) / (83.0 + 80)
    #cssval = (5 * table['center'][set] + 18 * table['shape'][set] + 3 * table['spread'][set]) / 26.0
    #descripval = (76 * table['h_to_d'][set] + 54 * table['d_to_h'][set]) / (76.0 + 54)

    whole = table['whole_tutor'][set]
    xy = table['xy'][set]
    #css = table['css'][set]
    #descrip = table['descrip'][set]

    bars.extend([xyval, xy, allval, whole])
    if 'BKT' in set:
        colors.extend(['lightgray','salmon','gray','red'])
    else:
        colors.extend(['lavender','lightsage','steelblue','green'])
    labels.extend([set + ' XY avg', 'combined', set + ' all skills avg', 'combined'])
plot_bars(bars, [0]*len(bars), labels, colors, 5, "resultplots/combined/" + 'all' + ".png")


#plot skill
for sk in ['xy', 'css', 'descrip', 'whole_tutor']:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)','LFKT-skills (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/skill/" + sk + ".png")


#plot KT
for sk in ['xy', 'css', 'descrip', 'whole_tutor']:
    bars = []
    errs = []
    colors = []
    labels = []

    for setting in ['BKT', 'LFKT (L1)','LFKT-knowledge-transfer (L1)', 'LFKT-knowledge-transfer-difficulty (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
            labels.append(labeltable[setting])
        except:
            continue
    plot_bars(bars, errs, labels, colors, 5, "resultplots/KT/" + sk + ".png")


print "done"


