import json
import numpy as np
import matplotlib.pyplot as plt

#settings = ['css','center','shape','spread','xy','x_axis','y_axis','descrip','d_to_h','h_to_d','histogram','whole_tutor',
#            'circleall','composition','parallel','pentagon','triangle','trap','all_geom']

def plot_bars(bars, errs, labels, colors, total_bars, fname):
    fig, ax = plt.subplots()

    plt.bar(np.arange(len(bars)), bars, yerr=errs, color=colors, ecolor='black')

    ax.set_ylim([0.35, 0.55])
    ax.set_xlim([0,total_bars])
    ax.get_xaxis().set_ticks([])
    plt.savefig(fname)
    plt.close(fig)

#plot_bars([0.4, 0.5, 0.6], [0.1, 0.1, 0.1], ['A','B','C'], ['red', 'green', 'orangered'], 12)



f = open("comparison.tsv", "w")
#f.write("Setting\tBKT-2\tBKT-3\tBKT-4\tC BKT-2\tC BKT-3\tC BKT-4\tLFKT-2\tLFKT-3\tLFKT-4\tC LFKT-2\tC LFKT-3\tC LFKT-4\n")

skills = ['x_axis','y_axis','xy','center','shape','spread','css','h_to_d','d_to_h','descrip','histogram','whole_tutor']
settings = ['BKT','BKT-4','LFKT','LFKT-4','LFKT (L1)', 'LFKT-transition-first (L1)', 'LFKT-transition-second (L1)', 'LFKT-transition-first-difficulty (L1)', 'LFKT-transition-second-difficulty (L1)', 'LFKT-skills (L1)', 'LFKT-knowledge-transfer (L1)', 'LFKT-knowledge-transfer-difficulty (L1)']



colors = ['r','orangered','b','navy','royalblue','gold','yellow','darkgreen','limegreen','lightgray','darkorchid','mediumvioletred']
colortable = {}
for c in range(len(settings)):
    colortable[settings[c]] = colors[c]

table = {}
stdtable = {}
for sk in skills:
    table[sk] = {}
    stdtable[sk] = {}

thresh = 0.03

def set_stuff(sk, set, fname, table, stdtable):
    try:
        vals = (json.load(open(fname,"r")))
        if np.max(vals) - np.min(vals) > thresh:
            print sk, np.max(vals), np.min(vals)
            vals.remove(np.max(vals))
        val = np.mean(vals)
        table[sk][set] = val
        stdtable[sk][set] = 2*np.std(vals)
    except:
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

    #LFKT-knowledge-transfer (L1)
    set_stuff(sk, 'LFKT-knowledge-transfer (L1)', "apr3_exps/RMSE_useful_" + sk + "_2states_1000iter.json", table, stdtable)

    #LFKT-knowledge-transfer-difficulty (L1)
    set_stuff(sk, 'LFKT-knowledge-transfer-difficulty (L1)', "apr3_exps/RMSE_usefuldiff_" + sk + "_2states_1000iter.json", table, stdtable)


f.write("--\t" + "\t".join(settings) + "\n")

for sk in skills:
    line = [sk]

    bars = []
    errs = []
    colors = []

    for setting in settings:
        try:
            val = table[sk][setting]
            val = int(1000 * val)/1000.0
            line.append(str(val))

            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])

        except:
            line.append("--")
    f.write("\t".join(line) + "\n")

    plot_bars(bars, errs, [], colors, 12, "resultplots/" + sk + ".png")

f.close()


#plot baselines
for sk in skills:
    bars = []
    errs = []
    colors = []

    for setting in ['BKT','LFKT','LFKT (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/baselines/" + sk + ".png")

#plot multi-state
for sk in skills:
    bars = []
    errs = []
    colors = []

    for setting in ['BKT','BKT-4','LFKT','LFKT-4']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/multistate/" + sk + ".png")

#plot transition
for sk in skills:
    bars = []
    errs = []
    colors = []

    for setting in ['BKT', 'LFKT-transition-first (L1)','LFKT-transition-second (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/trans/" + sk + ".png")

#plot transition-diff
for sk in skills:
    bars = []
    errs = []
    colors = []

    for setting in ['LFKT (L1)', 'LFKT-transition-first-difficulty (L1)','LFKT-transition-second-difficulty (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/transdiff/" + sk + ".png")



#plot combined....?


#plot skill
for sk in ['xy', 'css', 'descrip', 'whole_tutor']:
    bars = []
    errs = []
    colors = []

    for setting in ['BKT', 'LFKT (L1)','LFKT-skills (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/skill/" + sk + ".png")


#plot KT
for sk in ['xy', 'css', 'descrip', 'whole_tutor']:
    bars = []
    errs = []
    colors = []

    for setting in ['BKT', 'LFKT (L1)','LFKT-knowledge-transfer (L1)', 'LFKT-knowledge-transfer-difficulty (L1)']:
        try:
            val = table[sk][setting]
            bars.append(val)
            errs.append(stdtable[sk][setting])
            colors.append(colortable[setting])
        except:
            continue
    plot_bars(bars, errs, [], colors, 5, "resultplots/KT/" + sk + ".png")





