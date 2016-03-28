import json
import numpy as np

#settings = ['css','center','shape','spread','xy','x_axis','y_axis','descrip','d_to_h','h_to_d','histogram','whole_tutor',
#            'circleall','composition','parallel','pentagon','triangle','trap','all_geom']

f = open("comparison.tsv", "w")
#f.write("Setting\tBKT-2\tBKT-3\tBKT-4\tC BKT-2\tC BKT-3\tC BKT-4\tLFKT-2\tLFKT-3\tLFKT-4\tC LFKT-2\tC LFKT-3\tC LFKT-4\n")

skills = ['x_axis','y_axis','xy','center','shape','spread','css','h_to_d','d_to_h','descrip','histogram','whole_tutor']
settings = ['BKT','BKT-4','LFKT','LFKT-4','LFKT (L1)','LFKT-skills','LFKT-skills (L1)', 'LFKT-transition (L1)', 'LFKT-skill-interactions (L1)']

table = {}
for sk in skills:
    table[sk] = {}


for sk in skills:
    #BKT first
    try:
        val = np.mean(json.load(open("feb20_exps/RMSE_" + sk + "_bkt_2states_500iter.json","r")))
        table[sk]['BKT'] = val
    except:
        pass

    #BKT-4
    try:
        val = np.mean(json.load(open("feb20_exps/RMSE_" + sk + "_bkt_4states_500iter.json","r")))
        table[sk]['BKT-4'] = val
    except:
        pass

    #LFKT
    try:
        val = np.mean(json.load(open("feb20_exps/RMSE_" + sk + "_2states_500iter.json","r")))
        table[sk]['LFKT'] = val
    except:
        pass

    #LFKT-4
    try:
        val = np.mean(json.load(open("feb20_exps/RMSE_" + sk + "_4states_500iter.json","r")))
        table[sk]['LFKT-4'] = val
    except:
        pass

    #LFKT (L1)
    try:
        val = np.mean(json.load(open("mar_27_exps/RMSE_" + sk + "_2states_1000iter.json","r")))
        table[sk]['LFKT (L1)'] = val
    except:
        pass

    #LFKT-skills
    try:
        val = np.mean(json.load(open("march_ish_exps/RMSE_" + sk + "_2states_1000iter_skillmodel.json","r")))
        table[sk]['LFKT-skills'] = val
    except:
        pass

    #LFKT-skills (L1)
    try:
        val = np.mean(json.load(open("mar_27_exps/RMSE_" + sk + "_2states_1000iter_skillmodel.json","r")))
        table[sk]['LFKT-skills (L1)'] = val
    except:
        pass

    #LFKT-transition (L1)
    try:
        val = np.mean(json.load(open("mar_27_exps/RMSE_" + sk + "_trans_2states_1000iter.json","r")))
        table[sk]['LFKT-transition (L1)'] = val
    except:
        pass

    #LFKT-skill-interactions (L1)
    try:
        val = np.mean(json.load(open("mar_27_exps/RMSE_useful_" + sk + "_2states_1000iter.json","r")))
        table[sk]['LFKT-skill-interactions (L1)'] = val
    except:
        pass

#todo: fill in gaps by rerunning learned combo models and splitting skill errors (copypasta code?)

f.write("--\t" + "\t".join(settings) + "\n")

for sk in skills:
    line = [sk]
    for setting in settings:
        try:
            val = table[sk][setting]
            val = int(1000 * val)/1000.0
            line.append(str(val))
        except:
            line.append("--")
    f.write("\t".join(line) + "\n")

f.close()