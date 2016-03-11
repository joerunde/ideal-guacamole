import json
import numpy as np

settings = ['css','center','shape','spread','xy','x_axis','y_axis','descrip','d_to_h','h_to_d','histogram','whole_tutor',
            'circleall','composition','parallel','pentagon','triangle','trap','all_geom']


f = open("comparison.tsv", "w")
f.write("Setting\tBKT-2\tBKT-3\tBKT-4\tC BKT-2\tC BKT-3\tC BKT-4\tLFKT-2\tLFKT-3\tLFKT-4\tC LFKT-2\tC LFKT-3\tC LFKT-4\n")

for setting in settings:
    f.write(setting + "\t")
    for c in range(2,5):
        try:
            val = np.mean(json.load(open("RMSE_" + setting + "_bkt_" + str(c) + "states_500iter.json","r")))
            val = int(1000 * val)/1000.0
            f.write(str(val)+"\t")
        except:
            f.write("--\t")
    for c in range(2,5):
        try:
            val = np.mean(json.load(open("RMSE_constrained_" + setting + "_bkt_" + str(c) + "states_500iter.json","r")))
            val = int(1000 * val)/1000.0
            f.write(str(val)+"\t")
        except:
            f.write("--\t")
    for c in range(2,5):
        try:
            val = np.mean(json.load(open("RMSE_" + setting + "_" + str(c) + "states_500iter.json","r")))
            val = int(1000 * val)/1000.0
            f.write(str(val)+"\t")
        except:
            f.write("--\t")
    for c in range(2,5):
        try:
            val = np.mean(json.load(open("RMSE_constrained_" + setting + "_" + str(c) + "states_500iter.json","r")))
            val = int(1000 * val)/1000.0
            f.write(str(val)+"\t")
        except:
            f.write("--\t")
    f.write("\n")

f.close()