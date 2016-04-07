""" Run an MCMC fit on the MLFKT model (maybe in BKT mode though)
    Does all the train/test splitting and calculates RMSE over post test
"""

from Metropolis.mcmc_sampler import MCMCSampler
from Metropolis.mlfkt_usefulness_model import MLFKTSUModel
import sys, json, time, random, os, math
import numpy as np

print "usage: python fit_MLFKT_multi_bkt_model.py burnin iterations k(for 1/k test split) bkt(y/n) skills num_intermediate_states iterations"

iterz = int(sys.argv[7])

for it in range(iterz):
    skill = sys.argv[5]
    intermediate_states = int(sys.argv[6])
    total_states = intermediate_states + 2
    fname = skill.replace(" ","_")
    fname = fname.replace("\"","")
    """
    try:
        os.system("python bkt_data_split.py " + fname + ".csv \"" + skill + "\"")
    except Exception as e:
        print str(e)
        pass
    """

    #time.sleep(1)

    #load observations
    X = np.loadtxt(open("dump/observations_" + fname + ".csv","rb"),delimiter=",")
    #load problem IDs for these observations
    P = np.loadtxt(open("dump/problems_" + fname + ".csv","rb"),delimiter=",")
    S = np.loadtxt(open("dump/skills_" + fname + ".csv", "rb"), delimiter=",")

    start = time.time()

    k = int(sys.argv[3])
    #split 1/kth into test set
    N = X.shape[0]
    Xtest = []
    Ptest = []
    Stest = []
    Xnew = []
    Pnew = []
    Snew = []
    for c in range(N):
        if c % k == 0:#random.random() < 1 / (k+0.0):
            Xtest.append(X[c,:])
            Ptest.append(P[c,:])
            Stest.append(S[c,:])
        else:
            Xnew.append(X[c,:])
            Pnew.append(P[c,:])
            Snew.append(S[c,:])
    X = Xnew
    P = Pnew
    S = Snew

    Xtest = np.array(Xtest)
    Ptest = np.array(Ptest)
    Stest = np.array(Stest)
    X = np.array(X)
    P = np.array(P)
    S = np.array(S)

    print str(Xtest.shape[0]) + " test sequences"
    print str(X.shape[0]) + " training sequences"

    if 'y' in sys.argv[4]:
        model = MLFKTSUModel(X, P, S, intermediate_states, 0, True)
    else:
        model = MLFKTSUModel(X, P, S, intermediate_states, 0.1, True)

    mcmc = MCMCSampler(model, 0.15)

    burn = int(sys.argv[1])
    for c in range(20):
        mcmc.burnin(int(math.ceil((burn+0.0) / 20)))
        print("finished burn-in #: " + str((c+1)*burn/20))

    num_iterations = int(sys.argv[2])
    loop = 20
    per_loop = int(math.ceil((num_iterations+0.0) / loop))
    for c in range(loop):
        a = time.time()
        mcmc.MH(per_loop)
        b = time.time()
        print("finished iteration: " + str((c+1)*per_loop) + " in " + str(int(b-a)) + " seconds")

    end = time.time()

    print("Finished burnin and " + str(num_iterations) + " iterations in " + str(int(end-start)) + " seconds.")

    folder = "plots_" + fname
    #plotting samples will also load the MAP estimates
    #mcmc.plot_samples(folder + "/", str(num_iterations) + '_iterations')

    #load up test data and run predictions
    model.load_test_split(Xtest, Ptest, Stest)
    pred = model.get_predictions()
    num = model.get_num_predictions()
    mast = model.get_mastery()

    err = pred - Xtest
    rmse = np.sqrt(np.sum(err**2)/num)

    errl = np.zeros(num)
    predl = np.zeros(num)
    mastl = np.zeros(num)
    xtestl = np.zeros(num)
    i = 0
    for n in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            if pred[n][t] == -1:
                break
            errl[i] = err[n][t]
            predl[i] = pred[n][t]
            mastl[i] = mast[n][t]
            xtestl[i] = Xtest[n][t]
            i += 1

    from matplotlib import pyplot as plt
    """plt.hist(np.array([predl]).T, 30)
    plt.savefig(folder + "/Predictions_" + str(num_iterations) + '_iterations')
    plt.clf()
    plt.hist(np.array([errl]).T, 30)
    plt.savefig(folder + "/Errors_" + str(num_iterations) + '_iterations')
    plt.clf()"""

    print "RMSE:\t" + str(rmse)

    """f = open(folder + "/RMSE" + str(num_iterations) + '_iterations', "w+")
    f.write("RMSE: " + str(rmse) + "\n\n\nErrors: (prediction - observation)\n\n")
    for c in range(err.shape[0]):
        f.write(str(err[c,:]) + '\n')
    f.close()

    f = open(folder + "/mastery" + str(num_iterations) + '_iterations', "w+")
    for c in range(num):
        f.write(str(mastl[c]) + ',' + str(predl[c]) + ', ' + str(xtestl[c]) + '\n')
    f.close()"""

    #mcmc.save_model(folder + "/" + str(num_iterations) + '_iterations.model')

    if 'y' in sys.argv[4]:
        fname += '_bkt'

    rmsefname = 'dump/RMSE_multibkt_' + fname + "_" + str(total_states) + "states_" + str(num_iterations) +"iter" + '.json'
    if os.path.exists(rmsefname):
        rmsel = json.load(open(rmsefname,"r"))
    else:
        rmsel = []

    rmsel.append(rmse)
    json.dump(rmsel, open(rmsefname,"w"))

    paramfname = 'dump/PARAMS_multibkt_' + fname + "_" + str(total_states) + "states_" + str(num_iterations) +"iter" + '.json'
    if os.path.exists(paramfname):
        try:
            paraml = json.load(open(paramfname,"r"))
        except:
            paraml = []
    else:
        paraml = []
    p = model.get_parameters()
    pdict = {}
    for id, param in p.iteritems():
        #pdict[id] = param.get()
        if "D_" in id:
            pdict[id] = param.get()
        if "U" in id:
            pdict[id] = param.get()
    #pdict['Trans'] = list( [list(x) for x in model.make_transitions()] )
    #pdict['Pi'] = list(model.make_initial())
    #model.emission_mask[0] = False
    #pdict['Emit'] = list( [list(x) for x in model.make_emissions(0,0)] )

    paraml.append(pdict)
    json.dump(paraml, open(paramfname,"w"))


