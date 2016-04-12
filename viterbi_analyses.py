from Metropolis.mlfkt_usefulness_penalty_model import MLFKTSUPModel
from Metropolis.mlfkt_transition_model import MLFKTTransitionModel
from Metropolis.mlfkt_model import MLFKTModel
from Metropolis.kt_ideal import KTIDEAL
import sys, json, time, random, os, math
import numpy as np


def get_mastery_time(states):
    times = []
    for seq in states:
        found = False

        for t in range(len(seq)):
            if seq[t] == 1:
                found = True
                times.append(t)
                break
            #hax
            if seq[t] == -1:
                found = True
                times.append(t)
                break

        if not found:
            times.append(len(seq))

    if len(times) != len(states):
        print "ASDFASDGASDG"

    return times


for students in [10]:#,25,50,75,100,500]:


    # Do the transition model stuff first
    X = np.loadtxt(open("dump/observations_simulated_trans_"+str(students)+".csv","rb"),delimiter=",")
    #load problem IDs for these observations
    P = np.loadtxt(open("dump/problems_simulated_trans_"+str(students)+".csv","rb"),delimiter=",")
    #S = np.loadtxt(open("dump/skills_simulated_trans.csv", "rb"), delimiter=",")
    states = np.loadtxt(open("dump/states_simulated_trans_"+str(students)+".csv","rb"),delimiter=",")
    times = get_mastery_time(states)


    #get params learned from model
    pdictl = json.load(open("dump/PARAMS_simulated_trans_"+str(students)+"_L1_second_trans_2states_1000iter.json","r"))

    # L1 transition model, transition AFTER emission
    model = MLFKTTransitionModel(X, P, 0, 0.15, True, False)

    params = model.get_parameters()
    #Set the learned model parameters for transition model
    for k, v in params.iteritems():
        if k in pdictl[-1]:
            v.set(pdictl[-1][k])
        else:
            print
            print pdictl[-1]
            print "Uh oh,", k, "not in learned params for transition model"

    """params['D_8'].set(0.8834557934379552)
    params['D_9'].set(1.140053275980339)
    params['D_2'].set(-2.500678825112307)
    params['D_3'].set(-1.7398224556239437)
    params['D_0'].set(-2.2268556721015127)
    params['D_1'].set(-2.411186976135999)
    params['D_6'].set(-2.3003426304256855)
    params['D_7'].set(-1.4661162809707566)
    params['D_4'].set(-1.8692672253090565)
    params['D_5'].set(-1.583703195185952)

    params['G_0'].set(0.10618535357378697)
    params['S'].set(0.10018501374500711)
    params['L'].set(np.array([0.92012063497620489, 0.079879365023795085]))
    params['T'].set(0)"""

    #run viterbi
    state_estimates, masteries = model.viterbi()
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    time_estimates = get_mastery_time(state_estimates)

    trans_errs = []
    for c in range(len(times)):
        trans_errs.append(time_estimates[c] - times[c])

    model.load_test_split(X,P,False)
    trans_pred = model.get_predictions()
    trans_rmse = np.sqrt(np.mean( (trans_pred - X) ** 2))
    #print trans_errs



    # Setup KT-IDEAL model
    pdictl = json.load(open("dump/PARAMS_simulated_trans_"+str(students)+"_second_ktideal_2states_1000iter.json","r"))
    model = KTIDEAL(X,P,0,0.15,False,False)
    params = model.get_parameters()
    #Set the learned model parameters for transition model
    for k, v in params.iteritems():
        if k in pdictl[-1]:
            v.set(pdictl[-1][k])
        else:
            print
            print pdictl[-1]
            print "Uh oh,", k, "not in learned params for KTIDEAL model"
    state_estimates, masteries = model.viterbi()
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    time_estimates = get_mastery_time(state_estimates)

    ideal_errs = []
    for c in range(len(times)):
        ideal_errs.append(time_estimates[c] - times[c])

    model.load_test_split(X,P,False)
    ideal_pred = model.get_predictions()
    ideal_rmse = np.sqrt(np.mean( (ideal_pred - X) ** 2))




    # Now setup the regular BKT model
    pdictl = json.load(open("dump/PARAMS_simulated_trans_"+str(students)+"_bkt_L1_2states_1000iter.json","r"))
    model = MLFKTModel(X,P,0,0,True)
    #setup learned params
    params = model.get_parameters()
    #Set the learned model parameters for transition model
    params['G_0'].set(pdictl[-1]['G_0'])
    params['S'].set(pdictl[-1]['S'])

    params['L'].set(np.array(pdictl[-1]['L']))
    t = pdictl[-1]['T']
    tmat = np.array( [[1-t, t], [0.0, 1]] )
    params['T'].set(tmat)

    #run viterbi
    state_estimates, masteries, crap = model.viterbi()
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    #print state_estimates
    time_estimates = get_mastery_time(state_estimates)

    bkt_errs = []
    for c in range(len(times)):
        bkt_errs.append(time_estimates[c] - times[c])
    #print bkt_errs
    model.load_test_split(X,P,False)
    bkt_pred = model.get_predictions()
    bkt_rmse = np.sqrt(np.mean( (bkt_pred - X) ** 2))




    # Now setup the LFKT diff model
    pdictl = json.load(open("dump/PARAMS_simulated_trans_"+str(students)+"_L1_2states_1000iter.json","r"))
    model = MLFKTModel(X,P,0,0.15,True)
    #setup learned params
    params = model.get_parameters()
    #Set the learned model parameters for transition model
    params['G_0'].set(pdictl[-1]['G_0'])
    params['S'].set(pdictl[-1]['S'])
    params['L'].set(np.array(pdictl[-1]['L']))
    t = pdictl[-1]['T']
    tmat = np.array( [[1-t, t], [0.0, 1]] )
    params['T'].set(tmat)

    for k, v in params.iteritems():
        if k in pdictl[-1] and "D_" in k:
            v.set(pdictl[-1][k])

    #run viterbi
    state_estimates, masteries, crap = model.viterbi()
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    #print state_estimates
    time_estimates = get_mastery_time(state_estimates)

    lfkt_errs = []
    for c in range(len(times)):
        lfkt_errs.append(time_estimates[c] - times[c])
    #print bkt_errs
    model.load_test_split(X,P,False)
    lfkt_pred = model.get_predictions()
    lfkt_rmse = np.sqrt(np.mean( (lfkt_pred - X) ** 2))







    #mean_trans_err = np.mean(trans_errs)
    #mean_bkt_err = np.mean(bkt_errs)
    mean_abs_trans_err = np.mean(np.abs(trans_errs))
    mean_abs_bkt_err = np.mean(np.abs(bkt_errs))
    mean_abs_lfkt_err = np.mean(np.abs(lfkt_errs))
    mean_abs_ideal_err = np.mean(np.abs(ideal_errs))
    #rmse_trans = np.sqrt(np.mean(np.array(trans_errs)**2))
    #rmse_bkt = np.sqrt(np.mean(np.array(bkt_errs)**2))


    #print "ME transition model:", mean_trans_err
    #print "ME bkt model:       ", mean_bkt_err
    print students, "students"
    print "MAE transition model:", mean_abs_trans_err
    print "MAE KT-IDEAL model:  ", mean_abs_ideal_err
    print "MAE bkt model:       ", mean_abs_bkt_err
    print "MAE lfkt model:      ", mean_abs_lfkt_err
    print
    print "RMSE transition model:", trans_rmse
    print "RMSE KT-IDEAL model:  ", ideal_rmse
    print "RMSE bkt model:       ", bkt_rmse
    print "RMSE lfkt model:      ", lfkt_rmse
    #print "RMSE transition model:", rmse_trans
    #print "RMSE bkt model:       ", rmse_bkt





















    #### Now do all of that again for the KT model
    X = np.loadtxt(open("dump/observations_simulated_KT2_"+str(students)+".csv","rb"),delimiter=",")
    P = np.loadtxt(open("dump/problems_simulated_KT2_"+str(students)+".csv","rb"),delimiter=",")
    S = np.loadtxt(open("dump/skills_simulated_KT2_"+str(students)+".csv", "rb"), delimiter=",")
    states = np.loadtxt(open("dump/states_simulated_sep0_KT2_"+str(students)+".csv","rb"),delimiter=",")
    times = get_mastery_time(states)
    print times

    # Knowledge transfer model (w/ penalty for unlearned related skills)
    model = MLFKTSUPModel(X,P,S,0,0.15)

    pdictl = json.load(open("dump/PARAMS_pen_useful_simulated_KT2_"+str(students)+"_2states_1000iter.json","r"))
    params = model.get_parameters()
    #Set the learned model parameters for transition model
    for k, v in params.iteritems():
        if k in pdictl[-1]:
            v.set(pdictl[-1][k])
        else:
            print "Uh oh,", k, "not in learned params for transition model"

    """params = model.get_parameters()
    #Set the learned model parameters for KT model
    params['U-0-1'].set(1.0908236621413971)
    params['U-1-0'].set(1.1532099695095337)

    params['G-0-_0'].set(-1.73)
    params['S-0-'].set(-1.73)
    params['L-0-'].set(np.array([0.99, 0.01]))
    params['T-0-'].set(np.array([[0.9, 0.1],[0,1]]))

    params['G-1-_0'].set(-1.73)
    params['S-1-'].set(-1.73)
    params['L-1-'].set(np.array([0.99, 0.01]))
    params['T-1-'].set(np.array([[0.9, 0.1],[0,1]]))"""


    #run viterbi
    state_estimates, masteries, time_estimates2 = model.viterbi(0)
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    time_estimates = get_mastery_time(state_estimates)
    #jk use others
    time_estimates = time_estimates2

    kt_errs = []
    for c in range(len(times)):
        kt_errs.append(time_estimates[c] - times[c])

    #print state_estimates

    #quick test predictive stuff
    model.load_test_split(X,P,S, False)
    preds = model.get_predictions()
    errs = preds - X
    print "K trans RMSE:", np.sqrt(np.mean(errs ** 2))



    # Now setup the regular BKT model

    X = np.loadtxt(open("dump/observations_simulated_sep0_KT2_"+str(students)+".csv","rb"),delimiter=",")
    P = np.loadtxt(open("dump/problems_simulated_sep0_KT2_"+str(students)+".csv","rb"),delimiter=",")

    model = MLFKTModel(X,P,0,0,True)
    #setup learned params
    params = model.get_parameters()
    #Set the learned model parameters for transition model

    #sigmoid offsets
    params['G_0'].set(-1.4)
    params['S'].set(-1.82)

    params['L'].set(np.array([0.98902274897949349, 0.010977251020506663]))
    tmat = np.array( [[0.8824757084326732, 0.11752429156732683], [0.0, 1]] )
    params['T'].set(tmat)


    #run viterbi
    state_estimates, masteries, crap = model.viterbi()
    #first val needs to be pulled off (I think)
    state_estimates = [x[1:] for x in state_estimates]
    #print state_estimates
    time_estimates = get_mastery_time(state_estimates)


    mkt_errs = []
    for c in range(len(times)):
        mkt_errs.append(time_estimates[c] - times[c])
    #print bkt_errs


    mean_kt_err = np.mean(kt_errs)
    mean_mkt_err = np.mean(mkt_errs)
    mean_abs_kt_err = np.mean(np.abs(kt_errs))
    mean_abs_mkt_err = np.mean(np.abs(mkt_errs))
    rmse_kt = np.sqrt(np.mean(np.array(kt_errs)**2))
    rmse_mkt = np.sqrt(np.mean(np.array(mkt_errs)**2))


    print "ME knowledge transfer model:", mean_kt_err
    print "ME bkt model:       ", mean_mkt_err
    print "MAE knowledge transfer model:", mean_abs_kt_err
    print "MAE bkt model:       ", mean_abs_mkt_err
    print "RMSE knowledge transfer model:", rmse_kt
    print "RMSE bkt model:       ", rmse_mkt















