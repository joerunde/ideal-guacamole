""" Run an MCMC fit on the MLFKT model (maybe in BKT mode though)
    Does all the train/test splitting and calculates RMSE over post test
"""

from Metropolis.mcmc_sampler import MCMCSampler
from Metropolis.mlfkt_model import MLFKTModel
import sys, json, time, random, os, math
import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def plot_scatter(x, y, title, xaxis, yaxis, fname, noise=False, fit=False, fitLR=False):
    fig = plt.figure()

    allx = []
    for xl in x:
        allx.extend(xl)
    ally = []
    for yl in y:
        ally.extend(yl)

    if fit:
        print "boop"
        #linear...

        m,b = np.polyfit(allx, ally, 1)
        #print m, b
        xp = np.arange(int(np.min(allx)), int(np.max(allx)) + 2)
        plt.plot(xp, m*xp + b, 'g-')

    if fitLR:
        #logreg...
        for c in range(len(ally)):
            if ally[c] > 1:
                ally[c] = 1
        model = LogisticRegression()
        model.fit(np.array([allx]).T, np.array([ally]).T)
        w0 = model.intercept_
        w = model.coef_
        xp = np.arange(int(np.max(allx)) + 1)
        plt.plot(xp, expit(xp * w + w0)[0], 'r-')

    if noise:
        x = x + (np.random.random(len(allx)) - 0.5)/10.0
        y = y + (np.random.random(len(ally)) - 0.5)/10.0

    plt.title(title)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'gray', 'purple', 'black']
    for c in range(len(x)):
        plt.plot(x[c], y[c], 'o', color=colors[c])
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1-.5,y2+.5))
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

def get_skill_for_whole_tutor_prob(prob):
    probs = np.loadtxt(open("dump/problems_whole_tutor.csv","rb"),delimiter=",")
    skills = np.loadtxt(open("dump/skills_whole_tutor.csv","rb"),delimiter=",")
    for c in range(probs.shape[0]):
        for i in range(len(probs[c,:])):
            if int(probs[c,i]) == prob:
                return int(skills[c,i])

def run_learned_model(skill, diff_params = None):
    intermediate_states = 0
    fname = skill.replace(" ","_")
    fname = fname.replace("\"","")

    X = np.loadtxt(open("dump/observations_" + fname + ".csv", "rb"), delimiter=",")
    P = np.loadtxt(open("dump/problems_" + fname + ".csv","rb"),delimiter=",")

    k = 5
    #split 1/kth into test set
    N = X.shape[0]
    Xtest = []
    Xnew = []
    Ptest = []
    Pnew = []
    for c in range(N):
        if c % k == 0:#random.random() < 1 / (k+0.0):
            Xtest.append(X[c,:])
            Ptest.append(P[c,:])
        else:
            Xnew.append(X[c,:])
            Pnew.append(P[c,:])
    X = Xnew
    Xtest = np.array(Xtest)
    P = Pnew
    Ptest = np.array(Ptest)

    model = MLFKTModel(X, P, 0, 0.1)

    predl = []
    errl = []

    for c in range(3):
        param_dict = json.load(open("feb20_exps/PARAMS_"+skill+"_2states_500iter.json","r"))
        param_dict = param_dict[c]
        params = model.get_parameters()
        for k, v in param_dict.iteritems():
            #print k, v
            if k == "Pi":
                val = np.array(v)
                params["L"].set(val)
                params["L"].save()
            elif k == "Trans":
                val = np.array(v)
                params["T"].set(val)
                params["T"].save()
            elif k == "Emit":
                G = scipy.special.logit(v[0][1])
                S = scipy.special.logit(v[1][0])
                params["G_0"].set(G)
                params["S"].set(S)
                params["G_0"].save()
                params["S"].save()
            else:
                if diff_params is None:
                    params[k].set(v)
                    params[k].save()
                else:
                    params[k].set(diff_params[k])
                    params[k].save()

        params['Dsigma'].save()
        model.load_test_split(Xtest, Ptest)
        preds = model.get_predictions()
        err = preds - Xtest
        predl.append(preds)
        errl.append(err)



    return Xtest, Ptest, np.mean(predl,0), np.mean(errl,0)


Xtest, Ptest, preds, err = run_learned_model("whole_tutor")
#grab skills real quick
S = np.loadtxt(open("dump/skills_whole_tutor.csv", "rb"), delimiter=",")
Stest = []
Snew = []
for c in range(S.shape[0]):
    if c % 5 == 0:#random.random() < 1 / (k+0.0):
        Stest.append(S[c,:])
    else:
        Snew.append(S[c,:])
S = Snew
Stest = np.array(Stest)


num = 0
for c in range(len(preds)):
    for i in range(len(preds[c,:])):
        #print str(Stest[c,i]) + "\t" + str(err[c][i]) + "\t" + str(Xtest[c,i])
        if Xtest[c][i] > -1:
            num += 1

rmse = np.sqrt(np.sum(err**2)/num)
print rmse

skill_errs = []
for c in range(8):
    skill_errs.append([])

for c in range(len(preds)):
    for i in range(len(preds[c,:])):
        skill = int(Stest[c,i])
        if skill == -1:
            break
        skill_errs[skill].append(err[c,i])

skill_rmse = []
for c in range(8):
    #print len(skill_errs[c])
    skill_rmse.append( np.sqrt(np.sum( np.array(skill_errs[c])**2 )/len(skill_errs[c])) )

#print skill_errs
#print skill_rmse
"""
print "Center RMSE:     " + str(skill_rmse[0])
print "X axis RMSE:     " + str(skill_rmse[3])
print "Y axis RMSE:     " + str(skill_rmse[4])
print "Shape RMSE:      " + str(skill_rmse[1])
print "Histogram RMSE:  " + str(skill_rmse[7])
print "Spread RMSE:     " + str(skill_rmse[2])
print "H to D RMSE:     " + str(skill_rmse[5])
print "D to H RMSE:     " + str(skill_rmse[6])
"""

skill_combined_rmse = {}
skill_combined_rmse["center"] = skill_rmse[0]
skill_combined_rmse["x_axis"] = skill_rmse[3]
skill_combined_rmse["y_axis"] = skill_rmse[4]
skill_combined_rmse["shape"] = skill_rmse[1]
skill_combined_rmse["histogram"] = skill_rmse[7]
skill_combined_rmse["spread"] = skill_rmse[2]
skill_combined_rmse["h_to_d"] = skill_rmse[5]
skill_combined_rmse["d_to_h"] = skill_rmse[6]

skill_errs_dict = {}
skill_errs_dict["center"] = skill_errs[0]
skill_errs_dict["x_axis"] = skill_errs[3]
skill_errs_dict["y_axis"] = skill_errs[4]
skill_errs_dict["shape"] = skill_errs[1]
skill_errs_dict["histogram"] = skill_errs[7]
skill_errs_dict["spread"] = skill_errs[2]
skill_errs_dict["h_to_d"] = skill_errs[5]
skill_errs_dict["d_to_h"] = skill_errs[6]


"""
Okay, so we see that RMSE is comparable or **lower** with combined model than separated
Significance tests...?
"""

skill_separate_rmse = {}
skill_separate_errs = {}
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    Xtest, Ptest, preds, err = run_learned_model(skill)
    errlist = []
    for c in range(len(err[:,0])):
        for i in range(len(err[0,:])):
            if Xtest[c,i] > -1:
                errlist.append(err[c,i])

    skill_separate_rmse[skill] = np.sqrt(np.sum( np.array(errlist)**2 )/len(errlist))
    skill_separate_errs[skill] = errlist
    """
    print
    print len(errlist)
    print errlist
    print len(skill_errs_dict[skill])
    print skill_errs_dict[skill]
    print
    print len(Xtest)
    print
    """

for k in skill_combined_rmse.keys():
    print "Skill: " + k
    print "Combined RMSE: " + str(skill_combined_rmse[k])
    print "Separate RMSE: " + str(skill_separate_rmse[k])
    print scipy.stats.ranksums(skill_errs_dict[k], skill_separate_errs[k])
print rmse


"""
1 parameter per problem is required. Can we estimate that offline?
Correlation between difficulty offsets learned by model and % correct on problem?
"""

param_dict = json.load(open("feb20_exps/PARAMS_whole_tutor_2states_500iter.json","r"))
param_dict = param_dict[0]

model_difficulties = {}
for k,v in param_dict.iteritems():
    if "D_" in k:
        prob = int(k[2:])
        model_difficulties[prob] = v

X = np.loadtxt(open("dump/observations_whole_tutor.csv", "rb"), delimiter=",")
P = np.loadtxt(open("dump/problems_whole_tutor.csv","rb"),delimiter=",")

prob_accuracy = {}
for k in model_difficulties.keys():
    prob_accuracy[k] = []

for c in range(len(X)):
    for i in range(len(X[0])):
        if X[c,i] == -1:
            break
        prob = int(P[c,i])
        prob_accuracy[prob].append(X[c,i])

x1 = [ [], [], [], [], [], [], [], [] ]
x2 = [ [], [], [], [], [], [], [], [] ]
y1 = [ [], [], [], [], [], [], [], [] ]
y2 = [ [], [], [], [], [], [], [], [] ]
for k, v in prob_accuracy.iteritems():
    prob_accuracy[k] = np.mean(v)

    skill = get_skill_for_whole_tutor_prob(k)

    x1[skill].append(model_difficulties[k])
    y1[skill].append(max(-3,min(3,scipy.special.logit(1 - prob_accuracy[k]))))

    x2[skill].append(1 - scipy.special.expit(model_difficulties[k]))
    y2[skill].append(prob_accuracy[k])

    """
    print "Problem " + str(k)
    print "Model Difficulty:    " + str(model_difficulties[k])
    print "1-sig(model diff):   " + str(1 - scipy.special.expit(model_difficulties[k]))
    print "Accuracy:            " + str(prob_accuracy[k])
    print "logit(1-accuracy):   " + str(  max(-3,min(3,scipy.special.logit(1 - prob_accuracy[k])))     )
    """
plot_scatter(x1, y1, "model difficulty compared to estimate from accuracy", "model difficulty parameter", "logit(1-accuracy)", "model_diff_est.png", False, True)
plot_scatter(x2, y2, "accuracy estimate from model compared to accuracy", "1-sigmoid(model diff)", "accuracy", "model_accuracy_est.png", False, True)


# What happens if we load accuracy estimates into model?
diff_ests = {}
y = []
for yl in y1:
    y.extend(yl)
y1 = y
for c, y in enumerate(y1):
    diff_ests["D_" + str(c)] = y

Xtest, Ptest, preds, err = run_learned_model("whole_tutor", diff_ests)
rmse2 = np.sqrt(np.sum(err**2)/num)
print rmse2



"""
Sweet, so there is correlation and we can recover a good model using parameter estimates
How is the per-skill RMSE?
"""


Xtest, Ptest, preds, err = run_learned_model("whole_tutor", diff_ests)
skill_errs2 = []
for c in range(8):
    skill_errs2.append([])

for c in range(len(preds)):
    for i in range(len(preds[c,:])):
        skill = int(Stest[c,i])
        if skill == -1:
            break
        skill_errs2[skill].append(err[c,i])

skill_rmse2 = []
for c in range(8):
    #print len(skill_errs[c])
    skill_rmse2.append( np.sqrt(np.sum( np.array(skill_errs2[c])**2 )/len(skill_errs2[c])) )

skill_combined_rmse2 = {}
skill_combined_rmse2["center"] = skill_rmse2[0]
skill_combined_rmse2["x_axis"] = skill_rmse2[3]
skill_combined_rmse2["y_axis"] = skill_rmse2[4]
skill_combined_rmse2["shape"] = skill_rmse2[1]
skill_combined_rmse2["histogram"] = skill_rmse2[7]
skill_combined_rmse2["spread"] = skill_rmse2[2]
skill_combined_rmse2["h_to_d"] = skill_rmse2[5]
skill_combined_rmse2["d_to_h"] = skill_rmse2[6]

skill_errs_dict2 = {}
skill_errs_dict2["center"] = skill_errs2[0]
skill_errs_dict2["x_axis"] = skill_errs2[3]
skill_errs_dict2["y_axis"] = skill_errs2[4]
skill_errs_dict2["shape"] = skill_errs2[1]
skill_errs_dict2["histogram"] = skill_errs2[7]
skill_errs_dict2["spread"] = skill_errs2[2]
skill_errs_dict2["h_to_d"] = skill_errs2[5]
skill_errs_dict2["d_to_h"] = skill_errs2[6]

f = open("combined_rmse_tests.tsv","w")
f.write("Skill\tRMSE with models trained separately per skill\tRMSE with one model trained on all skills\tRMSE with one combined model using empirical estimates for item difficulty\n")

for k in skill_combined_rmse2.keys():
    print "Skill: " + k
    print "Combined (empirical difficulty estimates) RMSE:  " + str(skill_combined_rmse2[k])
    print "Separate RMSE:                                   " + str(skill_separate_rmse[k])
    print scipy.stats.ranksums(skill_errs_dict2[k], skill_separate_errs[k])
    f.write(k + "\t" + str(skill_separate_rmse[k]) + "\t" + str(skill_combined_rmse[k]) + "\t" + str(skill_combined_rmse2[k]) + "\n")
print rmse2
f.close()


