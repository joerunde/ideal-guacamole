from Metropolis.mcmc_sampler import MCMCSampler
from Metropolis.mlfkt_model import MLFKTModel
import sys, json, time, random, os, math
import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint
from moe.optimal_learning.python import bo_edu
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential

# Returns the best-looking point in the given Gaussian Process.
def moe_compute_best_pt_info(moe_exp, covariance_info, confidence=None,
                             mean_fxn_info=None, sample_pts=None, debug=False):
    if moe_exp.historical_data.num_sampled <= 0: return None, None
    covar = SquareExponential(covariance_info['hyperparameters'])
    cpp_gp = GaussianProcess(covar, moe_exp.historical_data,
                             mean_fxn_info=mean_fxn_info)
    if (sample_pts == None):
        sample_pts = np.array(moe_exp.historical_data.points_sampled)
    #moe_pts_r = np.array(moe_exp.historical_data.points_sampled)
    #moe_pts_d = moe_exp.domain.generate_uniform_random_points_in_domain(50)
    #sample_pts = np.concatenate((moe_pts_r, moe_pts_d), axis=0)
    cpp_mu = cpp_gp.compute_mean_of_points(sample_pts, debug)
    cpp_var = np.diag(cpp_gp.compute_variance_of_points(sample_pts))
    #print "sample_pts ", sample_pts, "\ncpp_mu ", cpp_mu, "\ncpp_var ", cpp_var
    if confidence is None:
        minidx = np.argmin(cpp_mu)
    else:
        upper_conf = scipy.stats.norm.interval(confidence, loc=cpp_mu,
                                               scale=np.sqrt(cpp_var))[1]
        minidx = np.argmin(upper_conf)
        #print " cpp_var ", cpp_var[minidx], " upper_conf ", upper_conf[minidx]
    #print "cpp_mu ", cpp_mu[minidx], " best_moe_pt ", sample_pts[minidx]
    return [sample_pts[minidx], cpp_mu[minidx], cpp_var[minidx]]



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

    #predl = []
    #errl = []

    for c in range(1):
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
        #model.load_test_split(Xtest, Ptest)
        #preds = model.get_predictions()
        #err = preds - Xtest
        #predl.append(preds)
        #errl.append(err)

    #return Xtest, Ptest, np.mean(predl,0), np.mean(errl,0), model
    return model

def load_probs(skill):
    problems = json.load(open("dump/problems_idx_" + skill + ".csv"))
    return [c for c, v in enumerate(problems) if "assess" not in v]

def load_test_probs(skill):
    problems = json.load(open("dump/problems_idx_" + skill + ".csv"))
    return [c for c, v in enumerate(problems) if "assess" in v]

def get_skill_for_whole_tutor_prob(prob):
    probs = np.loadtxt(open("dump/problems_whole_tutor.csv","rb"),delimiter=",")
    skills = np.loadtxt(open("dump/skills_whole_tutor.csv","rb"),delimiter=",")
    for c in range(probs.shape[0]):
        for i in range(len(probs[c,:])):
            if int(probs[c,i]) == prob:
                l = ['center','shape','spread','x_axis','y_axis','histogram','h_to_d','d_to_h']
                return l[int(skills[c,i])]

def skill_prob_to_whole_tutor(prob, skill):
    probs = json.load(open("dump/problems_idx_" + skill + ".csv"))
    name = probs[prob]
    allprobs = json.load(open("dump/problems_idx_whole_tutor.csv"))
    return allprobs.index(name)

def whole_tutor_prob_to_skill(prob, skill):
    probs = json.load(open("dump/problems_idx_whole_tutor.csv"))
    name = probs[prob]
    skillprobs = json.load(open("dump/problems_idx_" + skill + ".csv"))
    return skillprobs.index(name)

class Simulator:

    def __init__(self, unified):
        if unified:
            self.unified = True
            self.model = run_learned_model('whole_tutor')
            self.model.start_student()
            self.problems = load_probs('whole_tutor')
            random.shuffle(self.problems)
            self.test_probs = load_test_probs('whole_tutor')
        else:
            self.unified = False
            self.models = {}
            self.problems = {}
            self.test_probs = {}
            for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
                self.models[skill] = run_learned_model(skill)
                self.models[skill].start_student()
                self.problems[skill] = load_probs(skill)
                random.shuffle(self.problems[skill])
                self.test_probs[skill] = load_test_probs(skill)

    def give_problem(self, skill=None):
        if skill is None:
            if self.unified:
                if len(self.problems) > 0:
                    prob = self.problems.pop()
                    return (self.model.give_problem(prob), prob)
                else:
                    return (-1, None)
            else:
                skills = ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']
                random.shuffle(skills)
                for skill in skills:
                    if len(self.problems[skill]) > 0:
                        prob = self.problems[skill].pop()
                        return (self.models[skill].give_problem(prob), skill_prob_to_whole_tutor(prob,skill))
                return (-1, None)
        else:
            if self.unified:
                for prob in self.problems:
                    if get_skill_for_whole_tutor_prob(prob) == skill:
                        self.problems.remove(prob)
                        return (self.model.give_problem(prob), whole_tutor_prob_to_skill(prob, skill))
                return (-1, None)
            else:
                if len(self.problems[skill]) > 0:
                    prob = self.problems.pop()
                    return (self.models[skill].give_problem(prob), prob)
                return (-1, None)

    def give_test(self, skill=None):
        obs = []
        if skill is None:
            if self.unified:
                for prob in self.test_probs:
                    obs.append( (self.model.give_problem(prob), prob) )
            else:
                for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
                    for prob in self.test_probs[skill]:
                        obs.append( (self.models[skill].give_problem(prob), skill_prob_to_whole_tutor(prob, skill)) )
        else:
            if self.unified:
                for prob in self.test_probs:
                    if get_skill_for_whole_tutor_prob(prob) == skill:
                        obs.append( (self.model.give_problem(prob), whole_tutor_prob_to_skill(prob, skill)) )
            else:
                for prob in self.test_probs[skill]:
                    obs.append( (self.models[skill].give_problem(prob), prob) )
        return obs


def objective(score, num_probs):
    return 1 - ( score * math.pow(num_probs+1.0, -1.0/16.0) )


class UnifiedTrial:
    def __init__(self, student, params_dict):
        self.student = student
        self.model = run_learned_model('whole_tutor')
        self.model.set_4_params(params_dict['pg'], params_dict['ps'], 1-params_dict['pi'], params_dict['pt'])
        self.threshold = params_dict['threshold']

    def get_pm(self):
        X = np.array([ [x[0] for x in self.traj] ]) + 0.0
        P = np.array([ [x[1] for x in self.traj] ]) + 0.0

        #print X
        #print P
        self.model.load_test_split(X, P, False)
        self.model._predict(True)
        #print self.model.get_mastery()
        return self.model.get_mastery()[0,-1]


    def run(self):
        learned = False

        ##########
        #if self.student.model.student_state > 0:
        #    print "Started learnt"
        #    learned = True

        self.traj = self.student.give_test()

        #########
        #if self.student.model.student_state > 0 and not learned:
        #    learned = True
        #    print "Learnt after pretest"

        problems_given = 0

        while self.get_pm() < self.threshold:
            obs = self.student.give_problem()
            if obs[0] < 0:
                break # no more problems
            problems_given += 1
            self.traj.append(obs)

            ##############
            #if self.student.model.student_state > 0 and not learned:
            #    learned = True
            #    print "Learnt after problem: " + str(problems_given)

        post_obs = self.student.give_test()
        score = sum( [x[0] for x in post_obs] ) / 13.0

        #print problems_given
        #print student.model.student_state
        return objective(score, problems_given)


class SeparateSkillTrial:
    def __init__(self, student, skill_params_dict):
        self.student = student
        self.models = {}
        self.skills_left = []
        self.thresholds = {}
        self.trajs = {}
        for skill, params in skill_params_dict.iteritems():
            self.skills_left.append(skill)
            self.models[skill] = run_learned_model('whole_tutor')
            self.models[skill].set_4_params(params['pg'], params['ps'], 1-params['pi'], params['pt'])
            self.thresholds[skill] = params['threshold']
            self.trajs[skill] = self.student.give_test(skill)

    def get_pm(self, skill):
        X = np.array([ [x[0] for x in self.trajs[skill]] ]) + 0.0
        P = np.array([ [x[1] for x in self.trajs[skill]] ]) + 0.0

        #print X
        #print P
        self.models[skill].load_test_split(X, P, False)
        self.models[skill]._predict(True)
        #print self.model.get_mastery()
        return self.models[skill].get_mastery()[0,-1]


    def run(self):

        problems_given = 0

        while len(self.skills_left) > 0:

            random.shuffle(self.skills_left)
            skill = self.skills_left[0]

            obs = self.student.give_problem()
            problems_given += 1
            if obs[0] < 0:
                self.skills_left.remove(skill) # no more problems for that skill
                continue

            self.trajs[skill].append(obs)
            if self.get_pm(skill) > self.thresholds[skill]:
                self.skills_left.remove(skill)

        post_obs = self.student.give_test()
        score = sum( [x[0] for x in post_obs] ) / 13.0

        return objective(score, problems_given)



#grab a quick baseline reading:
gg = []
for c in range(50):
    params = {'pg':-1, 'ps':-1, 'pi':0.00001, 'pt':0.00001, 'threshold':0.999999999999999999}
    student = Simulator(True)
    trial = UnifiedTrial(student, params)
    y = trial.run()
    #print y
    gg.append(y)
print "BASELINE MASTERED: " + str(np.mean(gg))

gg = []
for c in range(50):
    params = {'pg':-1, 'ps':-1, 'pi':0.99, 'pt':0.99, 'threshold':0.01}
    student = Simulator(True)
    trial = UnifiedTrial(student, params)
    y = trial.run()
    #print y
    gg.append(y)
print "BASELINE NOT MASTERED: " + str(np.mean(gg))


## now for the BO shiz
NOISE_VAL = 0.15
MOE_PRIOR_VARIANCE = NOISE_VAL
hyper_params = [MOE_PRIOR_VARIANCE]
MOE_LENGTH_SCALE = 2
#hyper_params.extend([MOE_LENGTH_SCALE]*5)
hyper_params.extend([MOE_LENGTH_SCALE]*3)
MOE_LENGTH_SCALE_THRESH = 0.2
hyper_params.append(MOE_LENGTH_SCALE_THRESH)
moe_covariance_info = {'covariance_type': 'square_exponential',
                              'hyperparameters': hyper_params}
moe_sq_exp = {'covariance_type': 'square_exponential'}

def UnifiedBOExp(iter):
    #bounds = [ [-3,.5], [-3,.5], [0,1], [0,1], [0,1] ]

    #reduce dof for BO. Let's set hard threshold 0.95 and hard p(i) as 0.05
    bounds = [ [-3,.5], [-3,.5], [0,1] ]
    exp = Experiment(bounds)
    objs = []

    for c in range(iter):
        #get list of next params
        x = gp_next_points(exp, num_to_sample=1, covariance_info=moe_sq_exp)[0]
        #put into dict
        params = {'pg':x[0], 'ps':x[1], 'pi':0.05, 'pt':x[2], 'threshold':0.95}

        #setup trial
        student = Simulator(True)
        trial = UnifiedTrial(student, params)

        y = trial.run()
        exp.historical_data.append_sample_points([SamplePoint(x, y, NOISE_VAL)])
        objs.append(y)

        print "x is: "
        print x
        print "objective: "
        print y
        print "predicted best point: "
        #print moe_compute_best_pt_info(exp, moe_covariance_info)[0]
        print ["%0.2f" % i for i in moe_compute_best_pt_info(exp, moe_covariance_info)[0]]
        print

    return objs

UnifiedBOExp(50)
#print UnifiedBOExp(50)








"""
def f(x):
    return (x-3.2765)**2 - 1.2443
exp = Experiment([[-20,20]])

for c in range(25):
    x = gp_next_points(exp,num_to_sample=1, covariance_info=moe_covariance_info)[0][0]
    print "x; " + str(x)
    y = f(x)
    exp.historical_data.append_sample_points([SamplePoint(x, y, NOISE_VAL)])

    print "bo_edu output: " + str(moe_compute_best_pt_info(exp, moe_covariance_info)[0])
"""

"""

uni_student = Simulator(True)
trial = UnifiedTrial(uni_student, {'pg':-.1, 'ps':-.1, 'pi':0.01, 'pt':0.02, 'threshold':0.8})
print trial.run()

uni_student = Simulator(True)
p = {'pg':-.1, 'ps':-.1, 'pi':0.1, 'pt':0.2, 'threshold':0.8}
gg = {}
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    gg[skill] = p
trial = SeparateSkillTrial(uni_student, gg)
print trial.run()

"""


"""
# test both sims w/ no specified skills

uni_sim = Simulator(True)
sep_sim = Simulator(False)

obs = uni_sim.give_test()
print obs
c = 0
while True:
    obs = uni_sim.give_problem()
    if obs < 0:
        break
    #print obs
    c += 1
obs = uni_sim.give_test()
print str(c) + " problems"
print obs

print

sep_sim.give_test()
obs = sep_sim.give_test()
print obs
c = 0
while True:
    obs = sep_sim.give_problem()
    if obs < 0:
        break
    #print obs
    c += 1
obs = sep_sim.give_test()
print str(c) + " problems"
print obs



#test both sims with skills

uni_sim = Simulator(True)
sep_sim = Simulator(False)

for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    obs = uni_sim.give_test(skill)
    print obs

c = 0
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    print skill
    while True:
        obs = uni_sim.give_problem()
        if obs < 0:
            break
        #print obs
        c += 1
print str(c) + " problems"
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    obs = uni_sim.give_test(skill)
    print obs

print

for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    obs = sep_sim.give_test(skill)
    print obs

c = 0
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    print skill
    while True:
        obs = sep_sim.give_problem()
        if obs < 0:
            break
        #print obs
        c += 1
print str(c) + " problems"
for skill in ['center', 'x_axis', 'y_axis', 'shape', 'histogram', 'spread', 'h_to_d', 'd_to_h']:
    obs = sep_sim.give_test(skill)
    print obs

"""



"""
#load up x axis problem indices
problems = json.load(open("dump/problems_idx_histogram.csv"))

tutor_probs = []
test_probs = []

for c, v in enumerate(problems):
    if "assess" in v:
        test_probs.append(c)
    else:
        tutor_probs.append(c)

print test_probs
print tutor_probs
model = run_learned_model('histogram')

model.start_student()
for prob in test_probs:
    print model.give_problem(prob)
for prob in tutor_probs:
    print model.give_problem(prob)
for prob in test_probs:
    print model.give_problem(prob)
"""

