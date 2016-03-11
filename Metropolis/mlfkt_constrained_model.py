""" This model implements multi-state LFKT (still with no student ability parameter)
"""

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.stats import invgamma
import math

import time

import parameter

DIRICHLET_SCALE = 300
DIR_LOW_BOUND = 0.01

class MLFKTConstrainedModel:

    sigma = 0.15

    def __init__(self, X, P, intermediate_states, Dsigma):
        """
        :param X: the observation matrix, -1 padded to the right (to make it square)
        :param P: problem indices, -1 padded to the right
        :param intermediate_states: number of intermediate states (between no-mastery and mastery)
        :param Dsigma: model parameter- initial setting of variance for problem difficulty values. Set to 0 to lock to BKT
        :return: nix
        """

        self.data = {}
        self.data['X'] = X
        self.data['P'] = P
        numprobs = int(np.max(P) + 1)
        self.data['num_problems'] = numprobs
        self.N = len(X)
        self.T = len(X[0])
        self.numprobs = numprobs
        self.intermediate_states = intermediate_states
        total_states = intermediate_states + 2
        self.total_states = total_states
        self.params = {}

        #setup initial probability vector...
        #!!!!!!!!!!!!!!!!!!!!!! here L[0] is p(unlearned)
        val = np.ones(total_states)/(total_states + 0.0)
        self.params['L'] = parameter.Parameter(val, 0, 1, (lambda x: 1), (lambda x: self.sample_dir(DIRICHLET_SCALE * x)))
        print "pi starting as:"
        print val

        t_mat = np.ones([total_states, total_states])
        #setup transition triangle...
        for row in range(total_states):
            t_mat[row,0:row] = np.zeros(row)
            if row < total_states - 2:
                t_mat[row,row+2:] = np.zeros(total_states - (row+2))
            t_mat[row,:] = np.random.dirichlet(DIRICHLET_SCALE * t_mat[row,:])

        print "T starting as:"
        print t_mat
        self.params['T'] = parameter.Parameter(t_mat, 0, 1, (lambda x: 1), (lambda x: self.sample_dir_mat(DIRICHLET_SCALE * x)))

        #setup guess vector in really clunky way
        for c in range(intermediate_states + 1):
            self.params['G_' + str(c)] = parameter.Parameter(float(c) / total_states, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                                             (lambda x, c=c: self.sample_guess_prob(x, c)))
        self.params['S'] = parameter.Parameter(-2, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                               (lambda x: self.sample_slip_prob(x)))

        #problem difficulty vector, also in clunky way
        self.emission_mask = []
        self.emission_mats = []
        for c in range(numprobs):
            self.emission_mask.append(False)
            self.emission_mats.append(np.ones((total_states, 2)))
            self.params['D_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x, d_sig: norm.pdf(x, 0, d_sig)),
                                                             (lambda x: np.random.normal(x, 0.15)))

        self.params['Dsigma'] = parameter.Parameter(Dsigma, 0, 3, (lambda x: invgamma.pdf(x, 1, 0, 2)),
                                                    (lambda x: np.random.normal(x, 0.15)))

        #leave room later for test split
        self.test = {}

    """Some helper functions"""
    def log(self, X):
        if X <= 1e-322:
            return -float('inf')
        return math.log(X)

    def exp(self, X):
        if X == float('nan'):
            print "WTFBBQHAX NAN"
        if X == self.log(0):
            return 0
        return math.exp(X)

    def uniform(self, X, a, b):
        if X >= a and X <= b:
            return abs(1.0 / (b-a))
        return 0

    def sample_dir_mat(self, x):
        #sample a matrix where each row is sampled from a dirichlet
        xnew = np.copy(x)
        side_len = np.shape(x)[0]
        for c in range(side_len):
            #if np.min(xnew[c,c:side_len]) < .001 * DIRICHLET_SCALE:
            #    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dirichlet matrix hiccup"
            #    print x
            #    xnew[c,c:side_len] += 100000 * DIRICHLET_SCALE
            xnew[c,c:c+2] = self.sample_dir(xnew[c,c:c+2])
        return xnew

    def sample_dir(self, x):
        #ensure elements stay away from zero
        x = np.random.dirichlet(x)
        if np.min(x) < DIR_LOW_BOUND:
            #print "~~~~~~~~~~~~~~~~~~~~~~~~\tdirichlet bounce"
            #print x
            x[np.argmin(x)] = DIR_LOW_BOUND * 1.1
            x = x / np.sum(x)
            #print x
        return x

    def norm_bound(self, x, low, high):
        new = np.random.normal(x, 0.15)
        if low > high:
            print "GG rekt, bad bounds on guess parameter"
        while new < low or new > high:
            new = np.random.normal(x, 0.15)
        return new

    def sample_slip_prob(self, x):
        low = -3
        high = -self.params['G_' + str(self.total_states-2)].get()
        return self.norm_bound(x, low, high)

    def sample_guess_prob(self, x, state):
        self.emission_mask = [False] * self.data['num_problems']
        #constrain guess probabilities to strictly increase with knowledge state
        if state == 0:
            low = -3
        else:
            low = self.params['G_' + str(state-1)].get()

        if state < self.total_states - 2:
            high = self.params['G_' + str(state+1)].get()
        else:
            high = 3

        return self.norm_bound(x, low, high)

    def make_transitions(self):
        return self.params['T'].get()

    def make_emissions(self, diff, prob_num):
        if self.emission_mask[prob_num]:
            #print str(time.time()) + "\tusing saved emissions"
            return self.emission_mats[prob_num]
        else:
            #print str(time.time()) + "\tcalculating new emissions"
            self.emission_mask[prob_num] = True
            table = self.emission_mats[prob_num]
            #guesses...
            for row in range(self.total_states - 1):
                table[row, 1] = expit(self.params['G_' + str(row)].get() - diff)
                table[row, 0] = 1 - table[row, 1]
            #and slip
            table[row + 1, 0] = expit(self.params['S'].get() + diff)
            table[row + 1, 1] = 1 - table[row + 1, 0]
        """
        print
        print np.array(table)
        print self.params['G_0'].get()
        print self.params['G_1'].get()
        print self.params['S'].get()
        print
        """
        return table

    def make_initial(self):
        return self.params['L'].get()

    """expose parameters"""
    def get_parameters(self):
        return self.params

    """evaluate probability of the setting of parameter paramID, given the setting of the other parameters and the data"""
    def log_posterior(self, paramID):
        X = self.data['X']
        Probs = self.data['P']
        Dsigma = self.params['Dsigma']
        N = self.N
        T = self.T

        if paramID == 'Dsigma':
            ##get p(Dsigma | D) propto p(D | Dsigma) p(Dsigma)
            dprob = 0
            for d in range(self.numprobs):
                dprob += self.log( self.params['D_' + str(d)].prior(Dsigma.get()))
            return self.log(Dsigma.prior()) + dprob

        if 'D_' in paramID:
            self.emission_mask[int(paramID[2:])] = False

        trans = self.make_transitions()
        pi = self.make_initial()
        states = self.total_states

        loglike = 0

        for n in range(N):
            alpha = np.zeros( (T,states) )
            emit = self.make_emissions(self.params['D_' + str(int(Probs[n,0]))].get(), int(Probs[n,0]))
            alpha[0] = np.multiply(pi, emit[:,X[n,0]])

            last_t = 0
            for t in range(1,T):
                last_t = t
                if X[n,t] == -1:
                    break
                emit = self.make_emissions(self.params['D_' + str(int(Probs[n,t]))].get(), int(Probs[n,t]))
                #print B
                alpha[t,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[t-1,:]), trans)
                #if min(alpha[t,:]) < 1e-70:
                #   print "low alpha! " + str(min(alpha[t,:]))
                if min(alpha[t,:]) < 1e-250:
                    print "Oh snappy tappies! " + str(min(alpha[t,:]))

                #print min(alpha[t,:])
            #print sum(alpha[T-1,:])
            #print loglike

            loglike += self.log(sum(alpha[last_t-1,:]))

        #print "final loglike: " + str(loglike)
        if 'D_' in paramID:
            log_prior = self.log(self.params[paramID].prior(Dsigma.get()))
        else:
            log_prior = self.log(self.params[paramID].prior())
        #print "log_prior:     " + str(log_prior)
        log_post = loglike + log_prior
        return log_post

    #okay we finna need some inference up in here
    def load_test_split(self, Xtest, Ptest):
        self.test['X'] = Xtest
        self.test['P'] = Ptest
        self.test['Predictions'] = np.copy(Xtest)
        self.test['Mastery'] = np.copy(Xtest)
        self._predict()

    def _predict(self):
        ## prediction rolls through the forward algorithm only
        ## as we predict only based on past data
        X = self.test['X']
        Probs = self.test['P']
        N = X.shape[0]
        T = X.shape[1]

        #set params to mean
        for id, p in self.params.iteritems():
            avg = np.mean(p.get_samples(), 0)
            p.set(avg)

        pi = self.make_initial()
        trans = self.make_transitions()
        Preds = self.test['Predictions']
        Mast = self.test['Mastery']
        num = 0

        states = self.total_states
        for n in range(N):
            ##Similar to forward algo
            #print "\t\t\tPrediction for test sequence: " + str(n)
            alpha = np.zeros( (T,states) )
            emit = self.make_emissions(self.params['D_' + str(int(Probs[n,0]))].get(), int(Probs[n,0]))

            Mast[n,0] = pi[-1]
            #print P
            #print P[1]
            Preds[n,0] = np.sum(np.multiply(pi, emit[:,1]))
            alpha[0,:] = np.multiply(pi, emit[:,X[n,0]])
            #print "alpha0: " + str(alpha[0,:])
            num += 1
            for t in range(1,T):
                if X[n,t] == -1:
                    break
                emit = self.make_emissions(self.params['D_' + str(int(Probs[n,t]))].get(), int(Probs[n,t]))

                state_probs = np.copy(alpha[t-1,:])
                state_probs = np.dot(state_probs, trans)
                state_probs = state_probs / np.sum(state_probs)
                #print state_probs
                #print state_probs[1]
                Mast[n,t] = state_probs[-1]
                Preds[n,t] = np.sum(np.multiply(state_probs, emit[:, 1]))
                num += 1
                #print B
                alpha[t,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[t-1,:]), trans)
                #print alpha[t,:]
            #print
        self.test['num'] = num
        print "Pi:"
        print pi
        print
        print "Trans:"
        print trans
        print
        print "Emit:"
        self.emission_mask[0] = False
        emit = self.make_emissions(0,0)
        print emit
        print

    #get a single prediction for test sequence n, time t
    def get_prediction(self, n, t):
        return self.test['Predictions'][n,t]

    #get all predictions for test data
    def get_predictions(self):
        return self.test['Predictions']

    def get_mastery(self):
        return self.test['Mastery']

    def get_num_predictions(self):
        return self.test['num']




