""" This model implements multi-state LFKT (still with no student ability parameter)
"""

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.stats import invgamma
import math

import parameter

class MLFKTModel:

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

        #super clunky way to do parameter vectors or matrices...
        #setup initial probability vector...
        #!!!!!!!!!!!!!!!!!!!!!! here L_0 is p(unlearned)
        for c in range(intermediate_states + 1):
            val = 1.0 / (total_states + 4)
            if c == 0:
                val = 5.0 / (total_states + 4)
            self.params['L_'+str(c)] = parameter.Parameter(val, 0, 1, (lambda x: self.uniform(x, 0, 1)))

        #setup transition triangle...
        for row in range(intermediate_states + 1):
            for col in range(row+1, total_states):
                self.params['T_'+str(row) + '_' + str(col)] = \
                    parameter.Parameter(1.0 / (total_states - row), 0, 1, (lambda x: self.uniform(x, 0, 1)))

        #setup guess vector
        for c in range(intermediate_states + 1):
            self.params['G_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)))

        self.params['S'] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)))

        #problem difficulty vector
        for c in range(numprobs):
            self.params['D_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x, d_sig: norm.pdf(x, 0, d_sig) ))

        self.params['Dsigma'] = parameter.Parameter(Dsigma, 0, 3, (lambda x: invgamma.pdf(x, 1, 0, 2)))

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

    def make_transitions(self):
        table = []
        for row in range(self.total_states):
            table.append([0] * self.total_states)
            for col in range(row + 1, self.total_states):
                table[-1][col] = self.params['T_' + str(row) + '_' + str(col)].get()
            table[-1][row] = 1 - np.sum(table[-1])
        """
        print
        print np.array(table)
        print self.params['T_0_1'].get()
        print self.params['T_0_2'].get()
        print self.params['T_1_2'].get()
        print
        """
        return np.array(table)

    def make_emissions(self, diff):
        table = []
        #guesses...
        for row in range(self.total_states - 1):
            table.append([0, 0])
            table[-1][1] = expit(self.params['G_' + str(row)].get() - diff)
            table[-1][0] = 1 - table[-1][1]
        #and slip
        table.append([expit(self.params['S'].get() + diff), 0])
        table[-1][1] = 1 - table[-1][0]
        """
        print
        print np.array(table)
        print self.params['G_0'].get()
        print self.params['G_1'].get()
        print self.params['S'].get()
        print
        """

        return np.array(table)

    def make_initial(self):
        array = [0] * self.total_states
        for c in range(self.total_states - 1):
            array[c] = self.params['L_' + str(c)].get()
        array[-1] = 1 - np.sum(array)
        """
        print
        print np.array(array)
        print self.params['L_0'].get()
        print self.params['L_1'].get()
        print
        """
        return np.array(array)

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

        trans = self.make_transitions()
        pi = self.make_initial()
        states = self.total_states

        loglike = 0

        for n in range(N):
            alpha = np.zeros( (T,states) )
            emit = self.make_emissions(self.params['D_' + str(int(Probs[n,0]))].get())
            alpha[0] = np.multiply(pi, emit[:,X[n,0]])

            last_t = 0
            for t in range(1,T):
                last_t = t
                if X[n,t] == -1:
                    break
                emit = self.make_emissions(self.params['D_' + str(int(Probs[n,t]))].get())
                #print B
                alpha[t,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[t-1,:]), trans)
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
            emit = self.make_emissions(self.params['D_' + str(int(Probs[n,0]))].get())

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
                emit = self.make_emissions(self.params['D_' + str(int(Probs[n,t]))].get())

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




