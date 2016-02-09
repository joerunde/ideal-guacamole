""" This model implements simple LFKT without a student ability parameter
    (Equivalent to regular BKT if Dsigma = 0)
"""


import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.stats import invgamma
import math

import parameter

class LFKTModel:

    sigma = 0.15

    def __init__(self, X, P, Dsigma):
        """
        :param X: the observation matrix, -1 padded to the right (to make it square)
        :param P: problem indices, -1 padded to the right
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

        self.params = {}

        self.params['L'] = parameter.Parameter(0.5, 0, 1, (lambda x: self.uniform(x, 0, 1)))
        self.params['T'] = parameter.Parameter(0.5, 0, 1, (lambda x: self.uniform(x, 0, 1)))
        self.params['G'] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)))
        self.params['S'] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)))
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

    def make_emissions(self, G, S, diff, ability):
        gp = expit(G - diff + ability)
        sp = expit(S + diff - ability)
        return np.array([[1-gp,gp],[sp,1-sp]])


    """expose parameters"""
    def get_parameters(self):
        return self.params

    """evaluate probability of the setting of parameter paramID, given the setting of the other parameters and the data"""
    def log_posterior(self, paramID):
        X = self.data['X']
        L = self.params['L'].get()
        Tr = self.params['T'].get()
        S = self.params['S'].get()
        G = self.params['G'].get()
        Dsigma = self.params['Dsigma']
        #D = self.params['D'].get()
        Probs = self.data['P']
        N = self.N
        T = self.T

        if paramID == 'Dsigma':
            ##get p(Dsigma | D) propto p(D | Dsigma) p(Dsigma)
            dprob = 0
            for d in range(self.numprobs):
                dprob += self.log( self.params['D_' + str(d)].prior(Dsigma.get()))
            return self.log(Dsigma.prior()) + dprob

        A = np.array([[1-Tr,Tr],[0,1]])
        #B = np.array([[1-G,G],[S,1-S]])
        P = np.array([1-L, L])

        loglike = 0

        for n in range(N):
            alpha = np.zeros( (T,2) )
            B = self.make_emissions(G, S, self.params['D_' + str(int(Probs[n,0]))].get(), 0)
            #print ("Initial B:")
            #print B
            alpha[0] = np.multiply(P, B[:,X[n,0]])

            last_t = 0
            for t in range(1,T):
                last_t = t
                if X[n,t] == -1:
                    break
                B = self.make_emissions(G, S, self.params['D_' + str(int(Probs[n,t]))].get(), 0)
                #print B
                alpha[t,:] = np.dot( np.multiply( B[:,X[n,t]], alpha[t-1,:]), A)
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
        L = self.params['L'].get()
        Tr = self.params['T'].get()
        S = self.params['S'].get()
        G = self.params['G'].get()
        #Dsigma = self.params['Dsigma']
        #D = self.params['D'].get()
        Probs = self.test['P']
        N = X.shape[0]
        T = X.shape[1]

        A = np.array([[1-Tr,Tr],[0,1]])
        P = np.array([1-L, L])
        Preds = self.test['Predictions']
        Mast = self.test['Mastery']
        num = 0

        for n in range(N):
            ##Similar to forward algo
            #print "\t\t\tPrediction for test sequence: " + str(n)
            alpha = np.zeros( (T,2) )
            B = self.make_emissions(G, S, self.params['D_' + str(int(Probs[n,0]))].get(), 0)
            #print("sumb:   " + str(np.sum(B)))
            #print("B:      " + str(B))
            Mast[n,0] = L
            #print P
            #print P[1]
            Preds[n,0] = np.sum(np.multiply(P, B[:,1]))
            alpha[0,:] = np.multiply(P, B[:,X[n,0]])
            #print "alpha0: " + str(alpha[0,:])
            num += 1
            for t in range(1,T):
                if X[n,t] == -1:
                    break
                B = self.make_emissions(G, S, self.params['D_' + str(int(Probs[n,t]))].get(), 0)

                state_probs = np.copy(alpha[t-1,:])
                state_probs = np.dot(state_probs, A)
                state_probs = state_probs / np.sum(state_probs)
                #print state_probs
                #print state_probs[1]
                Mast[n,t] = state_probs[1]
                Preds[n,t] = np.sum(np.multiply(state_probs, B[:,1]))
                num += 1
                #print B
                alpha[t,:] = np.dot( np.multiply( B[:,X[n,t]], alpha[t-1,:]), A)
                #print alpha[t,:]
            #print
        self.test['num'] = num

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




