""" This model implements a mu, sigma univariate gaussian
"""


import numpy as np
from scipy.stats import norm
from scipy.stats import laplace
import math

import parameter

class LinRegModel:

    def __init__(self, X):
        """
        :param X: list of training samples
        """

        self.data = {}
        self.data['X'] = X

        feats = X.shape[1] - 1

        self.params = {}

        for c in range(feats):
            self.params[str(c)] = parameter.Parameter(0, -10, 10, (lambda x: laplace.pdf(x, 0, 1)), (lambda x: np.random.normal(x, 0.1)))
            #self.params[str(c)] = parameter.Parameter(0, -10, 10, (lambda x: 1), (lambda x: np.random.normal(x, 0.25)))

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

    """expose parameters"""
    def get_parameters(self):
        return self.params

    def get_error(self, row):
        y = row[-1]
        yhat = 0
        for k, v in self.params.iteritems():
            beta = v.get()
            yhat += row[int(k)] * beta

        #print "y:", y
        #print "yhat:", yhat
        return yhat - y


    """evaluate probability of the setting of parameter paramID, given the setting of the other parameters and the data"""
    def log_posterior(self, paramID):
        X = self.data['X']

        #using gaussian error model N(0,1)
        loglike = 0
        for c in range(X.shape[0]):
            err = self.get_error(X[c,:])
            loglike += self.log(norm.pdf(err, 0, 3))
            #print norm.pdf(err, 0, 5)

        log_prior = self.log(self.params[paramID].prior())
        #print "log_prior:     " + str(log_prior)
        log_post = loglike + log_prior

        #print loglike
        #print log_prior
        #print log_post

        return log_post

    #okay we finna need some inference up in here
    def load_test_split(self, Xtest):
        self.test['X'] = Xtest
        self._score()

    #score log likelihood of test sequence
    def _score(self):
        X = self.test['X']

        for id, p in self.params.iteritems():
            #print id
            avg = np.mean(p.get_samples(), 0)
            p.set(avg)

        #using gaussian error model N(0,1)
        errs = []
        for c in range(X.shape[0]):
            err = self.get_error(X[c,:])
            errs.append(err)

        errs = np.array(errs)
        errs = errs ** 2
        mean = np.mean(errs)
        rmse = math.sqrt(mean)

        self.score = rmse

    #get the log likelihood of test sequence
    def get_score(self):
        return self.score




