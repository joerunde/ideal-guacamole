""" This model implements a mu, sigma univariate gaussian
"""


import numpy as np
from scipy.stats import norm
import math

import parameter

class GaussianModel:

    def __init__(self, X):
        """
        :param X: list of training samples
        """

        self.data = {}
        self.data['X'] = X

        self.params = {}

        #uninformative priors
        self.params['mu'] = parameter.Parameter(0, -10, 10, (lambda x: self.uniform(x, -10, 10)))
        self.params['sigma'] = parameter.Parameter(1, 0, 5, (lambda x: self.uniform(x, 0, 5)))

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

    """evaluate probability of the setting of parameter paramID, given the setting of the other parameters and the data"""
    def log_posterior(self, paramID):
        X = self.data['X']
        mu = self.params['mu'].get()
        sigma = self.params['sigma'].get()

        loglike = 0
        for x in X:
            loglike += self.log(norm.pdf(x, mu, sigma))

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
        mu = self.params['mu'].get()
        sigma = self.params['sigma'].get()
        loglike = 0
        for x in self.test['X']:
            loglike += self.log(norm.pdf(x, mu, sigma))
        self.score = loglike

    #get the log likelihood of test sequence
    def get_score(self):
        return self.score




