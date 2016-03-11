""" This class runs metropolis-hastings MCMC over a model
    The model must expose its parameters and provide log-posterior evaluations over training data
"""
import mlfkt_model
import parameter
import numpy as np
import math, random, cPickle
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity as KD
from cloud.serialization.cloudpickle import dump

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint
from moe.easy_interface.simple_endpoint import gp_hyper_opt
from moe.easy_interface import simple_endpoint

class BOSampler:

    def __init__(self, model):
        """@type model: mlfkt_model.MLFKTModel"""
        self.model = model
        #self.samples = []

        bounds = model.get_params_for_BO()

        self.exp = Experiment(bounds)

    def BO_sample(self):
        x = gp_next_points(self.exp)[0]
        y = self.model.evaluate_params_from_BO(x)
        np.set_printoptions(3)
        print np.array(x)
        print y
        self.exp.historical_data.append_sample_points([SamplePoint(x,y,0.001)])
        #print gp_hyper_opt(self.exp.historical_data.points_sampled)
        """pts = self.exp.historical_data.points_sampled
        vals = self.exp.historical_data.points_sampled_value
        noise = self.exp.historical_data.points_sampled_noise_variance
        gg = []
        for c in range(len(pts)):
            gg.append((pts[c], vals[c], noise[c]))
        print gp_hyper_opt(gg)"""

        #simple_endpoint.

    def get_samples(self):
        return self.exp.historical_data

    """
    def _plot(self, folder, title, id, samples, a, b):
        low = np.min(samples)
        high = np.max(samples)

        #if samples all same value... don't try to do anything here
        if abs(low - high) < 0.001:
            print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Skipping parameter: " + id
            return

        frac = (high - low) / (b - a + 0.001)
        band = 0.1 * (high - low + 0.001)
        numbins = int(25 * frac * math.log(len(samples))) + 2

        #fit kernel density estimator, plot histogram and estimated density curve together
        kde = KD(kernel='gaussian', bandwidth=band).fit(samples)
        n, bins, patches = plt.hist(samples, numbins, normed=1)
        log_dens = kde.score_samples(np.array([bins]).T)
        plt.plot(bins, np.exp(log_dens), 'r-')

        #find the maximum point, set the parameter to that and plot on histogram too
        MAP = self._get_MAP(kde, a, b)
        #p.set(MAP)
        plt.plot([MAP], np.exp(kde.score_samples([MAP])), 'go')

        #Clean up and label figure
        plt.title(title + " MAP estimate: " + str(MAP))
        plt.ylabel("Posterior(" + id + ")")
        plt.xlabel(id)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((a,b,y1,y2))

        plt.savefig(folder + id.replace('_','/',1) + "_" + title)
        plt.clf()

        return MAP
    """
    """
    def plot_samples(self, folder = '', title = ''):
        params = self.model.get_parameters()

        for id, p in params.iteritems():
            val = p.get()
            if not hasattr(val, '__len__'):
                self._plot(folder, title, id, np.array([p.get_samples()]).T, p.min, p.max)
            else:
                for c in range(len(val)):
                    if not hasattr(val[c], '__len__'):
                        sam = np.array([[x[c] for x in p.get_samples()]]).T
                        self._plot(folder, title, id + '_' + str(c), sam, p.min, p.max)
                    else:
                        for i in range(len(val[c])):
                            #assume no more depth
                            sam = np.array([[x[c, i] for x in p.get_samples()]]).T
                            self._plot(folder, title, id + '_' + str(c) + '_' + str(i), sam, p.min, p.max)
        print("Plots saved!")
    """
    """
    def set_MAP(self):
        params = self.model.get_parameters()

        for id, p in params:
            samples = np.array([p.get_samples()]).T
            a = p.min
            b = p.max
            low = np.min(samples)
            high = np.max(samples)
            band = 0.1 * (high-low + 0.001)

            #fit kernel density estimator
            kde = KD(kernel='gaussian', bandwidth=band).fit(samples)

            #find the maximum point, set the parameter to that
            MAP = self._get_MAP(kde, a, b)
            p.set(MAP)

    def _get_MAP(self, kde, a=0, b=1):
        #return current MAP estimate of parameters (in range [a,b) )
        #uses kernel density estimation, with gaussian kernel
        n = 10000
        tempx = np.array(range(n))
        n += 0.0
        x = (tempx / n) * (b - a) + a

        log_dens = kde.score_samples(np.array([x]).T)
        return x[np.argmax(log_dens)]
    """
    def log(self, X):
        if X <= 1e-322:
            return -float('inf')
        return math.log(X)

    def exp(self, X):
        if X == float('nan'):
            print "WTFBBQHAX NAN"
        if X == self.log(0):
            return 0
        if X > 709:
            return float('inf')
        return math.exp(X)

    def save_model(self, fname):
        dump(self.model, open(fname, "wb"))

def load_model(fname):
    model = cPickle.load(open(fname, "rb"))
    return model


