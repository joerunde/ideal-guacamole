""" This class runs metropolis-hastings MCMC over a model
    The model must expose its parameters and provide log-posterior evaluations over training data
"""
import lfkt_model
import parameter
import numpy as np
import math, random, cPickle
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity as KD
from cloud.serialization.cloudpickle import dump

class MCMCSampler:

    def __init__(self, model, sigma):
        """@type model: lfkt_model.LFKTModel"""
        self.model = model
        self.sigma = sigma #for the lfkt case before I had this at 0.15

    def MH_sample(self, paramID, parameter):
        #paramID is just the name (given by the model) of the parameter
        """@type parameter : parameter.Parameter"""

        #quick hacky check... lock LFKT to BKT if that's the model we're using
        #cause this is research land
        if paramID == "Dsigma" and parameter.get() == 0:
            return

        P_old = self.model.log_posterior(paramID)
        self.MH_proposal(parameter)
        P_new = self.model.log_posterior(paramID)
        a = self.exp(P_new - P_old)

        #if paramID == 'G':
        #    print "---------------"
        #    print P_new
        #    print P_old
        #    print a

        #Leave the parameter at the new value with probability a
        #(a can be > 1)
        if random.random() < a:
            #leave it
            pass
        else:
            parameter.revert()

    def MH_proposal(self, parameter):
        """@type parameter : parameter.Parameter"""
        #use normal distribution as proposal function
        #val = parameter.get()
        #proposed = np.random.normal(val, self.sigma)
        #parameter.set(proposed)

        #jk use parameter's sample fn
        proposed = parameter.sample()
        parameter.set(proposed)

    def burnin(self, iterations):
        params = self.model.get_parameters()
        for i in range(iterations):
            for id, p in params.iteritems():
                self.MH_sample(id, p)

    def MH(self, iterations):
        params = self.model.get_parameters()
        for i in range(iterations):
            for id, p in params.iteritems():
                self.MH_sample(id, p)
                p.save()

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

    def plot_samples(self, folder = '', title = ''):
        params = self.model.get_parameters()

        for id, p in params.iteritems():
            """@type p : parameter.Parameter"""
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

            """
            samples = np.array([p.get_samples()]).T
            a = p.min
            b = p.max
            low = np.min(samples)
            high = np.max(samples)

            #if samples all same value... don't try to do anything here
            if low == high:
                continue

            frac = (high - low) / (b - a + 0.001)
            band = 0.1 * (high - low + 0.001)
            numbins = int(25 * frac * math.log(len(p.get_samples())))

            #fit kernel density estimator, plot histogram and estimated density curve together
            kde = KD(kernel='gaussian', bandwidth=band).fit(samples)
            n, bins, patches = plt.hist(p.get_samples(), numbins, normed=1)
            log_dens = kde.score_samples(np.array([bins]).T)
            plt.plot(bins, np.exp(log_dens), 'r-')

            #find the maximum point, set the parameter to that and plot on histogram too
            MAP = self._get_MAP(kde, a, b)
            p.set(MAP)
            plt.plot([MAP], np.exp(kde.score_samples([MAP])), 'go')

            #Clean up and label figure
            plt.title(title + " MAP estimate: " + str(MAP))
            plt.ylabel("Posterior(" + id + ")")
            plt.xlabel(id)
            x1,x2,y1,y2 = plt.axis()
            plt.axis((a,b,y1,y2))

            plt.savefig(folder + id.replace('_','/',1) + "_" + title)
            plt.clf()
            """

        print("Plots saved!")

    def set_MAP(self):
        params = self.model.get_parameters()

        for id, p in params:
            """@type p : parameter.Parameter"""
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


