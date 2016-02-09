import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.stats import invgamma
import math, random, time, sys, cPickle
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity as KD

class LFKT:

    sigma = 0.15
    bkt = False

    #for a BKT we have initial, transition, guess, and slip
    #but we also need to sample each student state along
    #the trajectories- Y: N x T

    samples = {}
    params = {}
    data = {}
    test = {}

    #params to store
    store_list = []

    
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

    def __init__(self, X, P, bktonly = False):
        #X is the observation matrix
        #one student per row for now
        self.data['X'] = X
        self.data['P'] = P
        numprobs = int(np.max(P) + 1)
        self.data['num_problems'] = numprobs
        N = len(X)
        T = len(X[0])
        
        Y = np.zeros( (N,T) )
        self.params['Y'] = Y
        self.params['L'] = 0.5
        self.params['T'] = 0.5
        self.params['G'] = 0
        self.params['S'] = 0
        self.params['D'] = np.zeros(numprobs)
        self.params['Dsigma'] = 0.1
        self.params['MAP'] = {}
        #TODO: ability

        if bktonly:
            self.store_list = ['L','T','G','S']
            self.params['Dsigma'] = 0
            self.bkt = True
        else:
            self.store_list = ['L','T','G','S','Dsigma','D']
        for param in self.store_list:
            self.samples[param] = []
        
    def MH_sample(self, paramID):
        #paramID is a list, the first value is the entry in self.params
        #the rest are indices for the proposal function, in case we're
        #sampling one value from a vector/matrix/tensor

        #index down to the actual value to read
        val = self.params[paramID[0]]
        #for i in paramID:
        #    val = val[i]

        proposed = self.MH_proposal(paramID)
        P_new = self.log_posterior(proposed, paramID)
        P_old = self.log_posterior(val, paramID)
        a = self.exp(P_new - P_old)

        #if paramID[0] == 'G':
        #    print "---------------"
        #    print P_new
        #    print P_old
        #    print a

        #Change the parameter to the new value with probability a
        #(a can be > 1)
        if random.random() < a:
            self.params[paramID[0]] = proposed

    def MH_proposal(self, paramID):
        if paramID[0] == 'T' or paramID[0] == 'L' or paramID[0] == 'G' or paramID[0] == 'S' or paramID[0] == 'Dsigma':
            return np.random.normal(self.params[paramID[0]], self.sigma)
        #otherwise return difficulty array
        if paramID[0] == 'D':
            newd = np.random.normal(self.params[paramID[0]][paramID[1]], self.sigma)
            arr = np.copy(self.params[paramID[0]])
            arr[paramID[1]] = newd
            return arr

        print("shitshitshit wrong parameter ID in MH_proposal")
        return 0

    def burnin(self, iterations):
        for i in range(iterations):
            self.MH_sample(['T'])
            self.MH_sample(['L'])
            self.MH_sample(['G'])
            self.MH_sample(['S'])
            if not self.bkt:
                self.MH_sample(['Dsigma'])
                for j in range(self.data['num_problems']):
                    self.MH_sample(['D',j])
    
    def MH(self, iterations):
        for i in range(iterations):
            self.burnin(1)
            self.save_sample()

    def save_sample(self):
        for k, v in self.params.iteritems():
            if k in self.store_list:
                self.samples[k].append(v)    

    def plot_samples(self, folder = '', title = ''):
        print("Saving plots...")
        numbins = int(8 * math.log(len(self.samples[self.store_list[0]])))
        
        for p in self.store_list:
            if p == 'D':
                continue
            samples = np.array([self.samples[p]]).T
            a = np.min(samples)
            b = np.max(samples)
            band = 0.1 * (b-a + 0.001)
            kde = KD(kernel='gaussian', bandwidth=band).fit(samples)
            n, bins, patches = plt.hist(self.samples[p], numbins, normed=1)
            log_dens = kde.score_samples(np.array([bins]).T)
            plt.plot(bins, np.exp(log_dens), 'r-')
            MAP = self.get_MAP(kde, a, b)
            self.params['MAP'][p] = MAP
            plt.plot([MAP], np.exp(kde.score_samples([MAP])), 'go')
            plt.title(title + " MAP estimate: " + str(MAP))
            plt.ylabel("Posterior(" + p + ")")
            plt.xlabel(p)
            x1,x2,y1,y2 = plt.axis()
            plt.axis((-3,3,y1,y2))
            if p == 'L' or p == 'T':
                plt.axis((0,1,y1,y2))
            plt.savefig(folder + p + "_" + title)
            plt.clf()
        
        self.params['MAP']['D'] = self.params['D']
        if not self.bkt:
            print("Working on difficulty params...")

            p = 'D'

            data = np.array(self.samples[p])
            for j in range(self.data['num_problems']):
                samples = np.array([data[:,j]]).T
                #print samples
                a = np.min(samples)
                b = np.max(samples)
                band = 0.1 * (b-a + 0.001)
                kde = KD(kernel='gaussian', bandwidth=band).fit(samples)
                n, bins, patches = plt.hist(samples, numbins, normed=1)
                log_dens = kde.score_samples(np.array([bins]).T)
                plt.plot(bins, np.exp(log_dens), 'r-')
                MAP = self.get_MAP(kde, a, b)
                self.params['MAP']['D'][j] = MAP
                plt.plot([MAP], np.exp(kde.score_samples([MAP])), 'go')
                plt.title(title + " MAP estimate: " + str(MAP))
                plt.ylabel("Posterior(" + p + ")")
                plt.xlabel("Problem " + str(j))
                x1,x2,y1,y2 = plt.axis()
                plt.axis((-3,3,y1,y2))
                plt.savefig(folder + "Difficulty/problem" + str(j) +  "_" + title)
                plt.clf()

        print("Plots saved!")


    def get_MAP(self, kde, a=0, b=1):
        #return current MAP estimate of parameters (in range [a,b) )
        #uses kernel density estimation, with gaussian kernel
        n = 10000
        tempx = np.array(range(n))
        n += 0.0
        x = (tempx / n) * (b - a) + a 
        log_dens = kde.score_samples(np.array([x]).T)
        return x[np.argmax(log_dens)]
        
    
    def uniform(self, X, a, b):
        if X >= a and X <= b:
            return abs(1.0 / (b-a))
        return 0

    def prior(self, x, paramID):
        p = paramID[0]
        #print "prior" + str(paramID) + str(x)
        if p == 'T' or p == 'L':
            return self.uniform(x, 0, 1)
        if p == 'G' or p == 'S':
            return self.uniform(x, -3, 3)
        if p == 'Dsigma':
            return invgamma.pdf(x, 1, 0, 2) #Inverse-Gamma(1,2)
        if p == 'D':
            return norm.pdf(x[paramID[1]], 0, self.params['Dsigma']) 
        print("wtfhax:" + str(paramID))
        return 0

    def make_emissions(self, G, S, diff, ability):
        gp = expit(G - diff + ability)
        sp = expit(S + diff - ability)
        return np.array([[1-gp,gp],[sp,1-sp]])

    def log_posterior(self, x, paramID):
        X = self.data['X']
        L = self.params['L']
        Tr = self.params['T']
        S = self.params['S']
        G = self.params['G']
        D = self.params['D']
        Probs = self.data['P']
        N = X.shape[0]
        T = X.shape[1]

        pid = paramID[0]
        if pid == 'L':
            L = x
        elif pid == 'T':
            Tr = x
        elif pid == 'G':
            G = x
        elif pid == 'S':
            S = x
        elif pid == 'D':
            D = x
        elif pid == 'Dsigma':
            ##get p(Dsigma | D) propto p(D | Dsigma) p(D)
            dprob = 1
            for d in D:
                dprob *= norm.pdf(d, 0, x)
            return self.log(self.prior(x, paramID)) + self.log(dprob)

        A = np.array([[1-Tr,Tr],[0,1]])
        #B = np.array([[1-G,G],[S,1-S]])
        P = np.array([1-L, L])

        loglike = 0

        #print paramID
        #print "D:"
        #print D
        #print "Probs:"
        #print Probs

        for n in range(N):
            alpha = np.zeros( (T,2) )
            B = self.make_emissions(G, S, D[Probs[n,0]], 0)
            #print ("Initial B:")
            #print B
            alpha[0] = np.multiply(P, B[:,X[n,0]])

            for t in range(1,T):
                if X[n,t] == -1:
                    break
                B = self.make_emissions(G, S, D[Probs[n,t]], 0)
                #print B
                alpha[t,:] = np.dot( np.multiply( B[:,X[n,t]], alpha[t-1,:]), A)
            #print sum(alpha[T-1,:])
            #print loglike
            #!!!!!! this depends on t holding the last iteration of the loop
            loglike += self.log(sum(alpha[t-1,:]))

        #print "final loglike: " + str(loglike)
        log_prior = self.log(self.prior(x, paramID))
        #print "log_prior:     " + str(log_prior)
        log_post = loglike + log_prior
        return log_post

    def save(self, fname):
        d = {}
        d['samples'] = self.samples
        d['data'] = self.data
        d['test'] = self.test
        d['params'] = self.params
        d['model'] = self
        cPickle.dump(d, open(fname, "wt"))


    #okay we finna need some inference up in here
    def load_test_split(self, Xtest, Ptest):
        self.test['X'] = Xtest
        self.test['P'] = Ptest
        self.test['Predictions'] = np.copy(Xtest)
        self.test['Mastery'] = np.copy(Xtest)
        self.predict()

    def predict(self):
        ## prediction rolls through the forward algorithm only
        ## as we predict only based on past data
        X = self.test['X']
        L = self.params['MAP']['L']
        Tr = self.params['MAP']['T']
        S = self.params['MAP']['S']
        G = self.params['MAP']['G']
        D = self.params['MAP']['D']
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
            B = self.make_emissions(G, S, D[Probs[n,0]], 0)
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
                B = self.make_emissions(G, S, D[Probs[n,t]], 0)

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

def LFKT_load(fname):
    d = cPickle.load(open(fname, "r+"))
    model = d['model']
    model.params = d['params']
    model.test = d['test']
    model.samples = d['samples']
    model.data = d['data']
    return model


