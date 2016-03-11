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

        #setup initial probability vector...
        #!!!!!!!!!!!!!!!!!!!!!! here L[0] is p(unlearned)
        val = np.ones(total_states)/(total_states + 0.0)
        self.params['L'] = parameter.Parameter(val, 0, 1, (lambda x: 1), (lambda x: self.sample_dir(DIRICHLET_SCALE * x)))
        #print "pi starting as:"
        #print val

        t_mat = np.ones([total_states, total_states])
        #setup transition triangle...
        for row in range(total_states):
            t_mat[row,0:row] = np.zeros(row)
            t_mat[row,:] = np.random.dirichlet(DIRICHLET_SCALE * t_mat[row,:])

        #print "T starting as:"
        #print t_mat
        self.params['T'] = parameter.Parameter(t_mat, 0, 1, (lambda x: 1), (lambda x: self.sample_dir_mat(DIRICHLET_SCALE * x)))

        #setup guess vector in really clunky way
        for c in range(intermediate_states + 1):
            self.params['G_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                                             (lambda x: self.sample_guess_prob(x)))
        self.params['S'] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                               (lambda x: self.sample_guess_prob(x)))

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
            xnew[c,c:side_len] = self.sample_dir(xnew[c,c:side_len])
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

    def sample_guess_prob(self, x):
        self.emission_mask = [False] * self.data['num_problems']
        return np.random.normal(x, 0.15)

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

    """expose parameters for BO search. Needs to be an array of [min,max] bounds"""
    def get_params_for_BO(self):
        bounds = []
        for c in range(self.numprobs):
            #begin with problem difficulties
            bounds.append([-1,1])

        #Pi
        for c in range(self.total_states-1):
            bounds.append([0,1])

        #Trans
        for c in range(self.total_states-1):
            for i in range(c,self.total_states-1):
                bounds.append([0,1])

        #G/S
        for c in range(self.total_states):
            bounds.append([-3,3])

        self.load_test_split(self.data['X'], self.data['P'], False)
        return bounds

    """ Set parameters from an array, evaluate model, return """
    def evaluate_params_from_BO(self, param_array):
        errval = 1

        off = 0
        for c in range(self.numprobs):
            #begin with problem difficulties
            self.params['D_'+str(c)].set(param_array[off])
            off += 1

        #Pi
        L = []
        for c in range(self.total_states-1):
            L.append(param_array[off])
            off += 1
        if sum(L) > 1:
            #err here
            return errval + 10*(sum(L)-1)
        L.append(1-sum(L))
        self.params['L'].set(np.array(L))

        #Trans
        trans = np.copy(self.params['T'].get())
        for c in range(self.total_states-1):
            for i in range(c,self.total_states-1):
                trans[c,i] = param_array[off]
                off += 1
            trans[c,-1] = 1-sum(trans[c,:-1])
            if trans[c,-1] < 0:
                return errval + 10 * (-trans[c,-1])
        self.params['T'].set(trans)

        #G/S
        for c in range(self.total_states):
            if c < self.total_states - 1:
                self.params['G_' + str(c)].set(param_array[off])
            else:
                self.params['S'].set(param_array[off])
            off += 1

        #dsigma hack for now
        self._predict(True)
        pred = self.get_predictions()
        num = self.get_num_predictions()
        err = pred - self.test['X']
        rmse = np.sqrt(np.sum(err**2)/num)
        return rmse
        #print self.params['L'].get()
        #return -self.log_posterior('Dsigma')

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
                    #print "low alpha! " + str(min(alpha[t,:]))
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


    #get dat viterbi on
    def viterbi(self):
        for id, p in self.params.iteritems():
            avg = np.mean(p.get_samples(), 0)
            p.set(avg)

        X = self.data['X']
        Probs = self.data['P']
        Dsigma = self.params['Dsigma']
        N = self.N
        T = self.T

        trans = self.make_transitions()
        pi = self.make_initial()
        states = self.total_states

        loglike = 0

        stateseqs = []
        mastseqs = []
        momentseqs = []

        for n in range(N):

            ptrs = np.zeros( (T + 1, states) )
            alpha = np.ones( (T+1, states))
            alpha[0,:] = pi
            last_t = 0
            for t in range(0,T):
                last_t = t
                if X[n,t] == -1:
                    break
                emit = self.make_emissions(self.params['D_' + str(int(Probs[n,t]))].get(), int(Probs[n,t]))
                for c in range(states):
                    gg = [alpha[t, old] * trans[old,c] for old in range(states)]
                    alpha[t+1, c] = max(gg)
                    ptrs[t+1, c] = int(gg.index(max(gg)))
                    alpha[t+1, c] *= emit[c, X[n,t]]

            #print alpha
            #print ptrs

            #np.set_printoptions(precision=3)

            mast_seq = []
            stateseq = []
            momentseq = []
            stateseq.append(alpha[last_t].argmax())
            mast_seq.append(alpha[last_t,1] / sum(alpha[last_t,:]))
            for t in range(last_t)[::-1]:
                stateseq.append(int(ptrs[t,stateseq[-1]]))
                mast_seq.append(alpha[t,1] / sum(alpha[t,:]))

            stateseq = list(reversed(stateseq))
            mast_seq = list(reversed(mast_seq))

            firstneg = 0
            firstpoint1 = 0
            for t in range(last_t):
                if t < 2:
                    momentseq.append(0)
                else:
                    moment = np.polyfit([0,1,2], [mast_seq[t-2], mast_seq[t-1], mast_seq[t]], 1)[0]
                    #momentseq.append( math.ceil(moment * 10 - .5)/10.0)
                    momentseq.append(moment)
                    if firstneg == 0 and moment < 0:
                        firstneg = t
                    if firstpoint1 == 0 and moment < .1:
                        firstpoint1 = t
            if firstneg == 0:
                firstneg = last_t
            if firstpoint1 == 0:
                firstpoint1 = last_t


            stateseqs.append(stateseq)
            mastseqs.append(mast_seq)
            momentseqs.append(momentseq)

            #print np.array(mast_seq)
            print str(stateseq) + '\t' + str(firstneg+1) + '\t' + str(firstpoint1+1)
            tmp = [int(x) for x in X[n,0:last_t]]
            tmp.insert(0,0)
            print tmp
            #print momentseq
        return stateseqs, mastseqs, momentseqs


    #okay we finna need some inference up in here
    def load_test_split(self, Xtest, Ptest, predict_now=True):
        self.test['X'] = Xtest
        self.test['P'] = Ptest
        self.test['Predictions'] = np.copy(Xtest) + 0.0
        self.test['Mastery'] = np.copy(Xtest) + 0.0
        if predict_now:
            self._predict()

    def _predict(self, use_current_params=False):
        ## prediction rolls through the forward algorithm only
        ## as we predict only based on past data
        X = self.test['X']
        Probs = self.test['P']
        N = X.shape[0]
        T = X.shape[1]

        #set params to mean
        if not use_current_params:
            for id, p in self.params.iteritems():
                #print id
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
                #print state_probs
                state_probs = np.dot(state_probs, trans)
                #print state_probs
                state_probs = state_probs / np.sum(state_probs)
                #print state_probs
                #print state_probs
                #print state_probs[1]
                Mast[n,t] = state_probs[-1]

                Preds[n,t] = np.sum(np.multiply(state_probs, emit[:, 1]))
                num += 1
                #print B
                alpha[t,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[t-1,:]), trans)
                #print alpha[t,:]

            #print Preds[n,:]
            #print X[n,:]
            #print
        self.test['num'] = num
        self.test['Mastery'] = Mast
        """
        if not use_current_params:
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
        """

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


    #setup generating process
    def start_student(self):
        #assume 2-state for now
        #and use *current* parameters
        seed = np.random.random()
        pi = self.params['L'].get()[0]

        #print "P_unlearned: " + str(pi)

        if seed < pi:
            self.student_state = 0
        else:
            self.student_state = 1


    def give_problem(self, prob_id, transition=True):

        diff = self.params['D_'+str(prob_id)].get()
        emit = self.make_emissions(diff, prob_id)

        #print "Student state: " + str(self.student_state)

        correct_prob = emit[self.student_state,1]
        #print "P(correct)   : " + str(correct_prob)

        seed = np.random.random()
        if seed < correct_prob:
            ans = 1
        else:
            ans = 0

        if self.student_state == 0 and transition:
            trans = self.params['T'].get()[0,1]
            seed = np.random.random()
            if seed < trans:
                self.student_state = 1

        return ans

    def set_4_params(self, g, s, l, t):
        self.params['G_0'].set(g)
        self.params['S'].set(s)
        L = np.array([l, 1-l])
        self.params['L'].set(L)
        T = np.array( [ [1-t, t], [0, 1] ] )
        self.params['T'].set(T)

        #print "L is " + str(self.params['L'].get())