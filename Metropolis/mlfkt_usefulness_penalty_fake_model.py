""" This model implements multi-state LFKT (still with no student ability parameter)
"""
import itertools
import numpy as np
from scipy.stats import norm
from scipy.stats import laplace
from scipy.special import expit
from scipy.stats import invgamma
import math

import time

import parameter

DIRICHLET_SCALE = 300
DIR_LOW_BOUND = 0.01

class MLFKTSUPModel:

    sigma = 0.15

    def __init__(self, X, P, S, intermediate_states, Dsigma, l1_b=0.15):
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
        self.data['S'] = S
        numprobs = int(np.max(P) + 1)
        numskills = int(np.max(S) + 1)
        self.data['num_problems'] = numprobs
        self.total_skills = numskills
        self.N = len(X)
        self.T = len(X[0])
        self.numprobs = numprobs
        self.intermediate_states = intermediate_states

        if intermediate_states > 0:
            raise(ValueError("don't put multiple states in this model yet"))

        total_states = intermediate_states + 2
        self.total_states = total_states
        self.params = {}

        print "L1 b:", l1_b

        #We need one set of BKT params for each skill...
        for sk in range(numskills):
            #setup initial probability vectors...
            #!!!!!!!!!!!!!!!!!!!!!! here L[0] is p(unlearned)
            val = np.ones(total_states)/(total_states + 0.0)
            self.params['L-'+str(sk)+'-'] = parameter.Parameter(val, 0, 1, (lambda x: 1), (lambda x: self.sample_dir(DIRICHLET_SCALE * x)))
            print "pi starting as:"
            print val

            t_mat = np.ones([total_states, total_states])
            #setup transition triangle...
            for row in range(total_states):
                t_mat[row,0:row] = np.zeros(row)
                t_mat[row,:] = np.random.dirichlet(DIRICHLET_SCALE * t_mat[row,:])

            print "T starting as:"
            print t_mat
            self.params['T-'+str(sk)+'-'] = parameter.Parameter(t_mat, 0, 1, (lambda x: 1), (lambda x: self.sample_dir_mat(DIRICHLET_SCALE * x)))

            #setup guess vector in really clunky way
            for c in range(intermediate_states + 1):
                self.params['G-'+str(sk)+'-_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                                                 (lambda x: self.sample_guess_prob(x)))
            self.params['S-'+str(sk)+'-'] = parameter.Parameter(0, -3, 3, (lambda x: self.uniform(x, -3, 3)),
                                                   (lambda x: self.sample_guess_prob(x)))

            #skill "usefulness" parameter. Same range/sampling as Guess/Slip
            for sk2 in range(numskills):
                if sk != sk2:
                    self.params['U-'+str(sk)+'-'+str(sk2)] = parameter.Parameter(0, 0, 3, (lambda x: laplace.pdf(x, 0, l1_b)),
                                                       (lambda x: self.sample_KT_param(x)))


        #problem difficulty vector, also in clunky way
        """for c in range(numprobs):
            self.params['D_' + str(c)] = parameter.Parameter(0, -3, 3, (lambda x, d_sig: norm.pdf(x, 0, d_sig)),
                                                             (lambda x: np.random.normal(x, 0.15)))
        """
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
        return np.random.normal(x, 0.15)

    def sample_KT_param(self, x):
        return max(0, np.random.normal(x, 0.15))

    def make_transitions(self, skill):
        return self.params['T-'+str(skill)+'-'].get()

    def make_emissions(self, diff, prob_num, skill, skills):
        #print str(time.time()) + "\tcalculating new emissions"
        table = np.ones((self.total_states,2))

        uvec = []
        for sk in range(self.total_skills):
            if sk == skill:
                uvec.append(0)
                continue
            uvec.append(self.params['U-'+str(skill)+'-'+str(sk)].get())

        #guesses...
        for row in range(self.total_states - 1):
            guess = self.params['G-'+str(skill)+'-_' + str(row)].get()
            # integrate over the other skills effects
            """sks = [x for x in itertools.product([0,1], repeat=self.total_skills)]
            prob = 0
            sump = 0
            for gg in sks: #for each setting of all masteries
                u = 0
                p = 1
                if gg[skill] == 0: # remove half of settings
                    continue

                for c in range(len(gg)):
                    if c == skill:
                        continue
                    #multiply in the probability of this skill being mastered or not
                    p = p * skills[c, gg[c]]
                    #add in or subtract the KT bonus
                    if gg[c]:
                        u += uvec[c]
                    else:
                        u -= uvec[c]
                prob += p * expit(guess + u)
                sump += p"""
            #print "sum p should be 1:", sump
            u = 0
            for sk in range(self.total_skills):
                #print "Belief:", skills[sk,1]
                if sk != skill:
                    # penalize here if sk has not been mastered
                    u += self.params['U-'+str(skill)+'-'+str(sk)].get() * (skills[sk,1] - 0.5)
            prob = expit(guess + u)

            table[row, 1] = prob
            table[row, 0] = 1 - table[row, 1]


        #and slip

        slip = self.params['S-'+str(skill)+'-'].get()
        # integrate over the other skills effects
        """ sks = [x for x in itertools.product([0,1], repeat=self.total_skills)]
        prob = 0
        sump = 0
        for gg in sks: #for each setting of all masteries
            u = 0
            p = 1
            if gg[skill] == 0: # remove half of settings
                continue

            for c in range(len(gg)):
                if c == skill:
                    continue
                #multiply in the probability of this skill being mastered or not
                p = p * skills[c, gg[c]]
                #add in or subtract the KT bonus
                if gg[c]:
                    u += uvec[c]
                else:
                    u -= uvec[c]
            prob += p * expit(slip - u)
            sump += p """
        #print "sum p should be 1:", sump

        u = 0
        for sk in range(self.total_skills):
            #print "Belief:", skills[sk,1]
            if sk != skill:
                # penalize here if sk has not been mastered
                u += self.params['U-'+str(skill)+'-'+str(sk)].get() * (skills[sk,1] - 0.5)
        prob = expit(slip - u)

        table[row + 1, 0] = prob
        table[row + 1, 1] = 1 - prob

        return table

    def make_initial(self, skill):
        return self.params['L-'+str(skill)+'-'].get()

    """expose parameters"""
    def get_parameters(self):
        return self.params

    """evaluate probability of the setting of parameter paramID, given the setting of the other parameters and the data"""
    def log_posterior(self, paramID, fake=False):
        X = self.data['X']
        Probs = self.data['P']
        Skill = self.data['S']
        Dsigma = self.params['Dsigma']
        N = self.N
        T = self.T

        if paramID == 'Dsigma':
            ##get p(Dsigma | D) propto p(D | Dsigma) p(Dsigma)
            dprob = 0
            for d in range(self.numprobs):
                dprob += 1#self.log( self.params['D_' + str(d)].prior(Dsigma.get()))
            return self.log(Dsigma.prior()) + dprob

        states = self.total_states

        loglike = 0

        for n in range(N):
            #alphas now store rolling joints per skill
            # alpha = [skill x state] matrix
            alpha = np.zeros( (self.total_skills, states) )
            beliefs = np.zeros( (self.total_skills, states) )

            for sk in range(self.total_skills):
                alpha[sk,:] = self.make_initial(sk)
                beliefs[sk,:] = alpha[sk,:]

            #emit = self.make_emissions(self.params['D_' + str(int(Probs[n,0]))].get(), int(Probs[n,0]), int(Skill[n,0]))
            #alpha[0] = np.multiply(pi, emit[:,X[n,0]])
            last_t = 0
            for t in range(0,T):
                last_t = t
                if X[n,t] == -1:
                    break
                skill = int(Skill[n,t])
                #print skill
                #print X[n,t]
                #print
                emit = self.make_emissions(0,0, skill, beliefs)
                trans = self.make_transitions(skill)
                alpha[skill,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[skill,:]), trans)
                beliefs[skill,:] = alpha[skill,:] / np.sum(alpha[skill,:])

                #if min(alpha[skill,:]) < 1e-150:
                #    print "Oh snappy tappies! " + str(min(alpha[t,:]))

                #print min(alpha[t,:])
            #print sum(alpha[T-1,:])
            #print loglike

            for sk in range(self.total_skills):
                loglike += self.log(sum(alpha[sk,:]))

        if fake:
            log_prior = 0
            for k,v in self.params.iteritems():
                log_prior += self.log(v.prior())
                    #print log_prior
            return loglike + log_prior

        #print "final loglike: " + str(loglike)
        log_prior = self.log(self.params[paramID].prior())
        #print "log_prior:     " + str(log_prior)
        log_post = loglike + log_prior
        return log_post


    #get dat viterbi on
    def viterbi(self, skill):

        X = self.data['X']
        Probs = self.data['P']
        Skill = self.data['S']
        Dsigma = self.params['Dsigma']
        N = self.N
        T = self.T

        #trans = self.make_transitions()
        pi = self.make_initial(skill)
        states = self.total_states

        stateseqs = []
        mastseqs = []

        times = []
        times2 = []

        n7_0probs = []
        n7_15probs = []
        n7_7probs = []
        n7obs = []
        n7_other_beliefs = []


        for n in range(N):

            #okay so the ptrs. etc code is just gonna sit here but...
            #we're gonna brute force this thing cause it's not working
            #assume the 15-problem sequences from simulation
            #then just 15 possible transitions (16 if it never happens)
            #which is most likely?

            sequence_probs = [1] * 16
            these_obs = []
            dot2 = True

            ptrs = np.zeros( (T + 1, states) )
            alpha = np.ones( (T+1, states))
            alpha[0,:] = pi
            beliefs = np.zeros( (self.total_skills, states) )
            for sk in range(self.total_skills):
                beliefs[sk,:] = np.copy(pi)

            last_t = 0
            t = -1
            # actual_t is the actual step number for the data matrices X, P and S
            # t is the step number for this skill
            #print "hi"
            for actual_t in range(0,T):
                last_t = t
                if X[n,actual_t] == -1:
                    break

                sk = int(Skill[n, actual_t])
                #print sk

                emit = self.make_emissions(0,0, sk, beliefs)
                trans = self.make_transitions(sk)

                #if sk == 0:
                    #print "Other skill belief: ", beliefs[1,1]
                    #print "Guess:", emit[0,1]
                    #print "Slip:", emit[1,0]

                #update beliefs w/ regular forward stuff
                beliefs[sk,:] = np.dot( np.multiply( emit[:,X[n, actual_t]], beliefs[sk,:]), trans)
                beliefs[sk,:] /= np.sum(beliefs[sk,:])
                #print "beliefs for sk:", sk, beliefs[sk,:]

                if sk != skill:
                    continue
                t += 1

                if beliefs[skill,1] > 0.5 and dot2:
                    dot2 = False
                    times2.append(t)

                for c in range(states):
                    # max operator for transitioning
                    gg = [alpha[t, old] * trans[old,c] for old in range(states)]
                    alpha[t+1, c] = max(gg)
                    # point backwards
                    ptrs[t+1, c] = int(gg.index(max(gg)))
                    alpha[t, c] *= emit[c, X[n, actual_t]]

                if n == 7:
                    n7obs.append(int(X[n,actual_t]))
                    n7_other_beliefs.append(beliefs[1,1])
                #update sequence probabilities
                for seq in range(16):
                    if t < seq: #we're guessing
                        sequence_probs[seq] *= emit[0,int(X[n, actual_t])]
                        if n == 7 and seq == 0:
                            n7_0probs.append(emit[0,int(X[n, actual_t])])
                        if n == 7 and seq == 7:
                            print "guess"
                            n7_7probs.append(emit[0,int(X[n, actual_t])])
                        if n == 7 and seq == 15:
                            n7_15probs.append(emit[0,int(X[n, actual_t])])
                        #    print "Guess:", int(X[n, actual_t]), emit[0,int(X[n, actual_t])]

                    else:   #we're slipping
                        sequence_probs[seq] *= emit[1,int(X[n, actual_t])]
                        if n == 7 and seq == 0:
                            n7_0probs.append(emit[1,int(X[n, actual_t])])
                        if n == 7 and seq == 7:
                            print "slip"
                            n7_7probs.append(emit[1,int(X[n, actual_t])])

                        if n == 7 and seq == 15:
                            n7_15probs.append(emit[1,int(X[n, actual_t])])
                        #if seq == 8:
                        #    print "Slip:", int(X[n, actual_t]), emit[1,int(X[n, actual_t])]

                these_obs.append(int(X[n,actual_t]))

            for seq in range(16):
                #if seq < 15:
                sequence_probs[seq] *= 0.1
                sequence_probs[seq] *= (0.9 ** seq)

            times.append(sequence_probs.index(max(sequence_probs)))
            #print these_obs, times[-1]
            #if n % 50 == 0:
            #    print sequence_probs.index(max(sequence_probs))

            #print alpha
            #print ptrs
            #print last_t

            #np.set_printoptions(precision=3)
            mast_seq = []
            stateseq = []

            # probably off by one...
            last_t += 1

            stateseq.append(alpha[last_t].argmax())
            mast_seq.append(alpha[last_t,1] / sum(alpha[last_t,:]))
            for t in range(last_t)[::-1]:
                stateseq.append(int(ptrs[t,stateseq[-1]]))
                mast_seq.append(alpha[t,1] / sum(alpha[t,:]))

            stateseq = list(reversed(stateseq))
            mast_seq = list(reversed(mast_seq))

            stateseqs.append(stateseq)
            mastseqs.append(mast_seq)

            #print np.array(mast_seq)
            #print str(stateseq)
            tmp = [int(x) for x in X[n,0:last_t]]
            tmp.insert(0,0)
            #print tmp
            #print momentseq
        print times
        #print
        #print times2

        print n7obs
        print n7_other_beliefs
        print
        print n7_0probs
        gg = 1
        for x in n7_0probs:
            gg *= x
        print gg

        print n7_7probs
        gg = 1
        for x in n7_7probs:
            gg *= x
        print gg


        print n7_15probs
        gg = 1
        for x in n7_15probs:
            gg *= x
        print gg

        print times[7]


        return stateseqs, mastseqs, times


    #okay we finna need some inference up in here
    def load_test_split(self, Xtest, Ptest, Skilltest, getMAP = True):
        self.test['X'] = Xtest
        self.test['P'] = Ptest
        self.test['S'] = Skilltest
        self.test['Predictions'] = np.copy(Xtest)
        self.test['Mastery'] = np.copy(Xtest)
        self._predict(getMAP)

    #hacky MAP estimation
    def set_map_params(self):
        #loop through all samples, return setting with highest likelihood on the data
        num_samples = len(self.params['Dsigma'].get_samples())
        maxp = -float('inf')
        maxc = 0
        print "searching for MAP"
        for c in range(num_samples):
            for k,v in self.params.iteritems():
                v.set(v.get_samples()[c])
            p = self.log_posterior('loljk', True)
            if p >= maxp:
                print p
                maxp = p
                maxc = c

        print "maxp:", maxp
        for k,v in self.params.iteritems():
                v.set(v.get_samples()[maxc])

    def _predict(self, getMAP = True):
        ## prediction rolls through the forward algorithm only
        ## as we predict only based on past data
        X = self.test['X']
        Probs = self.test['P']
        N = X.shape[0]
        T = X.shape[1]
        Skill = self.test['S']

        #set params to mean
        if getMAP:
            self.set_map_params()

        #pi = self.make_initial()
        #trans = self.make_transitions()
        Preds = self.test['Predictions']
        Mast = self.test['Mastery']
        num = 0

        states = self.total_states
        for n in range(N):
            ##Similar to forward algo
            #print "\t\t\tPrediction for test sequence: " + str(n)
            alpha = np.zeros( (self.total_skills, states) )
            beliefs = np.zeros( (self.total_skills, states) )
            for sk in range(self.total_skills):
                alpha[sk,:] = self.make_initial(sk)
                beliefs[sk,:] = alpha[sk,:]

            """
            Mast[n,0] = pi[-1]
            #print P
            #print P[1]
            Preds[n,0] = np.sum(np.multiply(pi, emit[:,1]))
            alpha[0,:] = np.multiply(pi, emit[:,X[n,0]])
            #print "alpha0: " + str(alpha[0,:])
            num += 1
            """

            for t in range(0,T):
                if X[n,t] == -1:
                    break
                skill = int(Skill[n,t])
                emit = self.make_emissions(0,0, skill, beliefs)
                trans = self.make_transitions(skill)

                #make prediction (normalized current alpha for state probs)
                belief = alpha[skill,:] / np.sum(alpha[skill,:])
                Preds[n,t] = np.sum(np.multiply(belief, emit[:, 1]))
                num += 1
                Mast[n,t] = belief[-1]

                #Transition alpha on the observation
                alpha[skill,:] = np.dot( np.multiply( emit[:,X[n,t]], alpha[skill,:]), trans)
                beliefs[skill,:] = alpha[skill,:] / np.sum(alpha[skill,:])

            #print
        self.test['num'] = num
        print "Skill u-offsets:"
        for sk in range(self.total_skills):
            for sk2 in range(self.total_skills):
                if sk != sk2:
                    print sk, sk2, self.params['U-'+str(sk)+'-'+str(sk2)].get()
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




