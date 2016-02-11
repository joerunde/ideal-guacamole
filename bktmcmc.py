""" oldass crap, ignore
"""

import numpy as np
import math, random
from scipy.stats import norm

def uniform(X, a, b):
	if X >= a and X <= b:
		return abs(1 / (b-a))
	return 0

def prior_T(T):
	return uniform(T, 0, 1)

def prior_L(L):
	return uniform(L, 0, 1)

def prior_G(G):
	return uniform(G, 0, 1)

def prior_S(S):
	return uniform(S, 0, 1)

def pT(Y, T):
	try:
		p = 0
		N = Y.shape[0]
		M = Y.shape[1]
		for n in range(N):
			p += math.log((1-T)**(np.sum(1-Y[n][1:])) * T**(Y[n][M-1]))
		p += math.log(prior_T(T))
		return p
	except ValueError:
		if T < 0.5 and T > 0:
			print "PT val error: " + str(T)
		return -100000

def pL(Y, L):
	try:
		p = 0
		N = Y.shape[0]
		for n in range(N):
			p += math.log(L**(Y[n][0]) * (1-L)**(1 - Y[n][0]))
			#print "In pL..."
			#print p
		p += math.log(prior_L(L))
		return p
	except ValueError:
		return -1000000
def pY(y, Y, n, t, X, T, L, G, S):
	p = 1
	M = Y.shape[1]
	#Y depends on:
	#observation
	if y == 0:
		p *= (G**X[n][t] * (1-G)**(1-X[n][t]))
	else:
		p *= ((1-S)**X[n][t] * S**(1-X[n][t]))

	#transition in
	if t == 0:
		p *= (L**y * (1-L)**(1-y))
	else:
		yp = Y[n][t-1]
		p *= (T**((1-yp) * y) * (1-T)**((1-yp) * (1-y)) * 0**(yp * (1-y)))

	#transition out
	if t < M-1 and Y[n][t] == 0:
		p *= (T**Y[n][t+1] * (1-T)**(1-Y[n][t+1]))
	return p

def pG(G, S, Y, X):
	p = 0
	N = Y.shape[0]
	M = Y.shape[1]

	try:
		for n in range(N):
			p += math.log(G**sum((1-Y[n])*X[n]) * (1-G)**sum((1-Y[n])*(1-X[n])) * S**sum(Y[n]*(1-X[n])) * (1-S)**sum(Y[n]*X[n]))
		p += math.log(prior_G(G))
	except ValueError:
		return -1000000
	return p

def pS(G, S, Y, X):
	p = 0
	N = Y.shape[0]
	M = Y.shape[1]
	try:
		for n in range(N):
			p += math.log(G**sum((1-Y[n])*X[n]) * (1-G)**sum((1-Y[n])*(1-X[n])) * S**sum(Y[n]*(1-X[n])) * (1-S)**sum(Y[n]*X[n]))
		p += math.log(prior_S(S))
		return p
	except ValueError:
		return -1000000

def sample_S(G, S, Y, X):
	#do actual metropolis hastings
	#Q is gaussian
	Snew = np.random.normal(S, 0.01)
	Pp = pS(G, Snew, Y, X)
	P = pS(G, S, Y, X)

	if Pp - P > 500:
		return Snew
	if P - Pp > 500:
		return S
	
	a1 = math.exp(Pp - P)
	if a1 >= 1 or random.random() > 0.1:
		return Snew
	return S

def sample_G(G, S, Y, X):
	#do actual metropolis hastings
	#Q is gaussian
	Gnew = np.random.normal(G, 0.01)
	Pp = pG(Gnew, S, Y, X)
	P = pG(G, S, Y, X)

	if Pp - P > 500:
		return Gnew
	if P - Pp > 500:
		return G
	
	#print P
	#print Pp
	a1 = math.exp(Pp - P)
	if a1 >= 1 or random.random() > 0.1:
		return Gnew
	return G

def sample_T(T, Y):
	#do actual metropolis hastings
	#Q is gaussian
	Tnew = np.random.normal(T, 0.01)
	Pp = pT(Y, Tnew)
	P = pT(Y, T)

	if Pp - P > 500:
		return Tnew
	if P - Pp > 500:
		return T
	
	a1 = math.exp(Pp - P)
	if a1 >= 1 or random.random() > 0.1:
		return Tnew
	return T

def sample_Y(n, t, X, Y, T, L, G, S):
	r = random.random()

	#do actual normalization
	p0 = pY(0, Y, n, t, X, T, L, G, S)
	
	p1 = pY(1, Y, n, t, X, T, L, G, S)
	p0 = p0 / (p1 + p0)

	if r > p0:
		return 1
	return 0

def sample_L(L, Y):
	#do actual metropolis hastings
	#Q is gaussian
	Lnew = np.random.normal(L, 0.01)
	Q = norm.pdf(L, L, 0.1)
	Qp = norm.pdf(Lnew, L, 0.1)
	Pp = pL(Y, Lnew)
	P = pL(Y, L)

	if Pp - P > 500:
		return Lnew
	if P - Pp > 500:
		return L
	
	#print P
	#print Pp
	a1 = math.exp(Pp - P)
	a2 = Q / Qp
	if a1 * a2 >= 1 or random.random() > 0.1:
		return Lnew
	return L


def sample(X, Y, G, S, T, L):
	N = X.shape[0]
	M = X.shape[1]

	for n in range(N):
		for t in range(M):
			Y[n][t] = sample_Y(n, t, X, Y, T, L, G, S)
	L = sample_L(L, Y)
	T = sample_T(T, Y)
	G = sample_G(G, S, Y, X)
	S = sample_S(G, S, Y, X)

	return [Y, G, S, T, L]

##load X
print("starting...")
X = np.loadtxt(open("joesdata.csv","rb"),delimiter=",")
X = X[0:50]

Y = np.zeros((X.shape[0], X.shape[1]))



G = 0.5
S = 0.5
T = 0.0001
L = 0.1

LL = []
TT = []
GG = []
SS = []

for c in range(250000):
	[Y, G, S, T, L] = sample(X, Y, G, S, T, L)
	#print("{0}:\tT:{1:.2f}\tL:{2:.2f}\tG:{3:.2f}\tS:{3:.2f}".format(c, T, L, G, S))
	#print(Y[22])
	if c % 1000 == 0:
		print c
	LL.append(L)
	TT.append(T)
	GG.append(G)
	SS.append(S)
	#print "T: " + str(T)
	#print "L: " + str(L)
	#print "G: " + str(G)

print sum(LL)/250000
print sum(TT)/250000
print sum(GG)/250000
print sum(SS)/250000

