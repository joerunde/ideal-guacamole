""" oldass test
"""

import numpy as np
from scipy.stats import norm
import math, random
from matplotlib import pyplot as plt

class MultiGaussMH:
	sigma = 1
	samples = []
	x = 0

	def conditional_Prob(self, x):
		#in general, the probability of some parameter, conditioned on every other one in the model
		#doesn't have to integrate to 1. This one integrates to 6
		#in this case, no dependence on self. Just mix of gaussians
		return math.log(norm.pdf(x, -1, 1) + 5 * norm.pdf(x, 3, 0.5))
		#return norm.pdf(x, 50, 2)

	def MH_proposal(self):
		#return (random.random() - 0.5) * 12
		return np.random.normal(self.x, self.sigma)

	def save_sample(self):
		self.samples.append(self.x)

	def MH_sample(self):
		proposed = self.MH_proposal()
		P_new = self.conditional_Prob(proposed)
		P_old = self.conditional_Prob(self.x)

		a = math.exp(P_new - P_old)

		# true if a >= 1, or with probability a
		if random.random() < a:
			self.x = proposed

	def plot_samples(self):
		numbins = int(8 * math.log(len(self.samples)))
		n, bins, patches = plt.hist(self.samples, numbins, normed=1)
		y = []
		for b in bins:
			y.append(math.exp(self.conditional_Prob(b)) / 6)
		plt.plot(bins, y, 'r--')
		plt.savefig('gg')
		plt.clf()
		plt.plot(bins, y, 'go')
		plt.savefig('ggg')

	def burnin(self, iterations):
		for c in range(iterations):
			self.MH_sample()
	
	def MH(self, iterations):
		for c in range(iterations):
			self.MH_sample()
			self.save_sample()

print norm.logpdf(0, 100, 1)
x = MultiGaussMH()

x.burnin(1000)

x.MH(3000)
x.plot_samples()
