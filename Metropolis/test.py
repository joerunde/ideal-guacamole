import numpy.random
import gaussian_model
import mcmc_sampler

X = numpy.random.normal(3.4, 2.5, 50)
model = gaussian_model.GaussianModel(X)

print "k"

mcmc = mcmc_sampler.MCMCSampler(model, 0.15)

mcmc.burnin(200)
print "burned"
mcmc.MH(400)
print "plotting"
mcmc.plot_samples('testout/', 'testplot')