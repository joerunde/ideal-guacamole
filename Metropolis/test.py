import numpy.random
import gaussian_model
import mcmc_sampler

X = numpy.random.normal(3.4, 2.5, 100)
model = gaussian_model.GaussianModel(X)

mcmc = mcmc_sampler.MCMCSampler(model, 0.15)

mcmc.burnin(200)
mcmc.MH(800)

mcmc.plot_samples('testout/', 'testplot')