import numpy.random
import gaussian_model
import linreg_model
import mcmc_sampler
import numpy as np

beta = np.array( [1, -0.2, 0, 0, 0, 0, 0, 0, 0, 0] )

feat = 10

X = np.zeros( (50,feat+1) )
for c in range(50):
    X[c,0:feat] = np.random.normal(0,3,feat)
    X[c,-1] = np.dot(beta, X[c,0:feat]) + np.random.normal(0,1.5)

print X
print len(beta)

model = linreg_model.LinRegModel(X)

print "k"

mcmc = mcmc_sampler.MCMCSampler(model, 0.15)

mcmc.burnin(200)
print "burned"
mcmc.MH(200)
print "plotting"
mcmc.plot_samples('testout/', 'testplot')


test = np.zeros( (50,feat+1) )
for c in range(50):
    test[c,0:feat] = np.random.normal(0,3,feat)
    test[c,-1] = np.dot(beta, test[c,0:feat])

model.load_test_split(test)

print "RMSE:", model.get_score()