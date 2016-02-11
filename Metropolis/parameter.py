""" Parameter interface for models using metropolis-hastings MCMC
"""


class Parameter:
    #Parameters must be able to sample themselves, and store lists of previous values as MCMC sampling is done
    #Min/max values are for convenience so posterior plots for values in the same ranges are easily compared
    #Prior should be a function lambda which can be evaluated with the parameter's value
    def __init__(self, val, minimum, maximum, prior_function, sample_function):
        """
        :param val: Initial value
        :param minimum: Minimum value to consider for plots and MAP estimate
        :param maximum: Maximum value to consider for plots and MAP estimate
        :param prior_function: function lambda to evaluate the prior P(param = val)
        :param sample_function: function lambda which returns new sample for MCMC. Takes current val as argument
        :return: nothing
        """
        self.vals = []
        self.val = val
        self.min = minimum
        self.max = maximum
        self.prior_fn = prior_function
        self.sample_fn = sample_function
        self.last_val = None

    def set(self, val):
        self.last_val = self.val
        self.val = val

    def get(self):
        return self.val

    def get_samples(self):
        return self.vals

    def save(self):
        #Save current value
        self.vals.append(self.val)

    def prior(self, other=None):
        if other is None:
            return self.prior_fn(self.val)
        else:
            return self.prior_fn(self.val, other)

    def sample(self):
        return self.sample_fn(self.val)

    def revert(self):
        #Provided for convenience of trying a parameter out, and reverting if it's worse
        if self.last_val is not None:
            self.val = self.last_val
