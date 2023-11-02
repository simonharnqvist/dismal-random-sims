from scipy import stats
import yaml
import numpy as np
from simulate import Simulation

def random_param_value(distribution, distr_params):
   """Sample a random parameter value from a distribution"""

   if distribution.lower() == "gamma":
      val = stats.gamma.rvs(a = distr_params[0], scale = distr_params[1])
   elif distribution.lower() == "uniform":
      rng = np.random.default_rng()
      val = rng.uniform(distr_params[0], distr_params[1])
   else:
      raise NotImplementedError("Distribution not implemented")
   return val