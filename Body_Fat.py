#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from scipy import stats


# In[3]:


import pymc3 as pm


# In[4]:


import multiprocessing
multiprocessing.set_start_method('fork') 


# In[5]:


from divergence import *


# In[6]:


import statsmodels.api as sm


# In[7]:


data = pd.read_csv('https://raw.githubusercontent.com/n-kall/powerscaling-sensitivity/master/case-studies/bodyfat/data/bodyfat.txt', sep = ';')


# In[8]:


data.shape


# In[9]:


print(data.head())


# In[10]:


y_obs = data['siri']
y_obs = y_obs.to_numpy()     # (251,)


# In[11]:


x_obs = data.iloc[:, [4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
x_obs = x_obs.to_numpy()

x1 = x_obs[:, 0]      #(251,)
x2 = x_obs[:, 1]
x3 = x_obs[:, 2]
x4 = x_obs[:, 3]
x5 = x_obs[:, 4]
x6 = x_obs[:, 5]
x7 = x_obs[:, 6]
x8 = x_obs[:, 7]
x9 = x_obs[:, 8]
x10 = x_obs[:, 9]
x11 = x_obs[:, 10]
x12 = x_obs[:, 11]
x13 = x_obs[:, 12]


# In[12]:


plt.scatter(x13, y_obs)


# ## Derive Bayesian Posterior

# In[13]:


basic_model = pm.Model()

with basic_model:
    beta_1 = pm.Normal('beta_1', 0, 1)
    beta_2 = pm.Normal('beta_2', 0, 1)
    beta_3 = pm.Normal('beta_3', 0, 1)
    beta_4 = pm.Normal('beta_4', 0, 1)
    beta_5 = pm.Normal('beta_5', 0, 1)
    beta_6 = pm.Normal('beta_6', 0, 1)
    beta_7 = pm.Normal('beta_7', 0, 1)
    beta_8 = pm.Normal('beta_8', 0, 1)
    beta_9 = pm.Normal('beta_9', 0, 1)
    beta_10 = pm.Normal('beta_10', 0, 1)
    beta_11 = pm.Normal('beta_11', 0, 1)
    beta_12 = pm.Normal('beta_12', 0, 1)
    beta_13 = pm.Normal('beta_13', 0, 1)
    
    beta_0 = pm.StudentT('beta_0', nu = 3, mu = 0, sigma = np.sqrt(9.2))
    sigma = pm.HalfStudentT('sigma', nu = 3, sigma = np.sqrt(9.2))
    
    mu = beta_0 + x1*beta_1 + x2*beta_2 + x3*beta_3 + x4*beta_4 + x5*beta_5 + x6*beta_6 + x7*beta_7\
    + x8*beta_8 + x9*beta_9 + x10*beta_10 + x11*beta_11 + x12*beta_12 + x13*beta_13      #(251,)
    
    Y = pm.Normal('y_obs', mu = mu, sigma = sigma, observed = y_obs)


# In[14]:


with basic_model:
    trace = pm.sample(draws=1000, tune=1000, chains=4, cores = 2)


# In[15]:


_ = pm.traceplot(trace)


# In[16]:


pm.summary(trace)


# In[17]:


base_draws_beta_1 = trace[beta_1]      #(4000,)
base_draws_beta_2 = trace[beta_2]
base_draws_beta_3 = trace[beta_3]
base_draws_beta_4 = trace[beta_4]
base_draws_beta_5 = trace[beta_5]
base_draws_beta_6 = trace[beta_6]
base_draws_beta_7 = trace[beta_7]
base_draws_beta_8 = trace[beta_8]
base_draws_beta_9 = trace[beta_9]
base_draws_beta_10 = trace[beta_10]
base_draws_beta_11 = trace[beta_11]
base_draws_beta_12 = trace[beta_12]
base_draws_beta_13 = trace[beta_13]


# In[18]:


base_draws_beta_1_13 = np.vstack((base_draws_beta_1, base_draws_beta_2, base_draws_beta_3, base_draws_beta_4, base_draws_beta_5,\
                       base_draws_beta_6, base_draws_beta_7, base_draws_beta_8, base_draws_beta_9, base_draws_beta_10,\
                       base_draws_beta_11, base_draws_beta_12, base_draws_beta_13)).T         #(4000,13)

base_draws_beta_0 = trace[beta_0][:, np.newaxis]      #(4000,1)

base_draws_sigma = trace[sigma][:, np.newaxis]        #(4000,1)


# ## Calculate weights for prior/likelihood perturbation

# In[19]:


# Calculate weights for Prior perturbation

# log(joint prior distribution^(α-1)) = (α-1)*log(joint prior distribution) 
# = (α-1) * (Σi=1-13 log(P(βi)) + log(P(β0)) + log(P(σ)))
def weights_prior(alpha):
    log_weights_prior_beta_1_13 = np.log(stats.norm.pdf(base_draws_beta_1_13, 0, 1))
    
    log_weights_prior_beta_0 = np.log(stats.t.pdf(base_draws_beta_0, 3, 0, np.sqrt(9.2)))
    log_weights_prior_sigma = np.log(stats.t.pdf(base_draws_sigma, 3, 0, np.sqrt(9.2)))

    log_weights_prior = np.concatenate((log_weights_prior_beta_1_13, log_weights_prior_beta_0, log_weights_prior_sigma), axis = 1)
    
    return (alpha - 1) * np.sum(log_weights_prior, axis = 1)


# In[20]:


print(weights_prior(1.01))


# In[26]:


weights_prior(1.01).shape


# In[31]:


# Calculate weights for Likelihood perturbation
# (α-1)*Σi=1-251 log(L(μ,σ|yi))
def weights_likelihood(alpha):
    weights_likelihood = list()

    for i in range(np.shape(base_draws_beta_1_13)[0]):
        mu = x_obs @ base_draws_beta_1_13[i,:][:, np.newaxis] + base_draws_beta_0[i]
        sigma = base_draws_sigma[i]
        log_likelihood = np.sum(np.log(stats.norm.pdf(y_obs[:, np.newaxis], mu, sigma)))
        weight = (alpha - 1) * log_likelihood
    
        weights_likelihood.append(weight)
    
    return np.array(weights_likelihood)


# In[32]:


weights_likelihood(1.01).shape


# ## Calculate Sensitivity

# In[38]:


# Prior Perturbation Sensitivity
def distance_prior(alpha):
    distance = list()
    
    weights = weights_prior(alpha)
    
    for i in range(13):
        kde_posterior = sm.nonparametric.KDEUnivariate(base_draws_beta_1_13[:,i])
        kde_posterior_perturbed = sm.nonparametric.KDEUnivariate(base_draws_beta_1_13[:,i])
        kde_posterior.fit()
        kde_posterior_perturbed.fit(weights = weights, fft = False)
    
        sens_beta = jensen_shannon_divergence_from_kde(kde_posterior, kde_posterior_perturbed)
    
        distance.append(sens_beta)
    
    return np.array(distance)


# In[39]:


sensitivity_prior_perturb = (distance_prior(0.99) + distance_prior(1.01))/2/np.log2(0.01)
print(sensitivity_prior_perturb)


# In[40]:


def distance_likelihood(alpha):
    distance = list()
    
    weights = weights_likelihood(alpha)
    
    for i in range(13):
        kde_posterior = sm.nonparametric.KDEUnivariate(base_draws_beta_1_13[:,i])
        kde_posterior_perturbed = sm.nonparametric.KDEUnivariate(base_draws_beta_1_13[:,i])
        kde_posterior.fit()
        kde_posterior_perturbed.fit(weights = weights, fft = False)
    
        sens_beta = jensen_shannon_divergence_from_kde(kde_posterior, kde_posterior_perturbed)
    
        distance.append(sens_beta)
    
    return np.array(distance)


# In[41]:


sensitivity_prior_perturb = (distance_likelihood(0.99) + distance_likelihood(1.01))/2/np.log2(0.01)
print(sensitivity_prior_perturb)




