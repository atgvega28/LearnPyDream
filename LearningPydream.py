from pydream.core import run_dream
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
import inspect
from pydream.convergence import Gelman_Rubin
import matplotlib.pyplot as plt
from pysb.examples.robertson import model

#Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0, 40, 50)
# model.parameters['k2'].value = 0.1
# model.parameters['k3'].value = 1
solver = Solver(model, tspan)
solver.run()

print(model.rules)
for rxn in model.reactions:
    print(rxn)
print(model.parameters)

for obs in model.observables:
    plt.plot(tspan, solver.yobs[obs.name], lw=2, label=obs.name)
plt.legend(loc=0)
# plt.show()
# quit()

# Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with uniform priors.
original_params = np.log10([param.value for param in model.parameters_rules()])
print(original_params)
# By printing original params, one gets the following
#[-1.39794001  7.47712125  4.        ]

# Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits = original_params - 3
print(lower_limits)
# By printing lower_limits, one gets the following:
#[-4.39794001  4.47712125  1.        ]

# Load experimental data to which Robertson model will be fit here.
# The "experimental data" in this case is just the total C trajectory at the default
# model parameters with a standard deviation of 0.01.
pydream_path = os.path.dirname(inspect.getfile(run_dream))
print(inspect.getfile(run_dream))
print(pydream_path)
from pydream.core import run_dream
location = pydream_path+'/examples/robertson/exp_data/'
exp_data_ctot = np.loadtxt(location+'exp_data_ctotal.txt')
'''
print(exp_data_ctot)
[0.         0.02815469 0.04996189 0.06772421 0.08271075 0.09568046
 0.10712091 0.11736312 0.12664126 0.13512705 0.14295015 0.15021059
 0.15698735 0.16334381 0.16933143 0.17499278 0.18036338 0.18547317
 0.19034762 0.19500866 0.1994752  0.20376372 0.20788856 0.21186256
 0.21569699 0.21940193 0.2229863  0.22645817 0.2298248  0.23309275
 0.23626797 0.2393559  0.24236147 0.24528923 0.24814335 0.25092763
 0.25364563 0.25630061 0.25889559 0.26143337 0.26391658 0.26634764
 0.26872882 0.27106224 0.27334991 0.27559366 0.27779525 0.27995632
 0.2820784  0.28416298]
 '''

exp_data_sd_ctot = np.loadtxt(location+'exp_data_sd_ctotal.txt')
'''
print(exp_data_sd_ctot)
 [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
  0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
  0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01
  0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01]
 '''
plt.errorbar(tspan, exp_data_ctot, yerr=exp_data_sd_ctot, fmt='o', ms=2, capsize=2)
plt.show()
quit()


# Example code for functions
def xfunc(value=10, number=1):
    if value > number:
        print("lets do")
    else:
        print('its over')


xfunc(number=4, value=5)
quit()

# def hello_world():
#     print('hello world')

hello_world = 10
print(hello_world)

# TODO: Start from here next time --LAH

# Create scipy normal probability distributions for data likelihoods
like_ctot = norm(loc=exp_data_ctot, scale=exp_data_sd_ctot)
print(like_ctot)
# <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f9a481b37f0>
# there is an object that is created and it is stored in a memory location

# Create lists of sampled pysb parameter names to use for subbing in parameter values in likelihood function.
pysb_sampled_parameter_names = [param.name for param in model.parameters_rules()]
# print(pysb_sampled_parameter_names)
# ['k1', 'k2', 'k3']


# Define likelihood function to generate simulated data that corresponds to experimental time points.
# This function should take as input a parameter vector (parameter values are in the order dictated by first argument to run_dream function below).
# The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):

    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}


    for pname, pvalue in param_dict.items():
        # Change model parameter values to current location in parameter space

        model.parameters[pname].value = 10 ** (pvalue)

    # Simulate experimentally measured Ctotal values.

    solver.run()

    # Calculate log probability contribution from simulated experimental values.

    logp_ctotal = np.sum(like_ctot.logpdf(solver.yobs['C_total']))

    # If model simulation failed due to integrator errors, return a log probability of -inf.
    if np.isnan(logp_ctotal):
        logp_ctotal = -np.inf

    return logp_ctotal

print(model.parameters)
likelihood(np.log10(np.array([0.04, 30000000.0, 10000.0])))


x =[1,2,3]
y = np.array([ 1,2,3])



# # Add vector of PySB rate parameters to be sampled as unobserved random variables to DREAM with uniform priors.
#
# original_params = np.log10([param.value for param in model.parameters_rules()])
# # Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
# lower_limits = original_params - 3
#
# parameters_to_sample = SampledParam(uniform, loc=lower_limits, scale=6)
#
# sampled_parameter_names = [parameters_to_sample]
#
# niterations = 10000
# converged = False
# total_iterations = niterations
# nchains = 5
#
# print(__name__)
#
# # This is optional
# if __name__ == '__main__':
#
#     # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
#     sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations, nchains=nchains,
#                                        multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1,
#                                        model_name='robertson_dreamzs_5chain', verbose=True)
#
#     # Save sampling output (sampled parameter values and their corresponding logps).
#     for chain in range(len(sampled_params)):
#         np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
#                 sampled_params[chain])
#         np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations), log_ps[chain])
#
#     # Check convergence and continue sampling if not converged
#
#     GR = Gelman_Rubin(sampled_params)
#     print('At iteration: ', total_iterations, ' GR = ', GR)
#     np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)
#
#     old_samples = sampled_params
#     if np.any(GR > 1.2):
#         starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
#         while not converged:
#             total_iterations += niterations
#
#             sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, start=starts,
#                                                niterations=niterations,
#                                                nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
#                                                history_thin=1, model_name='robertson_dreamzs_5chain', verbose=True,
#                                                restart=True)
#
#             for chain in range(len(sampled_params)):
#                 np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
#                         sampled_params[chain])
#                 np.save('robertson_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations),
#                         log_ps[chain])
#
#             old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
#             GR = Gelman_Rubin(old_samples)
#             print('At iteration: ', total_iterations, ' GR = ', GR)
#             np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)
#
#             if np.all(GR < 1.2):
#                 converged = True
#
#     try:
#         # Plot output
#         import seaborn as sns
#         from matplotlib import pyplot as plt
#
#         total_iterations = len(old_samples[0])
#         print(total_iterations)
#         burnin = int(total_iterations / 2)
#         print(burnin)
#         samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
#                                   old_samples[3][burnin:, :], old_samples[4][burnin:, :]))
#
#         ndims = len(old_samples[0][0])
#         colors = sns.color_palette(n_colors=ndims)
#         for dim in range(ndims):
#             fig = plt.figure()
#             sns.distplot(samples[:, dim], color=colors[dim])
#             fig.savefig('PyDREAM_example_Robertson_dimension_' + str(dim))
#
#     except ImportError:
#         pass
#
# else:
#     run_kwargs = {'parameters': sampled_parameter_names, 'likelihood': likelihood, 'niterations': 10000,
#                   'nchains': nchains, 'multitry': False, 'gamma_levels': 4, 'adapt_gamma': True, 'history_thin': 1,
#                   'model_name': 'robertson_dreamzs_5chain', 'verbose': True}

# # create sampled_parameter_names
# original_params = np.log10([param.value for param in model.parameters_rules()])
# quit()

# lower_limits = original_params - 3
# parameters_to_sample = SampledParam(uniform, loc=lower_limits, scale=6)
# sampled_parameter_names = [parameters_to_sample]
#
# #create likelihood
#
# tspan = np.linspace(0,40,50)
# solver = Solver(model, tspan)
# solver.run()
# pysb_sampled_parameter_names = [param.name for param in model.parameters_rules()]


# pydream_path = os.path.dirname(inspect.getfile(run_dream))
# location= pydream_path+'/examples/robertson/exp_data/'
# exp_data_ctot = np.loadtxt(location+'exp_data_ctotal.txt')
# # plt.plot(tspan, exp_data_ctot, 'o' )
# # plt.show()
#
# exp_data_sd_ctot = np.loadtxt(location+'exp_data_sd_ctotal.txt')
#
# #Create scipy normal probability distributions for data likelihoods
# like_ctot = norm(loc=exp_data_ctot, scale=exp_data_sd_ctot)
#
# def likelihood(parameter_vector):
#     param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}
#
#     for pname, pvalue in param_dict.items():
#
#         # Change model parameter values to current location in parameter space
#
#         model.parameters[pname].value = 10 ** (pvalue)
#
#     # Simulate experimentally measured Ctotal values.
#
#     solver.run()
#
#     # Calculate log probability contribution from simulated experimental values.
#
#     logp_ctotal = np.sum(like_ctot.logpdf(solver.yobs['C_total']))
#
#     # If model simulation failed due to integrator errors, return a log probability of -inf.
#     if np.isnan(logp_ctotal):
#         logp_ctotal = -np.inf
#
#     return logp_ctotal
#
# niterations = 10000
# #converged = False
# nchains = 1 # nchains = 5

# https://pydream.readthedocs.io/en/latest/pydream.html#pydream-dream-module
#pydream.core.run_dream(parameters, likelihood, nchains=5,niterations=50000,
#start=None, restart=False, verbose=True,nverbose=10, tempering=False, **kwargs)

#sampled_params: sampled parameters for each chain
#log_ps: log proabibility for each sampled point for each chain
#parameter: a list of parameter priors
#likelihood: a user-defined likelihood function
#verbose:whether to print verbose output and the current acceptance rate. Default:True
#niterations: the number of algorithm iterations to run. Default = 50,000
#nchains: the number of parallel DREAM chains to run. Default = 5
#kwargs: other arguments that will be passes to the Dream class on initialization

# sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations= niterations, nchains=nchains,
#                                    multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1,
#                                    model_name='robertson_dreamzs_5chain', verbose=True)
#
#                                     for chain in range(len(sampled_params)):
#                                         np.save('robertson_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total iteration),
#                                                     sampled_params[chain])
#
#                                         old_samples = [np.concatenate((old_samples[chain], sampled_params[chain]) for chain in range( nchains))]
#                                         GR = Gelman_Rubin(old_samples)
#                                         print(' A iteration: ', total_iterations, 'GR=', GR)
#                                         np.savetxt('robertson_dreamzs_5hcain_GelmanRubin_iteration_' +  str(total_iterations) + .txt, GR)

#
# # Import seaborn objects
# from .rcmod import *  # noqa: F401,F403
# from .utils import *  # noqa: F401,F403
# from .palettes import *  # noqa: F401,F403
# from .relational import *  # noqa: F401,F403
# from .regression import *  # noqa: F401,F403
# from .categorical import *  # noqa: F401,F403
# from .distributions import *  # noqa: F401,F403
# from .matrix import *  # noqa: F401,F403
# from .miscplot import *  # noqa: F401,F403
# from .axisgrid import *  # noqa: F401,F403
# from .widgets import *  # noqa: F401,F403
# from .colors import xkcd_rgb, crayons  # noqa: F401
# from . import cm  # noqa: F401
#
# # Capture the original matplotlib rcParams
# import matplotlib as mpl
# _orig_rc_params = mpl.rcParams.copy()
#
# # Define the seaborn version
# __version__ = "0.11.2"

#PUSH ONTO AHPCC AND RUNNING IN BATCH

#
#
#
# # Initialize PysB solver object for running simulations
# # Simulations timespan should match experimental data
#  tspan = np.linspace(0,40)
#  solver = Solver(model, tspan)
#  solver.run()
#
# #Load Experimental data to which Robertson model will be fit there
# # The " experimental data in this case is just the total trajectory
# # At the default model parameters with a standard deviation of 0.01
# pydream_path =  os.path.dirname(inspect.getfile(run_dream))
# location = pydream_path+ '/examples/roberston/exp_data/'
# exp_data_sc_ctot = np.loadtxt(location+ 'exp_data_sd_ctotal.txt')
#
# #Create Scipy normal probability Distributions for data likelihoods
# like_ctot = np.loadtx(location+'exp_data_sd_ctotal.txt')
#
# #Create scipy normal probability distributions for data likelihoods
# like_ctot = norm(loc=exp_data_ctot, scale=exp_data_sd_ctot)
#
# # Create lists of sampled pysb parameter names to use for subbing in
# # parameter values in likelihood function.
# pysb_sampled_parameter_names = [param.name for param in model.parameters_rules()]
#
# #Define likelihood function to generate simulated data that corresponds to
# # experimental time points
# # The function should take as input a parameter vector( parameter values are in the order
# # dictated by first argument to run dream function below).
# # The function returns a log probability value for the parameter vs vector
# # given the experimental data
#
# def likelihood(parameter_vector):
#     param_dic = {pname : pvalue for pname, pvalue in zip(pysb_sampled_parameter_vector, parameter_vector)}
#
#     for pname, pvalue in param_dict.items():
#
#         #Change model parameter values to current location in the parameter space
#         model.parameters[pname].value = 10 **(pvalue)
#
#         #Simulate experimentally measured Ctotal values
#
#         solver.run()
#
#         #Calculate log probability contribution from simulated experimental values
#
#         logp_ctotal = np.sum(like_ctot.logpdf(solver.yobs['C_total']))
#
#         # If model simulation failed due to integrator errors, return
#         # a log probability of -inf.
#         if np.isnan(logp_ctotal):
#             logp_ctotal = -np.inf
#
#         return logp_ctotal
#
#     # Add vector of PysB rate parameters to be sampled as unobserved
#     # random variables to DREAM
#
#     original_parameter_names =np.log10([param.value for param in model.parameter_rules()])
#
#     # Set upper and lower limits for uniform prior to be 3 orders
#     # of magnitude above and below original parameter values.
#
#     lower_limits = original_params - 3
#      parameters_to sample = SampledParam(uniform)uniform, loc=lower_limits, scale=6)
#
#     sampled_parameter_names = [parameters_to_sample]
#     niterations = 10000
#     converged = False
#     total_iterations = niterations
#     nchains = 5
#
# # if __name__ =='__main__':
#
#     #Run DREAM sampling. Documentation of DREAM options is in Dream.py
#     sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations = niterations, nchains, multitry =False, gamma_levels =4, adapt_gamma=True, history_thin=1, model_name='robertson_dreamzs_5chain', verbose=True)

    # # Save sampling output(sampled_parameter values and their corresponding logps).
    # for chain in range(len(sampled_params)):
    #     np.save('robertson_dreamzs_5chain_sampled_params_chain')
    #     np.save('robertson_dreamzs_5chain_logps_chain_'+str(chain)+(total_iterations),log_ps[chain])
    # # check convergence and continue sampling if not converged
    # GR = Gelman_Rubin(sampled_params)
    # print(' At iteration:', total_iterations, 'GR = ',GR)
    # np.savetxt('robertson_dreamzs_5chain_GelmanRubin_iteration_'+ str(total_iterations) + '.txt', GR)
    #
    # old_samples = sampled_params
    # if np.any(GR > 1.2):
    #     starts = [sampled_params[chain][-1,:] for chain in range(nchains)]
    #     while not converged:
    #         total_iterations += niterations
    #
    #         sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, start = starts, niterations=niterations,
    #                                            nchains =nchains, multitry= False, gamma_levels= 4,adapt_gamma = True,
    #                                            history_thin=1, model_name= 'robertson_dreamsz_5hcain', verbose=true, restart=True')
    #
    #         for chain in range(len(sampled_params)):
    #             np.save('robertson_dreamzs_5chain_logps_chain_'+ str(chain) + '_'+ str(total_iterations),
    #                    sampled_params[chain] ))
    #             np.save('robertson_dreamzs_5chain_logps_chain_'+ str(chain) + '_' + str(total_iterations),
    #                         log_ps[chain],)
    #
    #             old_samples = [np.concatenate((old_samples[chain], ))]
    #
    #             ndims = len(old_samples[0][0])
    #             colors = sns.color_palette(n_colors = ndims)
    #             for dim in range(ndims):
    #                 sns.distplot(samples[:, dim,], color=colors[dim])
    #                 fig.savefig('PyDREAM_example_Robertson_dimension_'+ str(dim))
    #
    #             except ImportError:
    #                 pass
    # else:
    # run:
    #     run_kwargs = {'parameters': sampled_parameter_names, 'likelihood':likelihood,'niteratioasn': 10000, 'nchains'} 'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'robertson_dreamzs_5chain', 'verbose':True}
