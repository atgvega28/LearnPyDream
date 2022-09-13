from matplotlib import pyplot as plt
from pydream.core import run_dream
from pysb.integrate import Solver
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
import os
import inspect
from pydream.convergence import Gelman_Rubin

from pysb.examples.robertson import model

# create sampled_parameter_names
original_params = np.log10([param.value for param in model.parameters_rules()])
lower_limits = original_params - 3
parameters_to_sample = SampledParam(uniform, loc=lower_limits, scale=6)
sampled_parameter_names = [parameters_to_sample]


#create likelihood

tspan = np.linspace(0,40,50)

solver = Solver(model, tspan)

pysb_sampled_parameter_names = [param.name for param in model.parameters_rules()]


pydream_path = os.path.dirname(inspect.getfile(run_dream))
location= pydream_path+'/examples/robertson/exp_data/'
exp_data_ctot = np.loadtxt(location+'exp_data_ctotal.txt')
# plt.plot(tspan, exp_data_ctot, 'o' )
# plt.show()

exp_data_sd_ctot = np.loadtxt(location+'exp_data_sd_ctotal.txt')

#Create scipy normal probability distributions for data likelihoods
like_ctot = norm(loc=exp_data_ctot, scale=exp_data_sd_ctot)

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

niterations = 10000

nchains = 1

sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations= niterations, nchains=nchains,
                                   multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1,
                                   model_name='robertson_dreamzs_5chain', verbose=True)
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
# # if __name__ =='__main__':(No needed only for command line)
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
