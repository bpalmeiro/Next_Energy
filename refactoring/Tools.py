"""
Created on Mon Mar 10 18:24:03 2017

@author: Brais Palmeiro
"""

import scipy.optimize as sop
import numpy as np
from scipy.special import gammaln
from copy import deepcopy
import scipy.stats as sps


def generalLogPoisson(x, mu):
    '''
    Returns the natural logarithm of the probability value of "x"
    for a Poisson distribution with mean "mu"
    '''
    #TODO cambiar por la gamma stadistic function
    #Cambiar el nombre
    return (-mu+x*np.log(mu+0.00000001)-gammaln(x+1))

def generate_LLh_fun(func, ydat):
    '''
    function meant to compute the LogLikelihood.
    Inputs:
        func: function to compute the Likelihood from
        ydat: data to compute the Likelihood from
    Returns:
        Likelihood function where the arguments are only the input function
        ones
    '''
    lm = lambda pars: -2*(np.array(generalLogPoisson(ydat, func(pars)))).sum()
    return lm

#def Get_Error(LLh, pars_best):
#    '''
#    kpos = 0
#    def LL_global(fixed):
#        return LL_local(fixed) - 1 - LL_min
#    def LL_local(fixed):
#        ffit = ll_reduced(fixed)
#        result  =  sop.minimize(ffit, aux_evs, method='Nelder-Mead', kwargs)
#        return result.fun
#    def LL_reduced(fixed):
#        def func(others):
#            pars = np.insert(others, kpos, fixed)
#            return LLh(pars)
#        return func
#
#    for ipos in range(len(pars_best)):
#        kpos = ipos
#        fixed = par_best[ipos]
#        sop.root(LL_global, fixed-np.sqrt(fixed ))
#    '''
##    return 0


def generate_LLh_scan(i, LLh, pars_best, **kwargs):
    '''
    Returns a LL scan function for i parameter
    Input:
        i: parameter index
        LLh: LL function
        pars_best: np array with the parameters that minimize LLh
    Output:
        Scan function for i parameter
    '''

    def LL_local(n_fixed):
        '''
        Returns LLh value for a given scanned parameter with the best estimation
        for the others
        Input:
            n_fixed: scanned parameter
        Output:
            LL value in n_fixed
        '''

        n_reduced = np.delete(pars_best, i)
        func      = lambda n_var: LLh(np.insert(n_var, i, n_fixed))
        result    =  sop.minimize(func, n_reduced,
                                  method='Nelder-Mead', **kwargs)
        return result.fun
    return LL_local


def confidence_interval(LLh, pars_best, sigma_value=1., **kwargs):
    '''
    Returns the confidence interval for a given LLh minimization
    Input:
        LLh: LL function
        pars_best: np array with the parameters that minimize LLh
        sigma_value: number of sigmas for de coverage interval
        (set 1 by default)
    Output:
        Confidence interval for all parameters in two np arrays
    '''

    LL_min = LLh(pars_best)
    root_list_L = np.zeros_like(pars_best)
    root_list_H = np.zeros_like(pars_best)
    for i in range(len(pars_best)):
        n_fixed      = pars_best[i]
        root_guess_L = n_fixed-np.sqrt(n_fixed)
        root_guess_H = n_fixed+np.sqrt(n_fixed)

        LLh_scan     = generate_LLh_scan(i, LLh, pars_best, **kwargs)

        error_root   = lambda n: LLh_scan(n)-sigma_value-LL_min
        root_L       = sop.root(error_root, root_guess_L)
        root_H       = sop.root(error_root, root_guess_H)

        root_list_L[i] = float(root_L.x[0])
        root_list_H[i] = root_H.x[0]

    return np.array([root_list_L, root_list_H])


def minimize_LLh(func, ydat, pars_0, conf_interval=True, sigma_value = 1.,
                 **kwargs):
    '''
    Function meant to perform the LogLikelihood minimization to data
    using n_0 as the initial values.
    Input:
        func: function to minimize. For a given set of parameters (func inputs)
        it must return a np array with y values for the whole range to
        compare with ydat array.
        ydat: array with 'experimental' y
        pars_0: initial parameter guess
        conf_interval: flag to compute or not the confidence interval (True by
        defalut)
        sigma_value: number of sigmas for de coverage interval
        (set 1 by default)
        kwargs: kwargs for scipy.optimize.minimize
    '''

    fit  = generate_LLh_fun(func, ydat)
    res  = sop.minimize(fit, pars_0, method='Nelder-Mead', **kwargs)
    yfit = func(res.x)
    chi2 = -1
    conf_int  = -1
    if (res.success):
        n_par = len(pars_0)
        chi2  = list(sps.chisquare(ydat, yfit, n_par))
        chi2[0] *= 1./(len(ydat)-len(pars_0))
        if conf_interval:
            conf_int  = confidence_interval(fit, res.x,
                                            sigma_value, **kwargs)[0]
    res.chi2 = chi2
    res.conf_interval  = conf_int
    return res


#def fit_LLh(hist_data, hist_model_list, )