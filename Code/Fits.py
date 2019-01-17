from Histogram import Histogram as hist
from PDF import PDF
import fitFunctions as fitf
import scipy.optimize as sop
import scipy.stats as sps
import numpy as np
from scipy.special import gammaln
from copy import deepcopy


def generalLogPoisson(x, mu):
    '''
    Returns the natural logarithm of the probability value of "x"
    for a Poisson distribution with mean "mu"
    '''
    return (-mu+x*np.log(mu+0.00001)-gammaln(x+1))


class Fit():
    '''
    Class meant to perform the fit
    '''
    def __init__(self, x, spectra, PDFs, nevs0):
        '''
        x       : dictiorary of x range (np array) for each dimension in fit 
        spectrum: dictionary with a np array for each variable for each dimension 
                  experimental points
        PDF     : dictionary with list of spectra PDFs for each dimension
        nevs    : numpy array with normalizations for the spectra (initial values
                  for the fit) for each dimension
        '''
        self.x         = deepcopy(x)
        self.dimension = list(self.x.keys())
        self.spectrum  = {}
        self.PDFs      = deepcopy(PDFs)
        self.PDF_Val   = {}
        self.nevs0     = deepcopy(nevs0.reshape(len(nevs0), 1))
        for dim in self.dimension: 
            self.spectrum[dim] = spectra[dim].hist[:]
            pdf_val = []
            for pdfi in self.PDFs[dim]:
                aux            = np.array(pdfi.pdf(self.x[dim]))
                aux[aux<1e-13] = 0
                pdf_val.append(aux)
                
            self.PDF_Val[dim] = np.array(pdf_val)
                
    def LLh(self, ratio):
        '''
        function meant to compute the LogLikelihood
        '''
        ratio = ratio.reshape(len(ratio), 1)
        lm    = 0
        for dim in self.dimension:
            ypdf = np.sum(ratio*self.nevs0*self.PDF_Val[dim], axis=0)
            ydat = self.spectrum[dim]
            lm  += (np.array(generalLogPoisson(ydat, ypdf))).sum()            
        return -2*lm
    def FitLLM(self, ratio, **kwargs):
        '''
        function meant to perform the LogLikelihood fit to data using ratio as
        the initial values
        '''
        ratio = ratio.reshape(len(np.array(ratio)), 1)
        fit   = self.LLh
        res   = sop.minimize(fit, ratio, method='TNC', **kwargs)
        chi2  = -1
        err   = -1
        if (res.success):
            chi2 = self.GetChi2 (res.x)
            err  = self.GetError(res.x)
        res.chi2 = chi2
        res.err  = err
        return res

    def FitLLMScan(self, nevs, fixn, npoint=100, **kwargs):
        '''
        function meant to return the LogLikelihood profile to data using:
            nevs as the initial values
            fixn as the index of the variable along the profile is made
            npoint as the number of points computed from 0 to 2 times the
                minimim value
        '''
        nevs = nevs.reshape(len(np.array(nevs)), 1)
        aux = nevs[fixn]
        aux_evs = np.delete(nevs, fixn)
        fun_aux = self.FitLLM(nevs, **kwargs).fun
        res_list = np.zeros(npoint)
        i = 0
        for aux_s in np.linspace(0, 2*aux, npoint):
            fit = lambda x_nevs: self.LLh(np.insert(x_nevs, fixn, aux_s))
            res = sop.minimize(fit, aux_evs, method='SLSQP', **kwargs)
            res_list[i] = res.fun-fun_aux
            i += 1
            if not(res.success):
                print('error')
        return np.linspace(0, 2*aux, npoint), res_list

    def GetChi2(self, nevsmin):
        '''
        function meant to compute the chi2 nevsmin as the minimum value (fit result)
        '''
        chi2    = np.array([0, 0], dtype=np.float64)
        nevsmin = nevsmin.reshape(len(nevsmin), 1)
       
        for dim in self.dimension:
            ypdf  = np.sum(nevsmin * self.nevs0 * self.PDF_Val[dim], axis=0)
            ydat  = self.spectrum[dim]
            chi2 += np.array(fitf.get_chi2_and_pvalue(ydat, ypdf, len(nevsmin)), dtype=np.float64)
        return chi2
    
    def GetError(self, nevsmin, **kwargs):
        '''
        function meant to compute the LogLikelihood fit errors using nevsmin as
        the minimum value (fit result)
        '''
        nevs = nevsmin.reshape(len(np.array(nevsmin)), 1)
        res_list = np.zeros(len(nevsmin))
        ll_min = self.LLh(nevsmin)
        for fixn in np.arange(len(nevsmin)):
            aux = nevs[fixn]
            aux_evs = np.delete(nevs, fixn)
            fit = lambda aux_s: (lambda x_nevs: self.LLh(np.insert(x_nevs,
                                                                   fixn,
                                                                   aux_s)))
            res = lambda aux_ss: (sop.minimize(fit(aux_ss), aux_evs,
                                               method='SLSQP',
                                               **kwargs)).fun-1-ll_min
            res_list[fixn] = (sop.root(res, aux-aux**0.5).x)[0]
        return nevsmin-res_list

    def GetSpectra(self, E, nevs):
        '''
        Returns y values for an energy range (np array) and a
        given nevs configuration.
        '''
        nevs = np.array(nevs)
        nevs = nevs.reshape(len(nevs), 1)
        ypdf = np.sum(nevs*self.PDF_Val, axis=0)
        return ypdf

    def GetSpectraSQ(self, E, *nevs):
        '''
        It is the same as GetSpectra but tunned to be used in less
        squares method.
        Returns y values for an energy range (np array) and a
        given nevs configuration.
        '''
        nevs = np.array(nevs)
        nevs = nevs.reshape(len(nevs), 1)
        ypdf = np.sum(nevs*self.PDF_Val, axis=0)
        return ypdf

    def FitLeastSQ(self, nevs, **kwargs):
        '''
        function meant to perform the less squares fit to data using nevs as
        the initial values
        '''
        nevs = np.array(nevs)
        nevs = nevs.reshape(1,len(nevs))[0]
        fit = self.GetSpectraSQ
        sigma = self.E**0.5
        sigma[sigma<1] = 1
        res = sop.curve_fit(fit, self.E, self.spectrum, p0=nevs, sigma= sigma, absolute_sigma=True, **kwargs)
        return res
