from Histogram import Histogram as hist
from PDF import PDF
import scipy.optimize as sop
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
    def __init__(self, E, spectrum, PDFs):
        '''
        E: x range (np array) spectrum: experimental points
        PDF: list of spectra PDFs
        nevs: normalizations for the spectra (they are the initial
              values for the fit)
        '''
        self.E = E[:]
        self.spectrum = spectrum.hist[:]
        self.PDFs = deepcopy(PDFs)
        self.PDF_Val = np.array([np.array(pdfi.pdf(E)) for pdfi in self.PDFs])

    def LLh(self, nevs):
        '''
        function meant to compute the LogLikelihood
        '''
        nevs = nevs.reshape(len(nevs), 1)
        ypdf = np.sum(nevs*self.PDF_Val, axis=0)
        ydat = self.spectrum
        lm = (np.array(generalLogPoisson(ydat, ypdf))).sum()
        return -2*lm

    def FitLLM(self, nevs, **kwargs):
        '''
        function meant to perform the LogLikelihood fit to data using nevs as
        the initial values
        '''
        nevs = nevs.reshape(len(np.array(nevs)), 1)
        fit = self.LLh
        res = sop.minimize(fit, nevs, method='Nelder-Mead', **kwargs)
        ypdf = np.sum(nevs*self.PDF_Val, axis=0)
        ydat = self.spectrum
        chi2 = -1
        err = -1
        if (res.success):
            chi2 = np.sum((ypdf-ydat)**2/((ydat+0.0001)*(len(ypdf)-len(nevs))))
            err = self.GetError(res.x)
        res.chi2 = chi2
        res.err = err
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
            res = sop.minimize(fit, aux_evs, method='Nelder-Mead', **kwargs)
            res_list[i] = res.fun-fun_aux
            i += 1
            if not(res.success):
                print('error')
        return np.linspace(0, 2*aux, npoint), res_list

    def GetError(self, nevsmin, **kwargs):
        '''
        function meant to compite the LogLikelihood fit errors using nevsmin as
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
                                               method='Nelder-Mead',
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
        nevs = nevs.reshape(len(nevs), 1)
        fit = self.GetSpectraSQ
        res = sop.curve_fit(fit, self.E, self.spectrum, nevs)
        return res
