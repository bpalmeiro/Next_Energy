import Estimation as st
from Histogram import Histogram as hist
from PDF import PDF
from Utils import IsotopeDic, PartDic
from scipy.stats import poisson
import matplotlib.pylab as plt
import scipy.optimize as sop
import numpy as np
from scipy.special import gammaln
from copy import copy

def generalLogPoisson(x,mu):
    #Çmu[mu<-2e-11] = np.nan
    '''for i in mu:
        if i<-2e-11:

       print(i)
    '''
    return (-mu+x*np.log(mu+0.00001)-gammaln(x+1))

class Fit():
    '''
    Class meant to perform the fit
    '''
    def __init__(self, E, spectrum, PDFs):
        '''
        E: x range (np array)
        spectrum: experimental points
        PDF: list of spectra PDFs
        nevs: normalizations for the spectra (they are the
            initial values for the fit)
        '''
        self.E = E[:]
        self.spectrum = spectrum.hist[:]
        self.PDFs = copy(PDFs)
        self.PDF_Val = np.array([np.array(pdfi.pdf(E)) for pdfi in self.PDFs])

    def LogLikelihood(self, nevs):
        '''
        function meant to compute the LogLikelihood
        '''
        nevs = nevs.reshape(len(nevs),1)
        ypdf = np.sum(nevs*self.PDF_Val,axis=0)
        ydat = self.spectrum
        lm = (np.array(generalLogPoisson(ydat,ypdf))).sum()
        return -2*lm



    def FitLLM(self,nevs,bounds=None, **kwargs):
        nevs = nevs.reshape(len(np.array(nevs)),1)
        fit = self.LogLikelihood
        res = sop.minimize(fit,nevs,method='Nelder-Mead',**kwargs)
        ypdf = np.sum(nevs*self.PDF_Val,axis=0)
        ydat = self.spectrum
        chi2 = -1
        err = -1
        if (res.success):
            chi2 = np.sum((ypdf-ydat)**2/((ydat+0.0001)*(len(ypdf)-len(nevs))))
            err = self.GetError(res.x)
        res.chi2 = chi2
        res.err = err
        return res

    def FitLLMScan(self,nevs, fixn, npoint=100, **kwargs):
        nevs = nevs.reshape(len(np.array(nevs)),1)
        aux = nevs[fixn]
        aux_evs = np.delete(nevs,fixn)
        fun_aux = self.FitLLM(nevs, **kwargs).fun
        res_list = []
        for aux_s in np.linspace(0,2*aux,npoint):
            fit = lambda x_nevs: self.LogLikelihood(np.insert(x_nevs,fixn,aux_s))
            #res = sop.minimize(fit,nevs,method='L-BFGS-B',bounds = bounds,**kwargs)
            res = sop.minimize(fit,aux_evs,method='Nelder-Mead',**kwargs)
            res_list.append(res.fun-fun_aux)
            #print(aux_s,res)
            if not(res.success):
                print('error')
        return np.linspace(0,2*aux,npoint),res_list

    def GetError(self, nevsmin, **kwargs):
        error = []
        nevs = nevsmin.reshape(len(np.array(nevsmin)),1)
        res_list = np.zeros(len(nevsmin))
        for fixn in range(len(nevsmin)):

            aux = nevs[fixn]
            aux_evs = np.delete(nevs,fixn)
            fit = lambda aux_s:  (lambda x_nevs: self.LogLikelihood(np.insert(x_nevs,fixn,aux_s)) )
            #res = sop.minimize(fit,nevs,method='L-BFGS-B',bounds = bounds,**kwargs)
            res = lambda aux_ss: (sop.minimize(fit(aux_ss),aux_evs,method='Nelder-Mead',**kwargs)).fun-1-715.134906606767
            res_list[fixn] = (sop.root(res,aux-aux**0.5).x)[0]
        return nevsmin-res_list


    def GetSpectra(self,E,*nevs):
        ypdf = np.array([sum([n*pdfi.pdf(Ei) for pdfi,n in zip(self.PDFs,nevs)]) for Ei in self.E])
        #print ypdf
        return ypdf

    def FitLeastSQ(self,nevs,**kwargs):
        nevs = np.array(nevs)
        fit = self.GetSpectra
        res = sop.curve_fit(fit,self.E,self.spectrum, nevs)
        return res
