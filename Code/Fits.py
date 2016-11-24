import Estimation as st
from Histogram import Histogram as hist
from PDF import PDF
from Utils import IsotopeDic, PartDic
from scipy.stats import poisson
import scipy.optimize as sop
import numpy as np
from scipy.special import gamma

def generalPoisson(x,mu):
    return np.exp(-mu)*np.power(mu,x)/gamma(x+1)


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
        self.E = E
        self.spectrum = spectrum
        self.PDFs = [pdf.Scale(1./pdf.Int) for pdf in PDFs]

    def LogLikelihood(self, nevs):
        '''
        function meant to compute the LogLikelihood
        '''
        print(self.PDFs,nevs)
        ypdf = np.array([sum([n*pdfi.pdf(Ei) for pdfi,n in zip(self.PDFs,nevs)]) for Ei in self.E])
        ydat = np.array(self.spectrum)
        lm = (np.array(generalPoisson(ydat,ypdf)))
        lm = np.log(np.abs(lm)).sum()
        plt.plot(self.E,ypdf)
        plt.plot(self.E,ydat,'+')

        return -lm

    def FitLLM(self,nevs):
        fit = self.LogLikelihood
        res = sop.minimize(fit,nevs)
        ypdf = np.array([sum([n*pdfi.pdf(Ei) for pdfi,n in zip(self.PDFs,res.x)]) for Ei in self.E])
        ydat = np.array(self.spectrum)
        chi2 = -1
        if (res.success):
            chi2 = np.sum((ypdf-ydat)**2)/(1.*(len(ypdf)-len(nevs)))
        res.chi2 = chi2
        return res
