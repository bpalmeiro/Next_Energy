import numpy as np
import scipy as sp

class PDF():
    """
    docstring for PDF
    It takes the samplename, the isotope, the volume, matherial where the bkg
    comes form, the file name where to look for the pdf, the key name in the TTRee
    
    """

    def __init__(self, exposure = None , low_edge = None , high_edge = None):
        self.nsamples = 0
        self.names = []
        self.histos = []
        self.low_edge = low_edge
        self.high_edge = high_edge
        self.exposure = exposure
        self.name = " "
        
    def Print(self):
        print "PDF for the spectrum of:"
        print self.name
        
    def ImportHistogram
    
    
    
    
    #samplename, isotope, volume, material, filename, keyname,                exposure,energy=None,lower_edge=None,higher_edge=None,gauss=False, rlim = 0.01)
