import numpy as np


class Histogram():
    '''
    Class to storage histograms
    '''
    def __init__ (self,data=[], nbin = 'nada', minlim = 'nada',maxlim = 'nada'):
        '''
        Crea un histograma a partir de una lista (por defecto vacía), 
        con limites inferior y superior, que por defecto seran el 
        maximo y el minimo de los datos y un numero de bines, que 
        por defecto sera la raiz del numero de datos
        '''  
        
        N = len(data)
        if N == 0:
            if minlim=='nada' or maxlim=='nada' or nbin=='nada':
                auxs = 'In order to build an empty histogram it '
                auxs += 'is needed a minlim, maxlim and nbin'
                    
                print auxs
                return
            else:   
                self.minlim = minlim
                self.maxlim = maxlim
                self.nbin = nbin
            
        else:
            if minlim=='nada':
                self.minlim = min(data)
            else:
                self.minlim =minlim
            if maxlim=='nada':
                self.maxlim = max(data)
            else:
                self.maxlim = maxlim 
            if nbin=='nada':
                self.nbin = int(N**0.5)
            else:
                self.nbin = nbin

        self.binsize = float(self.maxlim-self.minlim)/float(self.nbin)
        self.hist = np.array([])
        self.bins = np.array([])
        
        self.Build_hist(data)

    def __add__(self,hist):
        '''
        Suming histograms
        '''  
        
        if not self.minlim==hist.minlim and self.maxlim==hist.maxlim and self.nbin==hist.nbin:
            raise InputError('Histograms no compatible')
        else:
            auxhist = Histogram([],self.minlim, self.maxlim, self.nbin)
            auxhist.hist = self.hist + hist.hist
        return auxhist
        
        
    def Build_hist(self, data):
        '''
        Creating histogram using numpy ones but with bin centers instead
        '''
        auxhist,auxbins = np.histogram(np.array(data),self.nbin,[self.minlim,self.maxlim])
        auxbins = auxbins[:-1] + np.diff(auxbins)/2.
        self.hist = auxhist
        self.bins = auxbins
     
    def Fill_hist(self, data):
        '''
        Filling histogram
        '''
        data = np.array(data)
        self.hist += np.histogram(np.array(data),self.nbin,[self.minlim,self.maxlim])[0]
    
    
    
    def Scale(self,factor):
        '''
        Explicit __rmul__, scales the histogram by a factor
        '''
        self.hist *= factor
