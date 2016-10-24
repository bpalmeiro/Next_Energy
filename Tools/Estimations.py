import numpy as np


class BkgExpectation(trun=1., BckRej=1.):
    '''
    Class meant to yield an dictionry with the expected number 
    of events for backgrounds for all the isotopes.
    
    Input values:
    
    trun is meant to be in years
    
    BckRej is the fraction of Bck events rejected by topological
    cuts (it is a toy cut). It goes from 0 (perfect rejection)
    to 1 (no rejection)
    '''

    self.trun = trun #y
    self.texp = trun * 24 * 3600 #s
    self.BckRej = BckRej
    
    self.Expected = {60 : {}, 40 : {} , 214 : {} , 208: {} }
    self.BuildDic()

    def BuildDic(self)
    '''
    It builds the dictionary
    '''
    texp = self.texp
    BgrRej = self.BgrRej
    self.Expected = {
        60: {0 :  int(round(3.935333e-2 * 0       /1000.   *texp * BgrRej)),
             1 :  int(round(1.781900e-2 * 2.32e-1 /1000.   *texp * BgrRej)),
             2 :  int(round(2.663240e-3 * 8.82    /1000.   *texp * BgrRej)),
             3 :  int(round(6.060000e-5 * 8.4e-1  /1000.   *texp * BgrRej)),
             4 :  int(round(3.945800e-2 * 2.27e-1 /1000.   *texp * BgrRej)),
             5 :  int(round(4.147233e-2 * 9.66e-1 /1000.   *texp * BgrRej)),
             6 :  int(round(1.525725e-3 * 2.02    /1000.   *texp * BgrRej)),
             7 :  int(round(1.616075e-2 * 0       /1000.   *texp * BgrRej)),
             8 :  int(round(8.496444e-3 * 2.52e1  /1000.   *texp * BgrRej)),
             9 :  int(round(1.553625e-2 * 1.16e-1 /1000.   *texp * BgrRej)),
             10 : int(round(9.619500e-5 * 1.58e3  /1000.   *texp * BgrRej)),
             11 : int(round(3.361000e-3 * 2.03e-1 /1000.   *texp * BgrRej)),
             12 : int(round(6.919778e-3 * 4.56e1  /1000.   *texp * BgrRej)),
             13 : int(round(1.182263e-5 * 1.25e3  /1000.   *texp * BgrRej)),
             14 : int(round(3.444875e-6 * 1.00e2  /1000.   *texp * BgrRej)),
             15 : int(round(5.184667e-3 * 1.24e1  /1000.   *texp * BgrRej)),
             16 : int(round(4.619750e-4 * 2.84e3  /1000.   *texp * BgrRej))
             },
        40: {0 :  int(round(4.262175e-3 * 1.03    /1000.   *texp * BgrRej)),
             1 :  int(round(1.038175e-3 * 1.38e1  /1000.   *texp * BgrRej)),
             2 :  int(round(1.676425e-4 * 1.33e1  /1000.   *texp * BgrRej)),
             3 :  int(round(4.483846e-6 * 9.52e1  /1000.   *texp * BgrRej)),
             4 :  int(round(2.201075e-3 * 4.07e2  /1000.   *texp * BgrRej)),
             5 :  int(round(2.358350e-3 * 5.79e1  /1000.   *texp * BgrRej)),
             6 :  int(round(9.838308e-5 * 3.05    /1000.   *texp * BgrRej)),
             7 :  int(round(9.100154e-4 * 3.4e-1  /1000.   *texp * BgrRej)),
             8 :  int(round(5.244000e-4 * 3.81e1  /1000.   *texp * BgrRej)),
             9 :  int(round(8.711692e-4 * 4.44    /1000.   *texp * BgrRej)),
             10 : int(round(0           * 5.76e1  /1000.   *texp * BgrRej)),
             11 : int(round(2.022533e-4 * 2.55e1  /1000.   *texp * BgrRej)),
             12 : int(round(4.074067e-4 * 1.45e2  /1000.   *texp * BgrRej)),
             13 : int(round(0           * 1.87e3  /1000.   *texp * BgrRej)),
             14 : int(round(0           * 3.7e5   /1000.   *texp * BgrRej)),
             15 : int(round(3.162600e-4 * 1.88e1  /1000.   *texp * BgrRej)),
             16 : int(round(3.100538e-5 * 1.03e2  /1000.   *texp * BgrRej))
             },
        214:{0 :  int(round(3.188567e-2 * 3.34e-1 /1000.   *texp * BgrRej)),
             1 :  int(round(9.943231e-3 * 2.05e-1 /1000.   *texp * BgrRej)),
             2 :  int(round(1.409567e-3 * 2.58    /1000.   *texp * BgrRej)),
             3 :  int(round(3.629250e-5 * 1.79e2  /1000.   *texp * BgrRej)),
             4 :  int(round(2.299831e-2 * 2.12    /1000.   *texp * BgrRej)),
             5 :  int(round(2.362262e-2 * 1.04    /1000.   *texp * BgrRej)),
             6 :  int(round(8.227000e-4 * 5.9e-1  /1000.   *texp * BgrRej)),
             7 :  int(round(9.060444e-3 * 5.05e-1 /1000.   *texp * BgrRej)),
             8 :  int(round(4.535320e-3 * 7.38    /1000.   *texp * BgrRej)),
             9 :  int(round(8.353222e-3 * 5.65e-1 /1000.   *texp * BgrRej)),
             10 : int(round(6.012250e-5 * 1.66e2  /1000.   *texp * BgrRej)),
             11 : int(round(1.818040e-3 * 7.03    /1000.   *texp * BgrRej)),
             12 : int(round(3.713360e-3 * 4.2     /1000.   *texp * BgrRej)),
             13 : int(round(7.966818e-6 * 5.45e3  /1000.   *texp * BgrRej)),
             14 : int(round(2.331544e-6 * 1.05e5  /1000.   *texp * BgrRej)),
             15 : int(round(2.742629e-3 * 3.64    /1000.   *texp * BgrRej)),
             16 : int(round(2.630100e-4 * 2.97e2  /1000.   *texp * BgrRej))
             },
        208:{0 :  int(round(4.405350e-2 * 5.41e-2 /1000.   *texp * BgrRej)),
             1 :  int(round(1.300250e-2 * 2.52e-2 /1000.   *texp * BgrRej)),
             2 :  int(round(2.493200e-3 * 3.23e-1 /1000.   *texp * BgrRej)),
             3 :  int(round(1.078600e-4 * 5.6e1   /1000.   *texp * BgrRej)),
             4 :  int(round(3.341017e-2 * 3.3e-1  /1000.   *texp * BgrRej)),
             5 :  int(round(3.065883e-2 * 1.72e-1 /1000.   *texp * BgrRej)),
             6 :  int(round(1.416540e-3 * 7.13e-2 /1000.   *texp * BgrRej)),
             7 :  int(round(1.044113e-2 * 7.13e-2 /1000.   *texp * BgrRej)),
             9 :  int(round(7.782667e-4 * 9.23e-1 /1000.   *texp * BgrRej)),
             8 :  int(round(1.053888e-2 * 1.67e-1 /1000.   *texp * BgrRej)),
             10 : int(round(1.768800e-4 * 5.4e1   /1000.   *texp * BgrRej)),
             11 : int(round(2.669667e-3 * 2.3     /1000.   *texp * BgrRej)),
             12 : int(round(5.219333e-3 * 2.28    /1000.   *texp * BgrRej)),
             13 : int(round(2.816600e-5 * 5.30e2  /1000.   *texp * BgrRej)),
             14 : int(round(8.416333e-6 * 9.4e3   /1000.   *texp * BgrRej)),
             15 : int(round(4.837350e-3 * 4.55e-1 /1000.   *texp * BgrRej)),
             16 : int(round(7.067467e-4 * 9.68e1  /1000.   *texp * BgrRej))
             }
        }
    def GetValue(isotope, part='all', partlist=[]):
        '''
        Returns the expected value for the parte selected. If all
        is selected, returns all the events expected from every single part
        for the selected isotope. If multiple is selected, it sums all the 
        contributions from the parts in partlist
        '''

        if not part=='all':
            if not part=='multiple'
                return self.Expected[isotope][part]
            else:
                if not len(partlist)>0:
                    raise ValueError('Multiple selection requires a part list')
                aux = 0
                for i in partlist:
                    aux += self.Expected[isotope][i]
        else:
            return sum(Expected[isotope].values())
    
    def ReBuild(trun=1., BckRej=1.):
        '''
        Rebuilds the dictionary with a new trun and BckRej
        '''        
        self.trun = trun #y
        self.texp = trun * 24 * 3600 #s
        self.BckRej = BckRej

        self.Expected = {60 : {}, 40 : {} , 214 : {} , 208: {} }
        self.BuildDic()
        
    def __getitem__(self, isotope):
        '''
        Defines behavior for when an item is accessed, using the notation self[key]
        '''
        return self.Expectation[isotope]
