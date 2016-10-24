import numpy as np


def XeEstimation(mass=5.81, purity=0.91, trun=100, SigAcc=1.):
    '''
    Fuction meant to yield the expected number of events for 136Xe

    Input values:

    mass: Xe (all isotopes) mass in kg
    putiry
    trun is meant to be in years

    SigAcc is the fraction of events accepted by topological
    cuts (it is a toy cut). It goes from 0 (no event accepted)
    to 1 (all the events accepted)
    '''

    Nav = 6.022140857e23  # mol^-1
    Mmol = 135.907214484  # gmol
    ln2 = np.log(2)

    trun *= 1./365.24

    t12 = 2.165e21
    et12 = 0.061e21

    lamb = ln2/t12
    slamb = ln2*et12/t12**2

    N0 = Nav*mass*1000*purity/Mmol

    Nfin = N0*lamb*trun*SigAcc

    return Nfin


def BkgExpectation(trun=100., BkgRej=1.):
    '''
    Fuction meant to yield an dictionry with the expected number
    of events for backgrounds for all the isotopes.

    Input values:

    trun is meant to be in days

    BckRej is the fraction of Bck events rejected by topological
    cuts (it is a toy cut). It goes from 0 (perfect rejection)
    to 1 (no rejection)
    '''

    texp = trun * 24 * 3600  # s
    Expected = {
        60: {0 :  3.935333e-2 * 0       / 1000. * texp * BkgRej,
             1 :  1.781900e-2 * 2.32e-1 / 1000. * texp * BkgRej,
             2 :  2.663240e-3 * 8.82    / 1000. * texp * BkgRej,
             3 :  6.060000e-5 * 8.4e-1  / 1000. * texp * BkgRej,
             4 :  3.945800e-2 * 2.27e-1 / 1000. * texp * BkgRej,
             5 :  4.147233e-2 * 9.66e-1 / 1000. * texp * BkgRej,
             6 :  1.525725e-3 * 2.02    / 1000. * texp * BkgRej,
             7 :  1.616075e-2 * 0       / 1000. * texp * BkgRej,
             8 :  8.496444e-3 * 2.52e1  / 1000. * texp * BkgRej,
             9 :  1.553625e-2 * 1.16e-1 / 1000. * texp * BkgRej,
             10 : 9.619500e-5 * 1.58e3  / 1000. * texp * BkgRej,
             11 : 3.361000e-3 * 2.03e-1 / 1000. * texp * BkgRej,
             12 : 6.919778e-3 * 4.56e1  / 1000. * texp * BkgRej,
             13 : 1.182263e-5 * 1.25e3  / 1000. * texp * BkgRej,
             14 : 3.444875e-6 * 8.19e2  / 1000. * texp * BkgRej,
             15 : 5.184667e-3 * 1.24e1  / 1000. * texp * BkgRej,
             16 : 4.619750e-4 * 2.84e3  / 1000. * texp * BkgRej
             },
        40: {0 :  4.262175e-3 * 1.03    / 1000. * texp * BkgRej,
             1 :  1.038175e-3 * 1.38e1  / 1000. * texp * BkgRej,
             2 :  1.676425e-4 * 1.33e1  / 1000. * texp * BkgRej,
             3 :  4.483846e-6 * 9.52e1  / 1000. * texp * BkgRej,
             4 :  2.201075e-3 * 4.07e2  / 1000. * texp * BkgRej,
             5 :  2.358350e-3 * 5.82e1  / 1000. * texp * BkgRej,
             6 :  9.838308e-5 * 3.05    / 1000. * texp * BkgRej,
             7 :  9.100154e-4 * 3.4e-1  / 1000. * texp * BkgRej,
             8 :  5.244000e-4 * 3.81e1  / 1000. * texp * BkgRej,
             9 :  8.711692e-4 * 4.44    / 1000. * texp * BkgRej,
             10 : 7.260000e-6 * 5.76e1  / 1000. * texp * BkgRej,
             11 : 2.022533e-4 * 2.55e1  / 1000. * texp * BkgRej,
             12 : 4.074067e-4 * 1.45e2  / 1000. * texp * BkgRej,
             13 : 9.930000e-7 * 1.87e3  / 1000. * texp * BkgRej,
             14 : 2.870000e-7 * 2.41e5  / 1000. * texp * BkgRej,
             15 : 3.162600e-4 * 1.88e1  / 1000. * texp * BkgRej,
             16 : 3.100538e-5 * 1.03e2  / 1000. * texp * BkgRej
             },
        214:{0 :  3.188567e-2 * 3.34e-1 / 1000. * texp * BkgRej,
             1 :  9.943231e-3 * 2.05e-1 / 1000. * texp * BkgRej,
             2 :  1.409567e-3 * 2.58    / 1000. * texp * BkgRej,
             3 :  3.629250e-5 * 1.79e2  / 1000. * texp * BkgRej,
             4 :  2.299831e-2 * 2.12    / 1000. * texp * BkgRej,
             5 :  2.362262e-2 * 1.05    / 1000. * texp * BkgRej,
             6 :  8.227000e-4 * 5.9e-1  / 1000. * texp * BkgRej,
             7 :  9.060444e-3 * 5.02e-1 / 1000. * texp * BkgRej,
             8 :  4.535320e-3 * 7.38    / 1000. * texp * BkgRej,
             9 :  8.353222e-3 * 5.65e-1 / 1000. * texp * BkgRej,
             10 : 6.012250e-5 * 1.66e2  / 1000. * texp * BkgRej,
             11 : 1.818040e-3 * 7.03    / 1000. * texp * BkgRej,
             12 : 3.713360e-3 * 4.2     / 1000. * texp * BkgRej,
             13 : 7.966818e-6 * 5.46e3  / 1000. * texp * BkgRej,
             14 : 2.331544e-6 * 6.83e4  / 1000. * texp * BkgRej,
             15 : 2.742629e-3 * 3.64    / 1000. * texp * BkgRej,
             16 : 2.630100e-4 * 2.97e2  / 1000. * texp * BkgRej
             },
        208:{0 :  4.405350e-2 * 5.41e-2 / 1000. * texp * BkgRej,
             1 :  1.300250e-2 * 2.52e-2 / 1000. * texp * BkgRej,
             2 :  2.493200e-3 * 3.23e-1 / 1000. * texp * BkgRej,
             3 :  1.078600e-4 * 5.6e1   / 1000. * texp * BkgRej,
             4 :  3.341017e-2 * 3.3e-1  / 1000. * texp * BkgRej,
             5 :  3.065883e-2 * 1.72e-1 / 1000. * texp * BkgRej,
             6 :  1.416540e-3 * 7.38e-2 / 1000. * texp * BkgRej,
             7 :  1.044113e-2 * 7.13e-2 / 1000. * texp * BkgRej,
             8 :  7.782667e-4 * 9.23e-1 / 1000. * texp * BkgRej,
             9 :  1.053888e-2 * 1.67e-1 / 1000. * texp * BkgRej,
             10 : 1.768800e-4 * 5.4e1   / 1000. * texp * BkgRej,
             11 : 2.669667e-3 * 1.9     / 1000. * texp * BkgRej,
             12 : 5.219333e-3 * 2.28    / 1000. * texp * BkgRej,
             13 : 2.816600e-5 * 5.30e2  / 1000. * texp * BkgRej,
             14 : 8.416333e-6 * 6.77e3  / 1000. * texp * BkgRej,
             15 : 4.837350e-3 * 4.55e-1 / 1000. * texp * BkgRej,
             16 : 7.067467e-4 * 9.68e1  / 1000. * texp * BkgRej,
             }
        }
    return Expected
