import numpy as np

Nav = 6.022140857e23 #mol^-1
Mmol = 135.907214484 #gmol
ln2 = log(2)

def XeEstimation(mass = 5.81,purity = 0.91,trun = 100):

    trun *= 1./365.24

    t12 = 2.165e21
    et12 = 0.061e21

    lamb = ln2/t12
    slamb = ln2*et12/t12**2

    N0 = Nav*mass*1000*purity/Mmol

    Nfin = N0*lamb*trun

    return Nfin