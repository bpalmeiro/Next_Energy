#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:25:28 2017

@author: brais
"""

import Tools as tl
import scipy.stats as sps
import pytest
from numpy.testing import assert_allclose
from hypothesis import given, example
from hypothesis.strategies import integers, floats, lists
from hypothesis.extra.numpy import arrays
import numpy as np
import inspect 

def float_arrays(size       =   100,
                 min_value  = -1e20,
                 max_value  = +1e20,
                 mask       =  None,
                 **kwargs          ):
    elements = floats(min_value,
                      max_value,
                      **kwargs)
    if mask is not None:
        elements = elements.filter(mask)
    return arrays(dtype    = np.float32,
                  shape    =       size,
                  elements =   elements)


def FLOAT_ARRAY(*args, **kwargs):
    return float_arrays(*args, **kwargs).example()


def random_length_float_arrays(min_length =     0,
                               max_length =   100,
                               **kwargs          ):
    lengths = integers(min_length,
                       max_length)

    return lengths.flatmap(lambda n: float_arrays(       n,
                                                  **kwargs))




def test_generalLogPoisson_parameters():
    '''
    input arrays must have the same length
    '''
    with pytest.raises(TypeError):
        tl.generalLogPoisson([1,2,3],[1,2,3,4])

@given(floats(min_value=0.0001, allow_infinity=False) )  
def test_generalLogPoisson_NullValue(mu):
    '''
    At 0, the logPoiss should return -mu
    '''
    aux = tl.generalLogPoisson(0, mu)
    assert_allclose(-mu,aux)
        
@given(integers(min_value=0, max_value=50), 
       floats(min_value=0.0001, max_value=100))
def test_generalLogPoisson(x, mu):
    '''
    For int x inputs it should agree with the scipy 
    implementation
    '''
    aux1 = sps.poisson.pmf(x, mu)
    aux2 = np.exp(tl.generalLogPoisson(x, mu))
    assert_allclose(aux1, aux2, atol=1e-7, rtol= 1e-4)

    
def test_LLh_fun_args():
    '''
    Test the properties of the Lhh_fun return
    '''
    auxfun = lambda x: x
    aux    = inspect.getargspec(tl.generate_LLh_fun(auxfun,[1]))
    assert len(aux.args) == 1
    assert aux.varargs   == None
    assert aux.keywords  == None
    assert aux.defaults  == None

@given(floats(min_value=1, max_value =50, allow_infinity=False),
       floats(min_value=1, max_value =50, allow_infinity=False),
       floats(min_value=1, max_value =50, allow_infinity=False),
       floats(min_value=1, max_value =50, allow_infinity=False))
def test_LLh_fun_model(a1,a2,b1,b2):
    '''
    Test the properties of the Lhh_fun return. For any set of arguments, the 
    minimum value for the LL must be with the ones where the model is build 
    from
    '''
    x        = np.linspace(0,10,100)
    func     = lambda arg: arg[0]*x+arg[1]
    ydat     = func([a1,a2])
    LLh      = tl.generate_LLh_fun(func, ydat)
    val_test = LLh([b1,b2])
    val_reff = LLh([a1,a2])
    assert val_test - val_reff  >=-1e-7

    
@given(floats(min_value=0.0001, allow_infinity=False))    
def test_LLh_fun_Poiss_Null(mu):
    '''
    At 0, the logPoiss should return -mu. So, as LLh is -2*Poiss, LL with ydat
    input equals 0 must return 2*mu
    ''' 
    func     = lambda x: x
    LLh      = tl.generate_LLh_fun(func,0)
    val_test = LLh(mu)
    assert_allclose(2*mu, val_test)
    
@given(floats(min_value=0.0001, allow_infinity=False),
       floats(min_value=0.0001, allow_infinity=False),
       floats(min_value=0.0001, allow_infinity=False))    
def test_LLh_fun_PoissNullList(mu0, mu1, mu2):
    '''
    At 0, the logPoiss should return -mu. So, as LLh is -2*Poiss, LL with ydat
    input equals 0 must return 2*mu
    ''' 
    mu       = np.array([mu0, mu1, mu2])
    func     = lambda x: x
    LLh      = tl.generate_LLh_fun(func, np.array([0,0,0]))
    val_test = LLh(mu)
    assert_allclose(2*mu.sum(), val_test)

    
@given(floats(min_value=1, max_value=50, allow_infinity=False),
       floats(min_value=1, max_value=50, allow_infinity=False))    
def test_LLh_min_StraightLine_NoSmearing(a,b):
    '''
    Testing minimizing function
    '''
    
    x        = np.linspace(0,10,100)
    aux_func = lambda arg: arg[0]*x+arg[1]
    y        = aux_func([a,b])
    res      = tl.minimize_LLh(aux_func, y, [a,b], False)
    assert_allclose(res.x, [a,b])
    
    
@given(floats(min_value=10, max_value=50, allow_infinity=False),
       floats(min_value=10, max_value=50, allow_infinity=False))    
def test_LLh_min_StraightLine_Smearing(a,b):
    '''
    Testing minimizing function
    '''
    
    x        = np.linspace(0,10,1000)
    aux_func = lambda arg: arg[0]*x+arg[1]
    y        = np.random.normal(aux_func([a,b]),0.1)
    res      = tl.minimize_LLh(aux_func, y, [a,b], False)
    assert_allclose(res.x, [a,b], rtol=1e-2)

@given(floats(min_value=10, max_value=50, allow_infinity=False),
       floats(min_value=10, max_value=50, allow_infinity=False))        
def test_LLh_min_2Gauss_Smearing(a,b):
    '''
    Testing minimizing function
    '''
    x = np.linspace(0,10,1000)
    def aux_func(aux):
        A,B = aux
        return A*np.exp(-(x-2)**2/0.5) + B*np.exp(-(x-4)**2/1.5)       
    y   = np.random.normal(aux_func([a,b]), 0.1)
    res = tl.minimize_LLh(aux_func, y, [a,b], False)
    assert_allclose(res.x, [a,b], rtol=1e-2)    

    
@pytest.mark.skip()
@given(floats(min_value=10, max_value=50, allow_infinity=False),
       floats(min_value=10, max_value=50, allow_infinity=False))          
def test_Get_Chi2(a,b):
    '''
    #Testing Get_Chi function
    '''      
    ndim = 100
    x = np.array([100]*ndim)  
    y = np.random.gauss(100,10,ndim)
    aux = tl.Get_Chi2(x,y,0)
    aux1 = sps.chisquare(x,y,0)
    print(aux,aux1)
    assert pytest.approx(aux, 1.5) == 1. 
    
    
'''  
@given(floats(min_value=10, max_value=50, allow_infinity=False),
       floats(min_value=10, max_value=50, allow_infinity=False))          
def test_Get_Chi2(a,b):
'''
    #Testing Get_Chi function
'''      
    x = np.linspace(0,10,10)
    def aux_func(aux):
        A,B = aux
        return A*np.exp(-(x-2)**2/0.5) + B*np.exp(-(x-4)**2/1.5)       
    y   = np.random.poisson(aux_func([a,b]))
    res = tl.LLh_min(aux_func, y, [a,b])
    assert pytest.approx(res.chi2, 1.5) == 1. 
'''