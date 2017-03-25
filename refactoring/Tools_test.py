#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 19:25:28 2017

@author: brais
"""

import Tools as tl
import scipy.stats as sps
import pytest
from numpy.testing import assert_allclose#, assert_almost_equal
from hypothesis import given #, example
from hypothesis.strategies import integers, floats #, lists
from hypothesis.extra.numpy import arrays
import numpy as np
import inspect
from flaky import flaky

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

@given(integers(min_value=2 , max_value =100),
       integers(min_value=1 , max_value =10 ),
       integers(min_value=11, max_value =20 ) )
def test_build_Histogram_correct_return(nbin, minlim, maxlim):

    test_array     = (maxlim-minlim) * np.random.random(100) + minlim 
    nphist, npbins = np.histogram(test_array, nbin, [minlim, maxlim])
    npbins         = npbins[:-1] + np.diff(npbins)/2.
    hist           = tl.build_Histogram(test_array, nbin, minlim, maxlim)
    bool_bins      = (npbins == hist.bins).all()
    bool_hist      = (nphist == hist.hist).all()
    assert( bool_bins and bool_hist)

@given(integers(min_value=2 , max_value =100),
       integers(min_value=1 , max_value =10 ),
       integers(min_value=11, max_value =20 ) )
def test_generate_model_wrong_number_of_parameters(nbin, minlim, maxlim):

    test_array = (maxlim-minlim) * np.random.random(100) + minlim 
    xrange     = np.linspace(minlim-1, maxlim+1, 100)
    hist       = tl.build_Histogram(test_array, nbin, minlim, maxlim)
    model      = tl.generate_model(xrange, [hist], 'linear')
    with pytest.raises(ValueError):
        model( np.array([1., 2.]) )


@given(integers(min_value=2 , max_value =100),
       integers(min_value=1 , max_value =10 ),
       integers(min_value=11, max_value =20 ) )
def test_generate_model_zeros_out_of_range(nbin, minlim, maxlim):

    test_array = (maxlim-minlim) * np.random.random(100) + minlim 
    xrange     = np.linspace(minlim-1, maxlim+1, 100)
    hist       = tl.build_Histogram(test_array, nbin, minlim, maxlim)
    model      = tl.generate_model(xrange, [hist], 'linear')
    ydata      = model( np.array([1.]) )
    assert( ydata[0]==0. and ydata[-1]==0 )


@given(integers(min_value=2 , max_value =100),
       integers(min_value=1 , max_value =10 ),
       integers(min_value=11, max_value =20 ) )    
def test_generate_model_normalization(nbin, minlim, maxlim):

    test_array = (maxlim-minlim) * np.random.random(100) + minlim 
    xrange     = np.linspace(minlim-1, maxlim+1, 100)
    hist       = tl.build_Histogram(test_array, nbin, minlim, maxlim)
    model      = tl.generate_model(xrange, [hist], 'linear')
    ydata      = model( np.array([1.]) )
    ydata_mult = model( np.array([2.]) )
    assert_allclose( ydata.sum()     , 1.)
    assert_allclose( ydata_mult.sum(), 2.)
    


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
    test = tl.generalLogPoisson(0, mu)
    assert_allclose(-mu,test)

@given(integers(min_value=0, max_value=50),
       floats(min_value=0.0001, max_value=100))
def test_generalLogPoisson(x, mu):
    '''
    For int x inputs it should agree with the scipy implementation
    '''
    sp_value = sps.poisson.pmf(x, mu)
    test_val = np.exp(tl.generalLogPoisson(x, mu))
    assert_allclose(sp_value, test_val, atol=1e-7, rtol= 1e-4)


def test_LLh_fun_args():
    '''
    Test the properties of the Lhh_fun return. It must depend only on one
    variable
    '''
    func      = lambda x: x
    inspector = inspect.getargspec(tl.generate_LLh_fun(func, [1]))
    assert len(inspector.args) == 1
    assert inspector.varargs   == None
    assert inspector.keywords  == None
    assert inspector.defaults  == None

@given(floats(min_value=1, max_value =50),
       floats(min_value=1, max_value =50),
       floats(min_value=1, max_value =50),
       floats(min_value=1, max_value =50))
def test_LLh_fun_model(a1 ,a2, b1, b2):
    '''
    Test the properties of the Lhh_fun return. For any set of arguments, the
    minimum value for the LL must be with the ones used for building the model
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
    input equals 0 must return 2*mu.
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
    input equals 0 must return 2*mu.
    '''
    mu       = np.array([mu0, mu1, mu2])
    func     = lambda x: x
    LLh      = tl.generate_LLh_fun(func, np.array([0,0,0]))
    val_test = LLh(mu)
    assert_allclose(2*mu.sum(), val_test)

#@pytest.mark.skip
@pytest.mark.slow
@flaky(max_runs=3, min_passes=1)
@given(floats(min_value=10, max_value=50),
       floats(min_value=10, max_value=50))
def test_LLh_min_StraightLine_NoSmearing(a, b):
    '''
    Testing minimizing function.
    '''

    x     = np.linspace(0,10,100)
    func  = lambda arg: arg[0]*x+arg[1]
    ydata = func([a,b])
    res   = tl.minimize_LLh(func, ydata, [a,b], False)
    assert_allclose(res.x, [a,b], rtol=1e-5)

#@pytest.mark.skip
@pytest.mark.slow
@flaky(max_runs=3, min_passes=1)
@given(floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050))
def test_LLh_min_StraightLine_Smearing(a, b):
    '''
    Testing minimizing function with a fit to a straight line without
    smearing.
    '''

    x     = np.linspace(0,10,1000)
    func  = lambda arg: arg[0]*x+arg[1]
    ydata = np.random.poisson(func([a,b]))
    res   = tl.minimize_LLh(func, ydata, [a,b], False)
    assert_allclose(res.x, [a,b], rtol=1e-2)

#@pytest.mark.skip
@pytest.mark.slow
@flaky(max_runs=3, min_passes=1)
@given(floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050))
def test_LLh_min_2Gauss_Smearing(a, b):
    '''
    Testing minimizing function.
    '''
    x =   np.linspace(0,10,1000)
    def func(arg):
        A,B = arg
        return A*np.exp(-(x-2)**2/0.5) + B*np.exp(-(x-4)**2/1.5)
    ydata = np.random.poisson(func([a,b]))
    res   = tl.minimize_LLh(func, ydata, [a,b], False)
    assert_allclose(res.x, [a,b], rtol=1e-2)

#@pytest.mark.skip
@pytest.mark.slow
@flaky(max_runs=3, min_passes=1)
@given(floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050))
def test_generate_LLh_scan_minimun_value(a, b):
    '''
    Testing scanning LLh function. At minimum must return the minimum value
    already computed by minimize_LLh.
    '''

    x           = np.linspace(0,10,100)
    func        = lambda arg: arg[0]*x+arg[1]
    ydata       = func([a,b])
    LLh         = tl.generate_LLh_fun(func, ydata)
    res         = tl.minimize_LLh(func, ydata, [a,b], False)
    list_values = np.zeros_like(res.x)
    for i in range(len(res.x)):
        scan_function   = tl.generate_LLh_scan(i, LLh, res.x)
        list_values[i]  = scan_function(res.x[i])
    assert_allclose(list_values, res.fun*np.ones_like(res.x))

#@pytest.mark.skip
@pytest.mark.slow
@flaky(max_runs=3, min_passes=1)
@given(floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050),
       floats(min_value=1000, max_value=1050))
def test_generate_LLh_scan_list(a1, a2, b1, b2):
    '''
    Testing scanning LLh function. Any pair of values (b1, b2) different
    from the ones used to build the model (a1, a2) should yeld a Scan LL
    greater. So, LL_scan should have a global minimum at (a1, a2).
    '''

    res_t       = [b1, b2]
    x           = np.linspace(0, 10, 100)
    func        = lambda arg: arg[0]*x+arg[1]
    ydata       = func([a1, a2])
    LLh         = tl.generate_LLh_fun(func, ydata)
    res         = tl.minimize_LLh(func, ydata, [a1, a2], False)
    list_values = np.zeros_like(res.x)
    for i in range(len(res.x)):
        scan_function  = tl.generate_LLh_scan(i, LLh, res.x)
        list_values[i] = scan_function(res_t[i])
    ones_like  = np.ones_like(res.x)
    assert (list_values -res.fun*ones_like >= -1e-7*ones_like).any()

@pytest.mark.slow
@given(floats  (min_value=100, max_value=150),
       integers(min_value=2,   max_value=10))
def test_confidence_interval_NoSmearing(a, bins):
    '''
    Testing confidence interval function. For a given model (very simple) with
    no smearing, this confidence interval must be the poissonian variance of 
    the parameter used for building the model.
    '''
    
    x             = np.linspace(0, 10, bins)
    func          = lambda arg: np.array([a]*len(arg))
    ydata         = func(x)
    LLh           = tl.generate_LLh_fun(func, ydata)
    res           = tl.minimize_LLh(func, ydata, [a], False)
    conf_interval = tl.confidence_interval(LLh, res.x, 1.)
    test_vals     = np.array([[np.sqrt(a)], [np.sqrt(a)]])
    diff          = np.abs(conf_interval-np.array([res.x, res.x]))
    hoping_zero   = np.sum(test_vals - diff)
    assert_allclose(hoping_zero,0, atol=1e-13 )

@pytest.mark.slow
@given(floats  (min_value=100, max_value=150),
       integers(min_value=2,   max_value=10))
def test_confidence_interval_Smearing(a, bins):
    '''
    Testing confidence interval function. For a given model (very simple) with
    smearing, this confidence interval must be the poissonian variance of 
    the parameter value obtained fitting.
    '''
    
    x             = np.linspace(0, 10, bins)
    func          = lambda arg: np.array([a]*len(arg))
    ydata         = np.random.poisson(func(x))
    LLh           = tl.generate_LLh_fun(func, ydata)
    res           = tl.minimize_LLh(func, ydata, [a], False)
    conf_interval = tl.confidence_interval(LLh, res.x, 1.)
    test_vals     = np.array([np.sqrt(res.x), np.sqrt(res.x)])
    diff          = np.abs(conf_interval-np.array([res.x, res.x]))
    assert_allclose(test_vals, diff, rtol=1e-13)
