import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Type
from copy import deepcopy
from nlmpy import nlmpy
from sklearn.neighbors import KernelDensity
from dms_variants.ispline import Isplines
import dms_variants
from scipy.stats import norm, multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from importlib.resources import files, as_file
import shutil
import subprocess
import warnings
import time
import os
import dill
from math import comb
import rasterio as rio
import xarray as xr
import rioxarray as rxr

from . import pygdm

'''
simulate community composition at any or all points in a simulated landscape

input parameters include:
    - landscape dimensions
    - number of environmental layers
    - spatial autocorrelation of each of those layers
    - knots and coefficients for functions defining the relationship between
      environmental and ecological turnover (i.e., f(Env) functions)
    - γ-diversity (total species count on landscape)


NEXT STEPS:
    - add functionality for removing species (based on niche width/specialization or something like that)
    - add sampling function!
    - teaser scenarios:
        - show effects of sampling and sample site density on GAM richness model
        - b4/af env change scenario (CC, hab loss, combo)
        - nestedness vs turnover (fix all niche mus to landscape mean but let sigmas still vary)
    - allow both detection probabilities and lambda values (and ideally just
      whole user-built species, including niches) to be proapgated through or rest on Sim
    - GDM fits getting assigned to the sim leaves only space for one, which
      doesn't facilitate comparing results; reconfigure this (perhaps just
      create GDM class to return results in and give it a plot fn?)
    - prevent our f(Env) functions from being anchored at 0 on y-axis?
    - finalize input vs GDM-fitted f(Env) plotting issues
    - which is more justifiable, product of univariate normals or multivar normal?
    - is it a major problem that we're ignoring spatial autocorr in pres/abs determination? if so, add Gaussian random field?
    - add ability for knots and splines to be fed through as args to R's gdm()

'''

########################
# FUNCTIONS AND CLASSES:
########################

#-------------
# type aliases
#-------------

numerical = Union[float, int]
vectorlike = Union[List[numerical], Tuple[numerical], np.ndarray]
rasterlike = Union[np.ndarray, xr.core.dataarray.DataArray]
raster_or_vectorlike = Union[List[numerical], Tuple[numerical], np.ndarray, xr.core.dataarray.DataArray]


#-----------------
# helper functions
#-----------------

def _standardize_vec(vec: vectorlike):
    '''
    standardize a numerical vector-like object
    '''
    return (np.array(vec) - np.nanmean(vec))/np.nanstd(vec)


def _rescale_arr(arr: Union[vectorlike, rasterlike],
                 new_scale: Optional[vectorlike] = [0, 1],
                 by_rast_band : bool = False,
                ) -> Union[np.ndarray, xr.core.dataarray.DataArray]:
    '''
    linearly recast an array to a new interval (default: [0,1])
    using min-max scaling, optionally by raster band (i.e., by index on axis 0;
    defaults to rescaling the entire raster's set of values, regardless of axes)
    '''
    assert len(new_scale) == 2
    new_range = new_scale[1] - new_scale[0]
    assert new_range >= 0
    out = deepcopy(arr)
    if by_rast_band:
        assert len(out.shape) == 3
        for i in range(out.shape[0]):
            out[i] = (((out[i]-np.nanmin(out[i])) * new_range)/
                       (np.nanmax(out[i])-np.nanmin(out[i]))) + new_scale[0]
    else:
        out = (((out-np.nanmin(out)) * new_range)/
               (np.nanmax(out)-np.nanmin(out))) + new_scale[0]
    return out


def _standardize_arr(arr: Union[vectorlike, rasterlike],
                     by_rast_band : bool = False,
                    ) -> Union[np.ndarray, xr.core.dataarray.DataArray]:
    '''
    recast an array to the standard normal distribution (i.e., ~N(0, 1)),
    optionally by raster band (i.e., along axis 0),
    '''
    out = deepcopy(arr)
    if by_rast_band:
        assert len(out.shape) == 3
        for i in range(out.shape[0]):
            out[i] = (out[i] - np.nanmean(out[i]))/np.nanstd(out[i])
    else:
        out = (out - np.nanmean(out))/np.nanstd(out)
    return out



#--------
# classes
#--------

class fEnv:
    '''
    class for `f(Env)` function that relates environmental and ecological
    distances as a monotonic function (i.e., a linear combination of I-spline
    basis functions with non-negative coefficients)

    Includes a method for getting the function's approximate slope at any value
    along the range of the environmental variable.

    Parameters
    ----------
    id : int
        Integer identifier for this environmental variable/layer (used in
        axis labels and to align with raster bands elsewhere in `Sim`).
    knots : vectorlike
        Ordered (low to high) knot locations, in the units of the
        environmental variable, defining the I-spline basis.
    coeffs : vectorlike
        Coefficients for the linear combination of I-spline basis
        functions. Must have length `len(knots) + 1`, and the final
        coefficient must be 0.0 (ensuring f(Env)'s slope goes to 0 at the
        high end of the environmental gradient).
    order : int, default 3
        Order of the I-spline basis functions.
    env_x_vals : vectorlike or None, default None
        Ordered (low to high) sequence of environmental values at which to
        evaluate f(Env). If None, defaults to 1000 points evenly spaced
        between the minimum and maximum knot values.

    Attributes
    ----------
    x : np.ndarray
        Environmental values at which f(Env) is evaluated.
    y : np.ndarray
        Corresponding f(Env) values.
    splines : dms_variants.ispline.Isplines
        The underlying I-spline basis-function object.
    x_gdm_fit, y_gdm_fit : np.ndarray or None
        x and y values of a GDM-fitted version of this f(Env), populated
        via `_add_GDM_fit` once a GDM has been run; None until then.
    '''
    def __init__(self,
                 id: int,
                 knots: vectorlike,
                 coeffs: vectorlike,
                 order: int = 3,
                 env_x_vals: Optional[vectorlike] = None,
                ) -> None:
        self.knots = np.array(knots)
        self.coeffs = np.array(coeffs)
        # validate and process args
        assert np.all((self.knots[1:] - self.knots[:-1]) >= 0), ("knots "
                                                       "must be ordered "
                                                       "from low to high.")
        assert len(self.coeffs) == len(self.knots)+1, ("length of coeffs must "
                                        "be 1 greater than number of knots.")
        assert self.coeffs[-1] == 0, ("to ensure slope of 0 at high end of "
                                "f(Env), the final coefficient must be 0.0.")
        # space environmental values evenly between 0 and 1000, if not provided
        if env_x_vals is None:
            env_x_vals = np.linspace(np.min(self.knots),
                                     np.max(self.knots), 1000)
        else:
            assert type(env_x_vals) in [list, tuple, np.ndarray]
            assert np.all((env_x_vals[1:] - env_x_vals[:-1]) >= 0), ("ispline_x"
                                                   "must be an ordered "
                                                   "range of x values "
                                                   "(i.e., environmental "
                                                   "values).")
        # save integer ID of this spline
        self.id = id
        # create the I-splines and the f(Env) that is their linear combination
        fenv, isplines = self._make_fEnv(knots=self.knots,
                                         coeffs=self.coeffs,
                                         ispline_order=order,
                                         ispline_x=env_x_vals,
                                        )
        # check length and monotonicity of resulting f(Env)
        assert len(fenv) == len(env_x_vals)
        assert np.all((fenv[1:] - fenv[:-1])>=0)
        # save x (i.e., env) and y (i.e., f(Env))
        self.x  = env_x_vals
        self.y = fenv
        # save min and max values
        self._x_min = np.min(self.x)
        self._x_max = np.max(self.x)
        self._y_min = np.min(self.y)
        self._y_max = np.max(self.y)
        self._x_minmax = (self._x_min, self._x_max)
        self._y_minmax = (self._y_min, self._y_max)
        # save iSplines object
        self.splines = isplines
        # placeholders for GDM-fitted values
        self.x_gdm_fit = None
        self.y_gdm_fit = None
        # total range of the environmental gradient
        self._x_range = np.max(self.x) - np.min(self.x)
        # get min and max slope values
        self._slope_min = np.min((self.y[1:]-self.y[:-1])/(self.x[1:]-self.x[:-1]))
        self._slope_max = np.max((self.y[1:]-self.y[:-1])/(self.x[1:]-self.x[:-1]))


    def _make_fEnv(self,
                   knots,
                   coeffs,
                   ispline_order: Optional[int] = 3,
                   ispline_x: Optional[vectorlike] = None,
                  ) -> np.ndarray:
        '''
        use the knots and coefficients provided to create and return a numpy
        array approximating f(Env) (a linear combination of a series of
        I-spline basis functions), as well as the dms_variants.isplines.Isplines
        object that provides the Ispline basis functions
        '''
        # if x is not provided, create it as 1000-point linearly spaced array
        # of values between the environmental values of the lowest and highest
        # knots
                # create the I-spline basis functions
        isplines = self._make_Ispline_basis(mesh=knots,
                                            x=ispline_x,
                                            order=ispline_order,
                                           )
        # create function as linear combination
        fenv = np.stack([coeffs[i-1] * isplines.I(i) for i in range(1,
                                        isplines.n+1)]).sum(axis=0)/isplines.n
        return fenv, isplines


    def _make_Ispline_basis(self,
                            mesh: vectorlike,
                            x: vectorlike,
                            order: Optional[int] = 3,
                           ) -> dms_variants.ispline.Isplines:
        '''
        create set of I-splines basis functions and return them (as a
        `dms_variants.ispline.Isplines object)
        '''
        isplines = Isplines(order=order,
                            mesh=mesh,
                            x=x,
                           )
        return isplines


    def _get_approx_slope(self,
                         x: numerical,
                        ) -> float:
        '''
        calculate simple local approximation of slope at value x
        (NOTE: for values of x that fall  below or above the
        minimum and maximum x values specified on the f(Env) function,
        the values returned are simply the slopes at the minimum and
        maximum x values)
        '''
        assert pd.notnull(x), "x must not be null"
        assert not np.isinf(x), "x must not be infinite"
        ind = np.argmin(np.abs(self.x - x))
        assert ind>=0 and ind<= len(self.x-1)
        ind_lo, ind_hi = np.clip([ind-1, ind+1], a_min=0, a_max=len(self.x)-1)
        Δx = self.x[ind_hi] - self.x[ind_lo]
        Δy = self.y[ind_hi] - self.y[ind_lo]
        return Δy/Δx


    def _add_GDM_fit(self,
                    x_gdm_fit: np.ndarray,
                    y_gdm_fit: np.ndarray,
                   ) -> None:
        '''
        add attributes to store a GDM-fitted f(Env) function
        '''
        self.x_gdm_fit = x_gdm_fit
        self.y_gdm_fit = y_gdm_fit
        self._y_gdm_min = np.min(self.y_gdm_fit)
        self._y_gdm_max = np.max(self.y_gdm_fit)


    def plot(self,
             ax: Optional[mpl.axes._axes.Axes] = None,
             include_input: bool = True,
             legend: bool = False,
            ) -> None:
        '''
        make simple plot of f(Env) and its GDM fit
        '''
        if ax is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("$Env_%i$" % self.id, size=14)
        if self.x_gdm_fit is not None and self.y_gdm_fit is not None:
            new_scale = (self._y_gdm_min, self._y_gdm_max)
            y_plot = _rescale_arr(arr=self.y, new_scale=new_scale)
        else:
            y_plot = self.y[:]
        if include_input:
            ax.plot(self.x,
                    y_plot,
                    label='input',
                    linewidth=1.0,
                    color='black',
                    linestyle='-',
                   )
        if self.x_gdm_fit is not None and self.y_gdm_fit is not None:
            ax.plot(self.x_gdm_fit,
                    self.y_gdm_fit,
                    label='GDM fit',
                    linewidth=1.0,
                    color='red',
                    linestyle=':',
                    alpha=0.5,
                   )
        ax.set_xlabel("$Env_%i$" % self.id)
        ax.set_ylabel("$f(Env_%i)$" % self.id)
        if legend:
            ax.legend()
        ax.grid(True)


class Species:
    '''
    A simulated species, represented by its niche parameters, abundance
    parameter, and detection probability.

    Parameters
    ----------
    niche : list of vectorlike
        One (mu, sigma) pair per environmental layer, defining the mean and
        standard deviation of the species' Gaussian niche along that
        environmental axis (in the native units of that environmental
        layer).
    max_poisson_lambda : float
        The Poisson lambda used to draw abundance at a location where the
        species' probability of presence equals 1.0 (a ceiling on expected
        abundance; realized abundance at a given site is drawn from a Poisson
        ditstribution that uses this value linearly scaled down according to
        the Species' local probability of presence).
    prob_detect : float or None
        The species' intrinsic detection probability (i.e., on the [0, 1]
        interval), used when sampling with `by_detect_prob=True`.
        Can be None if detection probability isn't being modeled.

    Attributes
    ----------
    niche : list of 2-tuples
        Defined by the `niche` parameter.
    max_poisson_lambda : float
    prob_detect : float or None

    '''
    def __init__(self,
                 niche: list[vectorlike],
                 max_poisson_lambda: float,
                 prob_detect: float,
                ) -> None:
        # validate args
        for i in range(len(niche)):
            assert len(niche[i]) == 2 # mu and sigma
        assert prob_detect is None or 0 <= prob_detect <= 1
        # assign attributes
        self.niche = niche
        self.max_poisson_lambda = max_poisson_lambda
        self.prob_detect = prob_detect


class Sim:
    '''
    Top-level simulation object.

    A simulated landscape, species pool, and
    the resulting community composition at every site on that landscape.

    A `Sim` bundles: (1) the landscape (i.e., a stack of one or more
    environmental raster layers), (2) the regional species pool (i.e.,
    a set of `Species` objects, the total size of which is determined
    by the `gamma` parameter, with environmental niches drawn along each of the
    axes defined by the landscape's layers, (3) `fEnv` functions relating
    environmental turnover to ecological turnover (by way of providing the
    basis for `Species`' simualted niches), and (4) the resulting simulated
    communities (species presence and abundances) at every raster cell.

    Parameters
    ----------
    env : list of rasterlike
        One 2D raster (np.ndarray or xarray.DataArray) per environmental
        layer, all of identical shape.
    fenvs : list of fEnv
        One `fEnv` object per environmental layer in `env`, in the same
        order, describing that layer's relationship to ecological
        turnover.
    gamma : int
        Total number of species in the regional species pool
        (i.e., 'inventory diversity', sensu Whittaker).
    min_niche_sigma : float, default 0.001
        Minimum niche breadth (in standard-deviation units of the
        relevant environmental layer) that any species can have on any axis.
    max_niche_sigma : float, default 0.1
        Maximum niche breadth (in standard-deviation units) that any species
        can have on any axis. If None, defaults, for each landscape layer, to half
        of that layer's environmental range.
    use_multivar_normal_niche : bool, default False
        If True, model species' niches as a joint multivariate normal
        across all environmental axes, rather than as a product of
        independent univariate normals.
    prob_pres_thresh_round_to_1 : float or None, default None
        If provided, any computed presence probability at or above this
        threshold is rounded up to 1.0.
    max_poisson_lambdas : vectorlike or None, default None
        Per-species ceiling values for the abundance-determining Poisson
        distribution (see `Species.max_poisson_lambda`). If None, drawn
        randomly (uniform) between 1 and `max_poisson_lambda_across_spp`
        for each species.
    max_poisson_lambda_across_spp : int, default 1000
        Upper bound used when randomly drawing `max_poisson_lambdas`.
    detect_probs : vectorlike or None, default None
        Per-species detection probabilities. If None, drawn randomly
        (uniform on [0, 1]) for each species.
    verbose : bool, default False
        If True, print progress information during simulation setup.
    debug : bool, default False
        If True, print additional information useful for debugging.
    timeit : bool, default True
        If True, print the runtime of the slower setup steps (species
        creation, community simulation).

    Attributes
    ----------
    env : np.ndarray
        Stacked environmental layers, shape (n_layers, nrow, ncol) -- which is to
        say (n_layers, y, x).
    fenvs : list of `fEnv` objects
    gamma : int
    sites : list of 2-tuples of floats
        The (x, y) coordinates (cell centers) of every raster cell's simulated community.
    spp : dict
        Mapping of species IDs (ints) to `Species` objects.
    comms : list of dict
        The full simulated community (as a {species ID: abundance} dict) at every
        site in `sites`, and in the same order as `sites`, to allow ordinal
        indexing relied on throughout the package.
    surveys, gdm_fits, gdm_pca_rast
        These start as empty attributes, but are populated by `sim_obs` and
        `run_GDM`, if/when those methods have been executed. Until then, they
        are None.
    '''
    def __init__(self,
                 env: List[rasterlike],
                 fenvs: list[Type[fEnv]],
                 gamma: int,
                 min_niche_sigma: float = 0.001,
                 max_niche_sigma: float = 0.1,
                 use_multivar_normal_niche: bool = False,
                 prob_pres_thresh_round_to_1: Optional[float] = None,
                 max_poisson_lambdas: Optional[vectorlike] = None,
                 max_poisson_lambda_across_spp: int = 1000,
                 detect_probs: Optional[vectorlike] = None,
                 verbose: bool = False,
                 debug: bool = False,
                 timeit: bool = True,
                ) -> None:
        # validate args
        # store behavioral params
        self._verbose = verbose
        self._debug = debug
        self._timeit = timeit
        self._use_multivar_normal_niche = use_multivar_normal_niche
        self._prob_pres_thresh_round_to_1 = prob_pres_thresh_round_to_1
        # store hidden utility attributes
        self._env_cmaps = ['Reds', 'Greens', 'Blues']
        self._n_lyrs = len(env)
        self._dims = env[0].shape
        # store fixed params
        self.min_niche_sigma = min_niche_sigma
        self.max_niche_sigma = max_niche_sigma
        self.max_poisson_lambda_across_spp = max_poisson_lambda_across_spp
        assert len(fenvs) == self._n_lyrs
        self.fenvs = fenvs
        self._fenvs_slope_min = np.min([s._slope_min for s in self.fenvs])
        self._fenvs_slope_max = np.max([s._slope_max for s in self.fenvs])
        self.gamma = gamma
        # set the environment
        self.update_env(env,
                        verbose=self._verbose,
                        debug=self._debug,
                       )
        # make niche KDE
        self._make_niche_kde()
        # create community sites
        n_comms = np.prod(self._dims)
        sites = self._get_community_sites(n_comms)
        self.sites = sites
        self.n_sites = len(self.sites)
        # create all species
        self._make_all_species_runtime = None
        if self._timeit:
            start = time.time()
        if self._verbose:
            print(f"\n\n\tCREATING SPECIES...\n\n")
        self._make_all_species(max_poisson_lambdas=max_poisson_lambdas,
                               detect_probs=detect_probs,
                              )
        if self._timeit:
            stop = time.time()
            runtime_sec = stop-start
            self._make_all_species_runtime = runtime_sec
            if self._verbose:
                print(("\n\n\tALL SPECIES CREATED IN "
                       f"{np.round(self._make_all_species_runtime/60, 2)} "
                       "MINUTES.\n\n"))
        # simulate the communities
        self._runtime_sim_comms = None
        self._sim_comms(verbose=self._verbose,
                        timeit=self._timeit,
                        debug=self._debug,
                       )
        # save empty attributes that will be refilled/replaced as the Sim
        # object is used
        self.surveys = None
        self.gdm_fits = None
        self.gdm_pca_rast = None


    def update_env(self,
                   env: List[rasterlike],
                   verbose: Optional[bool] = None,
                   timeit: Optional[bool] = None,
                   debug: Optional[bool] = None,
                   recalc_std: bool = False,
                  ) -> None:
        '''
        Replace the Sim's environmental layers, to simulate
        environmental change, then re-simulate.

        Note: Defaults to not updating the standard deviation of the layers
              (which are used to rescale species niche widths).
        Note: Only resimulates community composition if the environment
              has actually changed.

        Parameters
        ----------
        env : list of rasterlike
            New set of environmental layers, one per existing layer,
            matching the landscape's original dimensions.
        verbose : bool or None, default None
            If True, print progress information. If None, uses the Sim's
            stored `_verbose` setting.
        timeit : bool or None, default None
            If True, print the runtime of the community re-simulation.
            If None, uses the Sim's stored `_timeit` setting.
        debug : bool or None, default None
            If True, print debugging information. If None, uses the Sim's
            stored `_debug` setting.
        recalc_std : bool, default False
            If True, recalculate each layer's standard deviation (used to
            rescale species' niche widths) from the new environment. If
            False, the standard deviations of the original environment
            are retained even though the environment itself is updated.

        Returns
        -------
        None
            Updates `self.env` (and `self.comms`, if the environment
            changed) in place.
        '''
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        assert len(env) == self._n_lyrs
        assert np.all(env[0].shape == self._dims)
        # check if the environment is different
        env_changed = hasattr(self, 'env') and (not np.all(self.env == env))
        # convert environment to np.ndarray and store it
        self.env = np.array(env)
        # add a depth-1 0th dimension, if env is just a single raster
        if self._n_lyrs == 1:
            self.env = self.env.reshape(-1, *self.env.shape)
        self._env_min_vals = [np.min(e) for e in self.env]
        self._env_max_vals = [np.max(e) for e in self.env]
        # set the environmetn's standard deviation, if doesn't yet exist and/or
        # if recalc_std is True
        if not hasattr(self, '_env_stds') or recalc_std:
            self._env_stds = [np.std(e) for e in self.env]
        # redraw communities, if the environment has changed
        if env_changed:
            del self.comms
            self._sim_comms(verbose=verbose, timeit=timeit, debug=debug)


    def _get_community_sites(self, n: int) -> List[Tuple[float]]:
        '''
        get the set of survey site points within a raster whose coordinates
        range from 0 to dim-1 in both axes in dims; each point will be in a
        separate raster cell, so n must not exceed the number of pixels
        '''
        dims = self._dims
        assert n > 0 and n <= np.prod(dims)
        X, Y = np.meshgrid(range(dims[0]), range(dims[1]))
        xs = X.ravel()
        ys = Y.ravel()
        pts = [*zip(xs, ys)]
        if n < np.prod(dims):
            np.random.shuffle(pts)
            idxs = np.random.choice(range(len(pts)), replace=False, size=n)
        else:
            idxs = np.array([*range(len(pts))])
        # NOTE: add 0.5 to all site points, to place them in cell centers
        pts = [tuple(np.array(pts[idx])+0.5) for idx in idxs]
        return pts


    def _make_niche_kde(self,
                        kernel='gaussian',
                        bandwidth='scott',
                       ):
        # got rid of failed alpha idea; just sampling whole environment evenly
        draw_cts = np.ones(self.env[0].shape)
        assert np.all(draw_cts % 1 == 0)
        draw_lists = []
        for e in self.env:
            draws = []
            for i, ct in enumerate(draw_cts.ravel()):
                for n in range(int(ct)):
                    draws.append(e.ravel()[i])
            draw_lists.append(draws)
        kde = KernelDensity(kernel=kernel,
                            bandwidth=bandwidth).fit(np.array(np.array(draw_lists).T))
        self._niche_kde = kde


    def _make_all_species(self,
                          max_poisson_lambdas: Optional[vectorlike] = None,
                          detect_probs: Optional[vectorlike] = None,
                         ) -> None:
        '''
        create a dict of all species' ecological niches
        (i.e., μ and σ values for all environmental layers)
        '''
        # draw species' niche centers from the niche KDE
        spp_mus = self._niche_kde.sample(self.gamma)

        # draw species' lambdas for Poisson distributions determining survey
        # results (will be multiplied by probability of presence at a location, so
        # this is the maximum value that a Poisson draw will take in a location
        # where probability of presence goes to 1.0)
        if max_poisson_lambdas is None:
            max_poisson_lambdas = np.random.uniform(1,
                                                    self.max_poisson_lambda_across_spp,
                                                    self.gamma,
                                                   )
        else:
            assert np.all(max_poisson_lambdas > 0)
        # draw detection probabilities randomly, if not provided
        if detect_probs is None:
            detect_probs = np.random.uniform(low=0, high=1, size=self.gamma)
        else:
            assert type(detect_probs) in [list, tuple, np.ndarray]
            assert len(detect_probs) == self.gamma
            assert np.all(detect_probs >= 0)
            assert np.all(detect_probs <= 1)
        # use inverse of f(Env) slope at each μ to draw each σ
        # (following a rationale derived independently but that aligns with
        # Bush et al. 2019)
        spp = {}
        for s, mus in zip(range(self.gamma), spp_mus):
            niche = []
            for i, mu in enumerate(mus):
                slope = self.fenvs[i]._get_approx_slope(mu)
                # use min-max scaling to determine slope proportional position
                # between min and max slope values, then remap to interval between
                # user-specified max and min niche widths
                slope_prop = ((slope - self._fenvs_slope_min)/
                              (self._fenvs_slope_max - self._fenvs_slope_min))
                if self.max_niche_sigma is None:
                    max_sigma_i = self.fenvs[i]._x_range/2
                else:
                    max_sigma_i = self.max_niche_sigma
                # NOTE: max niche width defaults to half the range of this
                #       environmental variable if not user-specified
                sigma = max_sigma_i - (
                        slope_prop*(max_sigma_i - self.min_niche_sigma))
                sigma = np.clip(sigma,
                                a_min=self.min_niche_sigma,
                                a_max=max_sigma_i,
                               )
                # now rescale sigma (currently expressed in standard deviations)
                # to the native distribution of the environmental layer (by
                # multiplying by its standard deviation)
                sigma_scaled = sigma * self._env_stds[i]
                niche.append((mu, sigma_scaled))
                # now multiply that by the standard deviation of the
                # environmental layer
            # create and save the Species
            sp = Species(niche=niche,
                         max_poisson_lambda=max_poisson_lambdas[s],
                         prob_detect=detect_probs[s],
                        )
            spp[s] = sp
        self.spp = spp


    def _calc_pres_prob(self,
                        env_vals: vectorlike,
                        niche: Tuple[float],
                        use_multivar_normal: bool = False,
                        prob_pres_thresh_round_to_1: Optional[float] = None,
                       ) -> float:
        '''
        for a location described by the given environmental values,
        calculate the probability of presence of a species with the given niche
        '''
        if not use_multivar_normal:
            # list of probability densities extracted from the normal distributions
            # describing the species' niches on each environmental axis
            probs = []
            for n, e in enumerate(env_vals):
                # get probability of presence for this axis by determining the
                # probability of drawing, from the species' niche distribution
                # on this environmental axis, a value equally or more extreme
                # than the survey position's environmental value
                # NOTE: calculating and then subtracting difference between survey
                #       location's environmental value and niche center, then
                #       subtracting that from the niche center in the CDF
                #       calculation, thus getting the probability of a value being
                #       that far below the niche center; then multiply by two to
                #       get two-tailed probability of a value as extreme or more so
                diff = np.abs(e-niche[n][0])
                prob_n = 2 * (norm.cdf(x=niche[n][0]-diff,
                                       loc=niche[n][0],
                                       scale=niche[n][1],
                                      ))
                assert 0 <= prob_n <= 1
                probs.append(prob_n)
            # determine overall probability of presence as the product of all
            # probabilities (i.e., the joint probability across all
            # environmental axes, treating the axes as if they are independent...
            # NOTE: ... even though in reality we could actually fold in cross-layer
            #       correlation to account for chance non-independence between
            #       environmental axes...)
            # NOTE: ... we also ignore spatial autocorrelation of presence in real
            #       species by ignoring any information about whether or not the
            #       species has been determined present in proximal locations...
            prob = np.prod(probs)**(1/3)
        else:
            # get arrays of niche centers and niche widths
            mus = np.array([n[0] for n in niche])
            sigmas = np.array([n[1] for n in niche])
            # construct covariance matrix (NOTE: without covariance between layers!)
            covar = np.zeros([len(mus)]*2)
            covar[np.diag_indices_from(covar)] = sigmas
            # get probability of presence using the cumulative distribution
            # function of the multivariate normal described by the species' niche
            # distributions on all axes (modeled as the probability of drawing
            # from within the species' multivariate normal niche space
            # a series of environmental values equally extreme as or more extreme
            # than the environmental values observed as the survey position)
            # NOTE: calculating and then subtracting difference between survey
            #       location's environmental values and multivariate niche center,
            #       then subtracting that from the niche center in the CDF
            #       calculation, thus getting the probability of a value being
            #       that far below the niche center; then multiplying by two to
            #       get the two-tailed probability of a value as extreme or more so)
            diffs = np.abs(env_vals-mus)
            distr = multivariate_normal(mean=mus, cov=covar, allow_singular=False)
            prob = 2 * distr.cdf(x=mus-diffs)
            assert 0 <= prob <= 1
        # round values to 1 above a certain value, if required
        if (prob_pres_thresh_round_to_1 is not None and
            prob >= prob_pres_thresh_round_to_1):
            prob = 1
        return prob


    def _sim_comm(self,
                  i: float,
                  j: float,
                  max_prob_pres: float = 1.0,
                  prob_pres_thresh_round_to_1: Optional[float] = None,
                  use_multivar_normal: bool = False,
                  debug: bool = False,
                 ) -> Dict[int, int]:
        '''
        use the list of environmental layers provided and the dict of species
        and their niches to simulate complete community composition at grid cell i,j
        '''
        # list to store all species present
        survey = {}
        # get environmental values at point grid cell i,j
        # NOTE: site points sit at cell centers, so int() converts to their cell indices
        env_vals = [e[int(i), int(j)] for e in self.env]
        for s, sp in self.spp.items():
            # calculate presence probability
            prob = self._calc_pres_prob(env_vals,
                                   sp.niche,
                                   use_multivar_normal=self._use_multivar_normal_niche,
                                   prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                 )
            # determine presence as a Bernoulli draw on that probability
            # NOTE: ... treating all layers as independent, even though
            #       in reality we could actually fold in cross-layer
            #       correlation to account for chance non-independence between
            #       environmental axes...)
            # NOTE: ... we also ignore spatial autocorrelation of presence in real
            #       species by ignoring any information about whether or not the
            #       species has been determined present in proximal locations...
            if np.random.binomial(1, prob):
                # if present, draw abundance from Poisson
                # NOTE: altogether, this models survey results as an
                # environmentally conditional zero-inflated Poisson
                survey[s] = np.random.poisson(prob * sp.max_poisson_lambda)
        return survey


    def _sim_comms(self,
                   verbose: Optional[bool] = None,
                   timeit: Optional[bool] = None,
                   debug: Optional[bool] = None,
                  ) -> None:
        '''
        Produces a list of dicts, where each dict is the community (species id
        keys and count values) at each site in self.sites.
        '''
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        if timeit:
            start = time.time()
        if verbose:
            print(f"\n\n\tSIMULATING COMMUNITIES AT SURVEY POINTS...\n\n")
        # create the simulated communities at each point
        if not hasattr(self, 'comms'):
            comms = []
            ct = 0
            for i, j in self.sites:
                if verbose:
                    if ct%25 == 0:
                        print(f"\n\t{np.round(100*(ct/len(self.sites)), 1)}% complete...\n")
                survey = self._sim_comm(i,
                                       j,
                                       use_multivar_normal=self._use_multivar_normal_niche,
                                       prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                       debug=debug,
                                      )
                comms.append(survey)
                ct+=1
                # store the full communities
                self.comms = comms
                # NOTE: flip the _env_changed flag to False (it will stay
                # that way unless and until the env is updated again)
                self._env_changed = False
        else:
            pass
        # store and report runtime, as needed
        if timeit:
            stop = time.time()
            runtime_sec = stop-start
            self._runtime_sim_comms = runtime_sec
            if verbose:
                print(("\n\n\tALL COMMUNITIES SIMULATED IN "
                       f"{np.round(self._runtime_sim_comms/60, 2)} "
                       "MINUTES.\n\n"))


    def save_to_file(self, filepath):
        '''
        Save entire `Sim` object to a pickle file (i.e., .pkl).

        Parameters
        ----------
        filepath : str
            Destination path; must end with '.pkl'.

        Returns
        -------
        None
        '''
        assert filepath.endswith('.pkl')
        with open(filepath, "wb") as f:
            dill.dump(self, f)


    @classmethod
    def load_from_file(cls, filepath):
        '''
        Load a `Sim` object from a pickle file (i.e., .pkl).

        Parameters
        ----------
        filepath : str
            Path to the pickle file to load; must end with '.pkl'.

        Returns
        -------
        Sim
            The restored `Sim` object.

        '''
        assert filepath.endswith('.pkl')
        with open(filepath, "rb") as f:
            return dill.load(f)


    def _get_site_indices(self, survey_sites):
        '''
        Returns the integer site indices pertaining to each of a list of survey
        sites, which can be directly used to index self.sites or self.comms.
        '''
        if survey_sites is None:
            list_inds = [*range(len(self.sites))]
        else:
            # NOTE: 1 and 0 are reverse-ordered because x,y site expression from user
            #       tranlsates to j,i indexing the way sites are identified in the model
            for s in survey_sites:
                assert 0<= s[1] <= self._dims[0]
                assert 0<= s[0] <= self._dims[1]
            # NOTE: each cell covers the span from its LL corner to just before its
            #       UR corner, and Py zero-indexed, so this works out to any pair
            #       of continuous coordinates being directly covertable to its cell
            #       using simple int flooring
            #       (e.g., (0.76, 2.3) falls within cell 2,0, which ranges from 0
            #       to just less than 1 in the x dimension and 2 to just less than
            #       3 in the y dimension)
            # NOTE: sites are id'd by the coordinate pair of their centroids, so
            #       add 0.5 to each
            cell_inds = [(int(s[0])+0.5, int(s[1])+0.5) for s in survey_sites]
            list_inds = [[i for i, s in enumerate(self.sites)
                          if s==cell_ind][0] for cell_ind in cell_inds]
        return list_inds


    def sim_obs(self,
                scheme: str = 'perfect',
                abund: bool = True,
                absen: bool = True,
                survey_sites: List[Tuple[float]] = None,
                effort: Optional[Union[int, float, list, tuple, np.ndarray]] = None,
                by_rel_abund: bool = True,
                by_detect_prob: bool = False,
                save: bool = False,
                site_survey_filepath: str = None,
                save_env: bool = False,
                env_raster_filepath: str = None,
                allow_overwrite: bool = False,
                verbose: Optional[bool] = None,
                timeit: Optional[bool] = None,
                debug: Optional[bool] = None,
               ) -> pd.DataFrame:
        '''
        Simulate observed species samples from a community.

        Simulates observations at all survey_sites using the given scheme
        (defaults to 'perfect', which simply returns the complete
        simulated communities at each site), site-specific measures of effort
        (defaults to None, which returns a single 'opportunistic' sighting),
        and whether sampling probabilities should be determined as a function of
        relative abundances and/or species' intrinsic detection probabilities
        (both default to None, which yields uniform sampling probabilities
        across all individuals)

        Returns a pandas.DataFrame of observations, with site-by-species (i x
        j) matrix structure for abundance-absence and presence-absence data,
        or with one row per species obseration for presence-only data

        Parameters
        ----------
        scheme : str
            Sampling scheme to use. Can be one of:
            - ``'perfect'`` : perfect detection, no sampling error.
            - ``'sample'`` : stochastic sampling of the underlying community.
        abund : bool, default True
            If True, report species' sampled abundances. If False, report presence
            only. Combines with `absen` to determine output data type (see Notes).
        absen : bool, default True
            If True, include species' absences (zero counts) in the output.
            Combines with `abund` to determine output data type (see Notes).
        survey_sites : list of two-tuples of floats, or None, default None
            A list of tuples of the x,y coordinates of all sites to be sampled.
            If None, defaults to taking one sample at the center of every
            raster cell on the landscape, where each simulated community is
            located.
        effort : float, or int, or vectorlike, or None, default None
            Per-site sampling effort, expressed as a float (or int) on the [0, 1]
            interval (where 1 = 100% effort = whole community observed).
            Can be a float (same sampling effort at all `survey_sites`) or a vectorlike
            (numpy.ndarray, list, or tuple of per-site sampling efforts).
            Defaults to None, which just returns a single observation per site.
        by_rel_abund : bool, default True
            If True, draw species observations in proportion to their relative
            abundances, such that rarer species are observed less often. Can
            be combined with `by_detect_prob` (see Notes).
        by_detect_prob : bool, default False
            If True, use each species' a priori specified detection
            probability to determine whether it is sampled. Can be combined
            with `by_rel_abund` (see Notes).
        save : bool, default False
            If True, saves results to file (ready for GDM input).
        site_survey_filepath : str, default None
            If `save` is True, a filepath must be provided for the site-survey
            data to be saved to (and it must end with '.csv').
        save_env : bool, default False
            If True (and if `save` also True), saves landscape to a raster file
            as well.
        env_raster_filepath : str, default None
            If `save_env` is True, a filepath must be provided for the
            environment raster to be saved to (and it must end with '.tif').
        allow_overwrite: bool, default False
            Whether or not to allow existing files to be overwritten.
        verbose : bool, default False
            If True, print progress and diagnostic information.
        timeit : bool, default False
            If True, print the total runtime of the function.
        debug : bool, default False
            If True, print information useful for debugging.

        Returns
        -------
        pandas.DataFrame
            For abundance-absence and presence-absence data, one row per
            sampling site, with site IDs and coordinates in 'site',
            'x', and 'y' columns and counts of species i in 'spp<i>' columns.
            For abundance-only and presence-only data, one row per species
            observation.

        Notes
        -----
        `abund` and `absen` jointly determine the output data type:

        - ``abund=True, absen=True``   : abundance-absence data (counts, with
          zeros for undetected species).
        - ``abund=True, absen=False``   : abundance-only data (counts for only
          species that were observed)
        - ``abund=False, absen=True``  : presence-absence data (1/0, with
          undetected species shown as 0).
        - ``abund=False, absen=False`` : presence-only data (only detected
          species are included, no zeros).

        `by_rel_abund` and `by_detect_prob` may both be True at once, in which
        case their effects combine multiplicatively: a species that is both
        rare and hard to detect is less likely to be observed than either
        factor alone would predict.
        '''
        # handle sampling-scheme arguments
        assert scheme in ['perfect', 'sample']
        if survey_sites is not None:
            assert isinstance(survey_sites, list)
            assert np.all([isinstance(s, tuple) for s in survey_sites])
        if effort is not None:
            if isinstance(effort, int):
                assert effort == 0 or effort == 1
                effort = float(effort)
            if isinstance(effort, float):
                assert 0 <= effort <= 1
                efforts = [effort] * len(self.comms)
            else:
                assert type(effort) in [list, tuple, np.ndarray]
                assert len(effort) == len(self.sites)
                assert np.all(effort >= 0)
                assert np.all(effort <= 1)
                efforts = effort
        else:
            efforts = None
        if save:
            assert site_survey_filepath is not None
            assert site_survey_filepath.endswith('.csv')
            if not allow_overwrite:
                assert not os.path.isfile(site_survey_filepath)
        if save_env:
            assert env_raster_filepath is not None
            assert env_raster_filepath.endswith('.tif')
            if not allow_overwrite:
                assert not os.path.isfile(env_raster_filepath)
        if verbose is None:
            verbose = self._verbose
        if timeit is None:
            timeit = self._timeit
        if debug is None:
            debug = self._debug
        if timeit:
            start = time.time()
        if verbose:
            if scheme == 'perfect':
                label = 'PERFECT '
            elif scheme == 'sample':
                label = ''
            print(f"\n\n\tSIMULATING {label}SAMPLING "
                  "AT SURVEY POINTS...\n\n")
        # get the site-indices associated with the input survey sites
        site_inds = self._get_site_indices(survey_sites)
        # just take communities, if scheme is 'perfect'...
        if scheme == 'perfect':
            obs = [self.comms[i] for i in site_inds]
        # ...otherwise, get list of simulated observations at each site
        elif scheme == 'sample':
            obs = []
            tot = len(site_inds)
            for i, comm in enumerate([self.comms[i] for i in site_inds]):
                if verbose and i % 100 == 0:
                    print(f"\t{np.round((i+1)/tot*100, 1)}% complete...")
                if efforts is not None:
                    effort = efforts[i]
                else:
                    effort = None
                ob = self._sim_sample(comm=comm,
                                      effort=effort,
                                      by_rel_abund=by_rel_abund,
                                      by_detect_prob=by_detect_prob,
                                     )
                obs.append(ob)
        # convert abundances to presences, if needed
        if not abund:
            obs = [{k: min((1, v)) for k,v in d.items()} for d in obs]
        # add absences for unobserved species, if needed
        if absen:
            for ob in obs:
                for spp in self.spp.keys():
                    if spp not in ob:
                        ob[spp] = 0
        # save data, if needed
        if abund:
            bio_data_type = 'abun'
        else:
            bio_data_type = 'pres'
        site_surv_df = self._prep_output_data(surveys=obs,
                                              survey_pts=survey_sites,
                                              bio_data_type=bio_data_type,
                                              absen=absen,
                                              save_site_surveys=save,
                                              site_survey_filepath=site_survey_filepath,
                                              save_env_rast=save_env,
                                              env_rast_filepath=env_raster_filepath,
                                             )
        return site_surv_df


    def _sim_sample(self,
                    comm: Dict[int, int],
                    effort: Optional[float] = None,
                    by_rel_abund: bool = True,
                    by_detect_prob: bool = False,
                   ) -> Dict[int, int]:
        '''
        simulate a sample of species observations from the community provided
        using sampling arguments including effort (default to None, in which case
        only a single 'opportunistic' sample is returned; otherwise constrained
        to [0, 1]), and whether or not relative abundances and/or intrinsic
        detection probabilities should influence species' observation probabilities
        '''
        # get total number of individuals in the whole community
        N = np.sum([*comm.values()])
        # just return empty sample, if community is empty
        if N == 0:
            return {}
        else:
            # copy the comm, for use as a counter object
            counter = deepcopy(comm)
            # create output object
            sample = {}
            # get vector of probs that a single sighting happens to be of each species
            # (starts as all ones, then gets multiplied by needed values)
            sp_probs = np.ones(len(comm))
            # multiply by abundances (normalized to probs),
            # if relative abundance needs to factor into sampling probs
            if by_rel_abund:
                sp_probs *= (np.array([*comm.values()])/(np.sum([*comm.values()])))
            # mutliply by species' intrinsic detection probabilities, if needed
            if by_detect_prob:
                sp_probs *= np.array([self.spp[sp].prob_detect for sp in comm.keys()])
            # now renormalize to probabilities that sum to 1
            sp_probs = sp_probs/np.sum(sp_probs)
            assert np.allclose(np.sum(sp_probs), 1)
            # use effort and rarefaction to determine size of sample...
            if effort is not None:
                # NOTE: FOR NOW, ASSUMES SIMPLE LINEAR SCALING OF SAMPLE SIZE WITH EFFORT
                n = int(np.round(N*effort, 0))
            # ... or set it to 1, if effort is not provided and this is thus an
            # 'opportunistic' sample
            else:
                n = 1
            # loop over sample size, draw samp, and pop it from counter into sample
            while np.sum([*sample.values()]) < n:
                sp = np.random.choice([*comm.keys()], p=sp_probs)
                if sp in counter:
                    counter[sp] -= 1
                    if counter[sp] == 0:
                        del counter[sp]
                    if sp in sample:
                        sample[sp] += 1
                    else:
                        sample[sp] = 1
                else:
                    pass
            # check all counts are <= full count in comm
            for sp in sample:
                assert sample[sp] <= comm[sp]
            # check total sample size is correct
            assert np.sum([*sample.values()]) == n
            if effort is None:
                assert np.sum([*sample.values()]) == 1
            return sample


    def _convert_wide_to_long_survey_df(self,
                                        df,
                                        drop_absen=True,
                                        drop_abund=True,
                                       ):
        '''
        Convert a site-by-species pd.DataFrame to one that just has a single
        row per species observation.
        '''
        # pivot
        df_melt = df.melt(id_vars=['site',
                                   'x',
                                   'y'],
                          value_vars=[c for c in df.columns if c.startswith('spp')],
                          var_name='spp',
                          value_name='abund',
                         )
        # recast species as integers
        df_melt['spp'] = [int(v.lstrip('spp')) for v in df_melt['spp'].values]
        # resort
        df_melt = df_melt.sort_values(['site', 'spp'])
        # drop zeros, for abundance-only or presence-only data
        if drop_absen:
            df_melt = df_melt[df_melt['abund']>0]
        # drop the abund column, if presence-only
        if drop_abund:
            df_melt = df_melt.drop(labels=['abund'], axis=1)
        return df_melt


    def _prep_output_data(self,
                          surveys: List[Dict[int, int]],
                          survey_pts: List[Tuple[float]] = None,
                          bio_data_type: str = 'abun',
                          absen: bool = True,
                          save_site_surveys: bool = False,
                          site_survey_filepath: str = 'sobig_site_survey.csv',
                          save_env_rast: bool = False,
                          env_rast_filepath: str = 'sobig_env_rast.tif',
                         ) -> None:
        '''
        Prep a set of files for ouput.

        Abundance-absence and presence-absence data are formatted to match inputs
        for the basic R script for running GDM (a site-by-species table).
        Abundance-only and presence-only data just have a row per species
        observation.
        '''
        # create and save 'site-survey' table
        # (sites in rows, species in columns)
        n_spp = self.gamma
        n_sites = len(surveys)
        # NOTE: adding 3 to include a site column and x and y columns
        add_cols = 3
        site_surv_mat = np.zeros((n_sites, n_spp+add_cols))
        # NOTE: add site column
        site_surv_mat[:, 0] = [*range(len(surveys))]
        for i, survey in enumerate(surveys):
            # add x and y survey-point columns
            if survey_pts is not None:
                pt = survey_pts[i]
            else:
            # NOTE: sites are expressed as (i, j) matrix indices,
            #       so flip them to express as (x, y) geographic coordinates)
                pt = self.sites[i]
            site_surv_mat[i, 1] = pt[1]
            site_surv_mat[i, 2] = pt[0]
            for j, abund in survey.items():
                site_surv_mat[i, j+add_cols] = abund
        # convert counts to presences, if needed
        if bio_data_type == 'pres':
            site_surv_mat[:, 3:] = np.clip(site_surv_mat[:, 3:], a_min=None, a_max=1)
        site_surv_df = pd.DataFrame(site_surv_mat)
        site_surv_df.columns = ['site', 'x', 'y'] + [f'spp{i}' for i in range(n_spp)]
        # reformat as a simple 'species list' table, if no absences are to be
        # included (and drop the abund-column, if this is presence-only data
        # ratehr than abundance-only... recognizing that the latter is a bit odd)
        if not absen:
            site_surv_df = self._convert_wide_to_long_survey_df(site_surv_df,
                                                                drop_absen=True,
                                                                drop_abund=bio_data_type=='pres',
                                                               )
        if save_site_surveys:
            site_surv_df.to_csv(site_survey_filepath, index=False)
        print("\n\tSITE-DATA TABLE SAVED TO DISK.\n")
        # create and save environmental raster
        if save_env_rast:
            ydim, xdim = self.env.shape[1], self.env.shape[2]
            n_bands = self.env.shape[0]
            dtype = self.env.dtype
            crs = 'EPSG:3857' # just a stand-in projected EPSG, to avoid CRS issues
            transform = rio.transform.from_origin(0, ydim, 1, 1) # top-left corner
            with rio.open(env_rast_filepath,
                          'w',
                          driver='GTiff',
                          height=ydim,
                          width=xdim,
                          count=n_bands,
                          dtype=dtype,
                          crs=crs,
                          transform=transform) as dst:
                for n in range(n_bands):
                    dst.write(self.env[n], n + 1)
            print("\n\tENV RAST SAVED TO DISK.\n")
        else:
            print("\n\tENV RAST NOT SAVED.\n")
        return site_surv_df


    def run_GDM(self,
                surveys: Optional[List[Dict[int, int]]] = None,
                gdm_data_type: str = 'abun',
                site_survey_filepath: str = 'sobig_site_survey.csv',
                env_rast_filepath: str = 'sobig_env_rast.tif',
                delete_intermed_files: bool = False,
                fits_filepath: str = 'sobig_GDM_fits.csv',
                pca_rast_filepath: str = 'sobig_GDM_env_rast_PCA.tif',
                plot_it: bool = False,
                plot_fenv_input: bool = True,
                plot_title: str = '',
                verbose: bool = False,
                implementation: str = 'r', # NOTE: 'r' runs GDM using the Fitzpatrik
                                            #       et al. code;
                                            #       'py' runs GDM hastily ported to
                                            #       Python by Claude (since I suddenly
                                            #       stopped being able to install R's
                                            #       'gdm' package and didn't feel like
                                            #       wasting more time debugging), 
                                            # which it's worth noting is much slower,
                                            #       lacking the Cpp optimizer.
               ) -> None:
        '''
        Run Generalized Dissimilarity Modeling (GDM) on simulatedcommunity
        data.

        Prepares the site-by-species survey table and environmental
        raster, writes them to disk, then runs GDM using either R's `gdm`
        package (via an Rscript subprocess, if available) or
        else a minimalist Python port of GDM (ported by Claude, as a standin).
        Attaches the results (fitted fEnvs curves and PCA-transformed raster)
        as attributes on the Sim.

        Parameters
        ----------
        surveys : list of dict, or None, default None
            The per-site community data (each site's data being a
            {species ID: abundance} dict) to run GDM on.
            If None, uses the Sim's full set of complete, simulated
            communities (`Sim.comms`).
        gdm_data_type : str, default 'abun'
            Either ``'abun'`` (abundance data) or ``'pres'``
            (presence/absence data).
        site_survey_filepath : str, default 'sobig_site_survey.csv'
            Path to write the site-by-species survey table to.
        env_rast_filepath : str, default 'sobig_env_rast.tif'
            Path to write the environmental raster to.
        delete_intermed_files : bool, default False
            If True, delete `site_survey_filepath`
            and `env_rast_filepath` after GDM has run
            (and try to delete `fits_filepath` and
            `pca_rast_filepath` as well).
        fits_filepath : str, default 'sobig_GDM_fits.csv'
            Path the fitted I-spline curves are written to
            (for R implementation only).
        pca_rast_filepath : str, default 'sobig_GDM_env_rast_PCA.tif'
            Path the GDM-transformed PCA raster is written to
            (for R implementation only).
        plot_it : bool, default False
            If True, call `self.plot` after GDM has run.
        plot_fenv_input : bool, default True
            Passed through to `self.plot` if `plot_it` is True.
        plot_title : str, default ''
            Passed through to `self.plot` if `plot_it` is True.
        verbose : bool, default False
            If True, print progress information (including the exact
            Rscript command run, if applicable).
        implementation : str, default 'r'
            Either ``'r'`` (run GDM via R's `gdm` package, invoked as a
            subprocess) or ``'py'`` (use the Python port of GDM). Falls
            back automatically from ``'r'`` to ``'py'`` (with a warning)
            if `Rscript` or the R `gdm` package aren't available, or if the
            R run fails. Note that the Python implementation was simply ported
            from R's `gdm` package by Claude, and it runs considerably
            slower it lacks the support of the R package's compiled (C++) optimizer.

        Returns
        -------
        gdm_fits : pandas.DataFrame
            Fitted I-spline curves, one x/y column pair per environmental
            raster band.
        pca_rast_rescaled : xarray.DataArray
            A raster of the top three PCs from the GDM transform, min-max
            rescaled by band, and padded with all-zero bands if
            fewer than 3 environmental layers were used.

        Notes
        -----
        As a side effect, this also sets `self.gdm_fits`,
        `self.gdm_pca_rast`, and, for each `fEnv` in `self.fenvs`, its
        `x_gdm_fit` / `y_gdm_fit` attributes.
        '''
        assert isinstance(gdm_data_type, str)
        assert gdm_data_type in ['abun', 'pres']
        assert implementation in ['r', 'py']
        print(f"\n\nRUNNING GDM...\n\n")
        # use the complete communities, if surveys were not provided
        if surveys is None:
            surveys = self.comms
        # prep and save GDM input data
        site_surv_df = self._prep_output_data(surveys=surveys,
                                              survey_pts=None,
                                              bio_data_type=gdm_data_type,
                                              absen=True,
                                              save_site_surveys=True,
                                              site_survey_filepath=site_survey_filepath,
                                              save_env_rast=True,
                                              env_rast_filepath=env_rast_filepath,
                                             )
        if implementation == 'r':
            if shutil.which("Rscript") is None:
                warnings.warn("Rscript was not found on the system. "
                              "Skipping GDM analysis. Please ensure R is installed and "
                              "Rscript is available on your PATH.\n"
                              "Meanwhile, defaulting to slower Python implementation of GDM.\n\n",
                              RuntimeWarning,
                             )
                implementation = 'py'
        if implementation == 'r':
            # run R script
            if gdm_data_type == 'abun':
                abund = 'TRUE'
            else:
                abund = 'FALSE'

            r_script = files("sobig").joinpath("_r", "run_gdm.R")
            with as_file(r_script) as script_path:
                R_cmd = ["Rscript",
                         "--vanilla",
                         r_script,
                         site_survey_filepath,
                         env_rast_filepath,
                         abund,
                         fits_filepath,
                         pca_rast_filepath,
                        ]
                if verbose:
                    print(f"\tNOW RUNNING: > {R_cmd}\n")
                result = subprocess.run(R_cmd,
                                        capture_output=True,
                                        text=True,
                                       )
            if result.returncode != 0:
                warnings.warn("The R GDM analysis failed. "
                              "This may indicate that R and/or the R 'gdm' package "
                              "are not properly installed.\n"
                              f"R output:\n{result.stderr}\n"
                              "Meanwhile, defaulting to slower Python implementation of GDM.\n\n",
                              RuntimeWarning,
                             )
                implementation = 'py'
            else:
                # read and return results
                gdm_fits = pd.read_csv(fits_filepath)
                pca_rast = rxr.open_rasterio(pca_rast_filepath)
        if implementation == 'py':
            site_survey_df = pd.read_csv(site_survey_filepath)
            env_rast = rxr.open_rasterio(env_rast_filepath)
            gdm_fits, pca_rast = pygdm.run_GDM(site_table=site_survey_df,
                                               env_raster=env_rast,
                                               abund=gdm_data_type=='abun',
                                               geo=False,
                                               n_splines=3,
                                               curve_points=200,
                                               max_iter=100,
                                              )
        # min-max scale raster (comes in as 0-255 from R, or centered on 0 from
        # py implementation)
        pca_rast_rescaled = _rescale_arr(pca_rast, by_rast_band=True)
        # save the output GDM fits and raster to their Sim attributes
        self.gdm_fits = gdm_fits
        for i, fenv in enumerate(self.fenvs, start=1):
            fenv._add_GDM_fit(x_gdm_fit=self.gdm_fits[f"x.env_rast_{i}"],
                              y_gdm_fit=self.gdm_fits[f"y.env_rast_{i}"],
                             )
        # extend the first axis of the GDM PC raster to length 3, if necessary,
        # by providing layers of all 0s
        n_lyrs_add = 3 - pca_rast_rescaled.shape[0]
        if n_lyrs_add > 0:
            pca_rast_rescaled = xr.concat([pca_rast_rescaled,
                    pca_rast_rescaled[:n_lyrs_add, :, :]*0], dim='band')
            # NOTE: update the 'long_name' field
            pca_rast_rescaled = pca_rast_rescaled.assign_attrs({'long_name':
                                                        ['PC1', 'PC2', 'PC3']})
        self.gdm_pca_rast = pca_rast_rescaled
        if plot_it:
            self.plot(scatter_points=False,
                      plot_fenv_input=plot_fenv_input,
                      title=plot_title,
                      save=False,
                     )
        # delete intermediate files, if indicated
        if delete_intermed_files:
            os.remove(site_survey_filepath)
            os.remove(env_rast_filepath)
            # try to delete fitted fEnv and PCA raster files too, but if the R
            # run failed they may not have been successfully created, so in
            # that case just skip on by
            try:
                os.remove(fits_filepath)
                os.remove(pca_rast_filepath)
            except Exception:
                pass
        return gdm_fits, pca_rast_rescaled


    def plot(self,
             scatter_points: bool = True,
             plot_fenv_input: bool = True,
             title: str = '',
             save: bool = False,
             fig_filepath: str = None,
            ) -> None:
        '''
        Plot a summary figure of the simulation.

        Plot includes environmental layers,
        their fitted/input f(Env) curves, observed alpha-diversity across
        all rasters cells with communities, and an RGB raster of the top three
        PCs from a GDM PCA transform.

        Note: Requires `run_GDM` to have already been run (uses
        `self.gdm_pca_rast`).

        Parameters
        ----------
        scatter_points : bool, default True
            If True, overlay community point loations on the environmental
            raster and the PCA raster.
        plot_fenv_input : bool, default True
            If True, include each layer's originally specified (input)
            f(Env) curve alongside its GDM fit.
        title : str, default ''
            Overall figure title.
        save : bool, default False
            If True, save the figure to the filepath indicated by
            `fig_filepath`.
        fig_filepath : str, default None
            Filepath to save figure to, if `save` == True.


        Returns
        -------
        matplotlib.figure.Figure
            The assembled figure.
        '''
        fig = plt.figure(figsize=(16,16))
        fig.suptitle(title)
        gs = fig.add_gridspec(80, 100)

        # plot environment rasters
        axwidth = int(100/self.env.shape[0])-1
        for i, e in enumerate(self.env):
            ax = fig.add_subplot(gs[:20,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
            img = ax.imshow(e,
                            vmin=self._env_min_vals[i],
                            vmax=self._env_max_vals[i],
                            cmap=self._env_cmaps[i],
                           )
            plt.colorbar(img)
            # add survey sites
            if scatter_points:
                for point in self.sites:
                    ax.scatter(point[0],
                               point[1],
                               color='white',
                               edgecolor='black',
                               alpha=0.8,
                               s=24,
                              )
            ax.set_title("$Env_%s$" % i, size=14)

        # plot their fEnvs
        fenv_axs = []
        for i, fenv in enumerate(self.fenvs):
                        ax = fig.add_subplot(gs[25:40,
                                    (i*axwidth)+(i*1):((i+1)*axwidth)+((i+1)*1)])
                        fenv.plot(ax=ax,
                                  legend=i==(self.env.shape[0]-1),
                                  include_input=plot_fenv_input,
                                 )
                        fenv_axs.append(ax)
        fenv_ax_max_ylim = np.max([np.max(ax.get_ylim()) for ax in fenv_axs])
        for ax in fenv_axs:
            ax.set_ylim(0, fenv_ax_max_ylim)

        # plot raster of observed alpha values at all surveyed cells
        ax = fig.add_subplot(gs[50:, :30])
        survey_len_arr = np.ones(self.env[0, :, :].shape)*np.nan
        for pt, survey in zip(self.sites, self.comms):
            survey_len_arr[int(pt[0]), int(pt[1])] = len(survey)
        survey_lengths = [len(survey) for survey in self.comms]
        img = ax.imshow(survey_len_arr,
                        vmin=min(survey_lengths),
                        vmax=max(survey_lengths),
                       )
        plt.colorbar(img)
        ax.set_title('α-diversity at surveyed sites', size=14)

        # plot PCA rast from GDM transform
        ax = fig.add_subplot(gs[50:, 35:65])
        self.gdm_pca_rast.plot.imshow(ax=ax)
        # add survey sites
        if scatter_points:
            for point in self.sites:
                ax.scatter(point[0],
                           point[1],
                           color='white',
                           edgecolor='black',
                           alpha=0.8,
                           s=24,
                          )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')
        ax.set_title('top 3 PCs from GDM transform')

        # format plot and save
        fig.subplots_adjust(hspace=.25,
                            wspace=.25,
                           )
        fig.show()
        if save:
            assert fig_filepath is not None
            fig.savefig(fig_filepath, dpi=500)
        return fig


    def plot_expec_vs_obser_sp_distr(self,
                                     sp: int,
                                     title: Optional[str] = None,
                                     cmap: str = 'viridis',
                                     save: bool = False,
                                     fig_filepath: str = None,
                                     expec_ax: Axes = None,
                                     obser_ax: Axes = None,
                                    ) -> None:
        '''
        Plot a given species' presence-probability raster (i.e., expected
        presence) and its observed presence/absence across surveyed sites.

        Parameters
        ----------
        sp : int
            Species ID (i.e., the species' key within `Sim.spp`).
        title : str or None, default None
            Overall figure title (only used if a new figure is created,
            i.e. when `expec_ax`/`obser_ax` are not provided). If None,
            defaults to "sp. {sp}", including the species' detection
            probability if set.
        cmap : str, default 'viridis'
            Colormap used for both panels.
        save : bool, default False
            If True (and a new figure was created), save the figure to
            `fig_filepath`.
        fig_filepath : str, default None
            Filepath to save figure to, if `save` == True.
        expec_ax : matplotlib.axes.Axes or None, default None
            Axes to plot the expected-distribution panel on. If either
            this or `obser_ax` is None, a new figure (with its own
            environmental-layer panels) is created instead.
        obser_ax : matplotlib.axes.Axes or None, default None
            Axes to plot the observed-distribution panel on. See
            `expec_ax`.

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, if `expec_ax`/`obser_ax` were not
            provided; otherwise None (the panels are drawn directly onto
            the provided axes).
        '''
        # get species' niche
        niche = self.spp[sp].niche
        # calculate map of expected distribution
        expec = np.zeros(self.env[0, :, :].shape)
        # calculate presence probability at all cells
        for i in range(self.env[0, :, :].shape[0]):
            for j in range(self.env[0, :, :].shape[1]):
                prob = self._calc_pres_prob(env_vals=[e[i, j] for e in self.env],
                                       niche=self.spp[sp].niche,
                                       use_multivar_normal=self._use_multivar_normal_niche,
                                       prob_pres_thresh_round_to_1=self._prob_pres_thresh_round_to_1,
                                      )
                expec[i, j] = prob
        # calculate map of all pixels where species is observed
        # (setting pixels without communities to NaNs)
        obser = np.zeros(self.env[0, :, :].shape)
        for pt, survey in zip(self.sites, self.comms):
            if sp in survey:
                obser[int(pt[0]), int(pt[1])] = survey[sp]
        for i in range(self.env[0, :, :,].shape[0]):
            for j in range(self.env[0, :, :].shape[1]):
                if (i+0.5, j+0.5) not in self.sites:
                    obser[i, j] = np.nan
        # plot both
        show_fig = False
        if expec_ax is None or obser_ax is None:
            make_fig = True
            fig = plt.figure(figsize=(14,8))
        else:
            make_fig = False
        if title is None:
            title = f"sp. {sp}"
            if self.spp[sp].prob_detect is not None:
                title = title + " ($P(detect) = %0.2f$)" % self.spp[sp].prob_detect
        if make_fig:
            fig.suptitle(title)
            gs = fig.add_gridspec(nrows=80, ncols=140)
            axs_env = [fig.add_subplot(gs[:25,
                (i*20)+(i*5):(i+1)*20+(i*5)]) for i in range(self.env.shape[0])]
            expec_ax = fig.add_subplot(gs[25:, :55])
            obser_ax = fig.add_subplot(gs[25:, 85:])
            for i, e in enumerate(self.env):
                ax = axs_env[i]
                img = ax.imshow(e,
                                vmin=self._env_min_vals[i],
                                vmax=self._env_max_vals[i],
                                cmap=self._env_cmaps[i],
                               )
                ax.set_xticks(())
                ax.set_xticks(())
                plt.colorbar(img)
                ax.set_title("$Env_%s$" % i, size=14)
        im = expec_ax.imshow(expec,
                             cmap=cmap,
                             vmin=0,
                             vmax=1,
                            )
        plt.colorbar(im, label='$P(presence)$')
        expec_ax.set_title(f'sp. {sp}: expected')
        im = obser_ax.imshow(obser,
                             cmap=cmap,
                             vmin=0,
                             vmax=1,
                            )
        plt.colorbar(im, label='$presence$')
        obser_ax.set_title(f'sp. {sp}: observed')
        if make_fig:
            fig.subplots_adjust(hspace=0.25,
                                wspace=0.25,
                               )
            fig.show()
            if save:
                assert fig_filepath is not None
                fig.savefig(fig_filepath, dpi=500)
            return fig


def run_demo(dims=(10,10),
             gamma=20,
             seed=1,
             use_env_change=False,
             gdm_implementation='r',
            ):
    """
    Run a simple, self-contained demo of sobig's end-to-end functionality.

    Builds a small simulated landscape, a matching set of `fEnv` functions,
    and a species pool of total richness equal to `gamma`, then simulates
    community composition across the landscape. Runs GDM on
    the resulting data and plots the results.
    Optionally repeats the whole  process again after simulating
    an environmental change event.

    Parameters
    ----------
    dims : tuple of int, default (10, 10)
        (nrow, ncol) dimensions of the simulated landscape.
    gamma : int, default 20
        Gamma diversity (i.e., number of species in the landscape-wide species pool).
    seed : int or None, default 1
        Seed for NumPy's random number generator, for reproducibility. If
        None, no seed is set.
    use_env_change : bool, default False
        If True, after the initial GDM run and plot, perturb one
        environmental layer by adding spatially autocorrelated noise, then
        re-run GDM and plotting on the changed landscape.
    gdm_implementation : str, default 'r'
        Passed through to `Sim.run_GDM`'s `implementation` argument:
        either ``'r'`` or ``'py'``.

    Returns
    -------
    Sim
        The `Sim` object created and used for the demo (reflecting its
        state after the environmental-change step, if `use_env_change`
        was True).

    """
    # behavioral params
    VERBOSE = True
    DEBUG = True
    TIMEIT = True
    PLOT_IT = True
    SAVEPLOTS = True
    USE_MULTIVAR_NORMAL_NICHE = False
    MIN_NICHE_SIGMA = 0.01
    MAX_NICHE_SIGMA = 1.5
    PROB_PRES_THRESH_ROUND_TO_1 = None
    MAX_POISSON_LAMBDA_ACROSS_SPP = 1000
    GDM_DATA_TYPE = 'abun'
    if seed is not None:
        np.random.seed(seed)
    # param to determine number of species on whole landscape
    # (i.e., 'inventory' diversity, a la Whittaker)
    GAMMA = gamma
    # knots and coeffs for f(Env)
    knots = ([-1.5, -1, 0, 1, 1.5],
             [-1.3, -0.2, 0.2, 1.1, 1.3],
             [0, 10, 90, 100],
            )
    coeffs = ([1, 1.5, 2, 2.5, 3, 0],
              [0.1, 0.2, 0, 0.2, 6, 0],
              [0.1, 0.1, 0.1, 0.1, 0],
             )
    FENV = [fEnv(id=i, knots=k, coeffs=c) for i, (k, c) in enumerate(zip(knots,
                                                                         coeffs))]
    # landscape params
    DIMS = dims 
    ENV_H = (0.5, 0.5, 0.5)
    ADD_NOISE = True
    dist_source = np.zeros(DIMS)
    dist_source[int(DIMS[0]/2-1):int(DIMS[0]/2+1),
                int(DIMS[1]/2-1):int(DIMS[1]/2+1)] = 1
    ENV = [nlmpy.edgeGradient(nRow=DIMS[0], nCol=DIMS[1], direction=0),
           nlmpy.edgeGradient(nRow=DIMS[0], nCol=DIMS[1], direction=90),
           nlmpy.distanceGradient(dist_source),
          ]
    if ADD_NOISE:
        NOISE = [nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=h) for h in ENV_H]
        ENV = [nlmpy.blendArrays([e, n]) for e, n in zip(ENV, NOISE)]
    # rescale to a normal centered on 0
    ENV = [_rescale_arr(e, new_scale=fenv._x_minmax) for fenv, e in zip(FENV, ENV)]
    # species-specific lambdas (for ~Pois distributions determining abundance)
    MAX_POISSON_LAMBDAS = None
    # detection probability vector (or None, to have randomly assigned)
    DETECT_PROBS = None
    # create the simulator
    sim = Sim(env=ENV,
              fenvs=FENV,
              gamma=GAMMA,
              min_niche_sigma=MIN_NICHE_SIGMA,
              max_niche_sigma=MAX_NICHE_SIGMA,
              use_multivar_normal_niche=USE_MULTIVAR_NORMAL_NICHE,
              prob_pres_thresh_round_to_1=PROB_PRES_THRESH_ROUND_TO_1,
              max_poisson_lambda_across_spp=MAX_POISSON_LAMBDA_ACROSS_SPP,
              max_poisson_lambdas=MAX_POISSON_LAMBDAS,
              detect_probs=DETECT_PROBS,
              verbose=VERBOSE,
              debug=DEBUG,
              timeit=TIMEIT,
             )
    # run GDM on full communities
    sim.run_GDM(surveys=None,
                gdm_data_type=GDM_DATA_TYPE,
                implementation=gdm_implementation,
                site_survey_filepath = 'sobig_demo_site_survey.csv',
                env_rast_filepath = 'sobig_demo_env_rast.tif',
                delete_intermed_files=True,
               )
    # plot and save results
    fig = sim.plot(scatter_points=False,
                   plot_fenv_input=True,
                   title='before change'*use_env_change,
                   save=False,
                  )
    gs = fig.axes[-1].get_subplotspec().get_gridspec()
    expec_ax = fig.add_subplot(gs[65:80, 67:82])
    obser_ax = fig.add_subplot(gs[65:80, 85:])
    # plot expected vs. observed distribution for random species
    sp = [0]
    for s in sp:
        sim.plot_expec_vs_obser_sp_distr(sp=s,
                                         title=f"before change: sp {s}",
                                         cmap='viridis',
                                         save=False,
                                         expec_ax=expec_ax,
                                         obser_ax=obser_ax,
                                        )
    # deepcopy sim (just in case)
    sim_b4 = deepcopy(sim)
    if use_env_change:
        # update the environment to simulate environmental change, then rerun the
        # same set of GDM results
        increase = nlmpy.mpd(nRow=DIMS[0], nCol=DIMS[1], h=1)*0.8
        ENV[1] = ENV[1] + increase
        sim.update_env(ENV)
        sim.run_GDM(surveys=None,
                    gdm_data_type=GDM_DATA_TYPE,
                    implementation=gdm_implementation,
                    site_survey_filepath = 'sobig_demo_site_survey.csv',
                    env_rast_filepath = 'sobig_demo_env_rast.tif',
                   delete_intermed_files=True,
                   )
        # plot again
        fig = sim.plot(scatter_points=False,
                       plot_fenv_input=True,
                       title='after change',
                       save=False,
                      )
        gs = fig.axes[-1].get_subplotspec().get_gridspec()
        expec_ax = fig.add_subplot(gs[65:80, 67:82])
        obser_ax = fig.add_subplot(gs[65:80, 85:])
        # plot expected vs. observed distribution for random species
        sp = [0]
        for s in sp:
            sim.plot_expec_vs_obser_sp_distr(sp=s,
                                             title=f"before change: sp {s}",
                                             cmap='viridis',
                                             save=False,
                                             expec_ax=expec_ax,
                                             obser_ax=obser_ax,
                                            )
    return sim

