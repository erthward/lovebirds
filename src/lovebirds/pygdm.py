"""
gdm_py.py

NOTE: CODE ENTIRELY PORTED FROM R BY CLAUDE! ONLY USED FOR VIZ!

A self-contained Python port of the *core* functionality of the R package `gdm`
(https://github.com/fitzLab-AL/gdm), for the specific default case:

    - geo = FALSE  (geographic distance is not used as a predictor)
    - I-spline predictor transforms (Ramsay 1988 monotone-spline construction)
    - fit via IRLS with a negative-exponential link and non-negativity-constrained
      spline coefficients (this is a reimplementation of the fitting logic
      described in Ferrier et al. 2007 / the gdm methodology; the original R
      package's fitting engine is implemented in C, so this will NOT reproduce
      its coefficients bit-for-bit -- validate against your R output before
      relying on exact numbers. It should reproduce qualitatively similar
      transform shapes / PCA structure.)

Main entry point: run_GDM()

Dependencies: numpy, pandas, xarray, scipy, scikit-learn
"""

from itertools import combinations
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import BSpline
from scipy.optimize import nnls
from sklearn.decomposition import PCA


# --------------------------------------------------------------------------
# I-spline basis construction (Ramsay 1988 monotone regression splines)
# --------------------------------------------------------------------------

def _build_knot_vector(x_ref, n_basis, degree=3):
    """
    Build a clamped B-spline knot vector such that the resulting B-spline
    basis has exactly `n_basis` functions of the given `degree` (or a
    reduced degree if n_basis is too small for the requested degree), with
    interior knots placed at evenly spaced quantiles of x_ref.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_ref = x_ref[np.isfinite(x_ref)]
    lo, hi = np.min(x_ref), np.max(x_ref)
    if hi <= lo:
        hi = lo + 1e-9

    # A clamped B-spline needs n_basis >= degree + 1; reduce degree if the
    # requested number of basis functions is too small for a cubic.
    degree = min(degree, max(n_basis - 1, 1))

    n_interior = max(n_basis - degree - 1, 0)
    if n_interior > 0:
        qs = np.linspace(0, 1, n_interior + 2)[1:-1]
        interior_knots = np.quantile(x_ref, qs)
    else:
        interior_knots = np.array([])

    t = np.concatenate((
        np.repeat(lo, degree + 1),
        interior_knots,
        np.repeat(hi, degree + 1),
    ))
    return t, lo, hi, degree


def _ispline_basis_matrix(x_eval, t, degree, n_basis, lo, hi):
    """
    Evaluate the I-spline basis (integral of the normalized B-spline / M-spline
    basis) at x_eval, given knot vector t. Returns an (len(x_eval), n_basis)
    matrix, each column monotonically non-decreasing, running from 0 (at/below
    lo) to 1 (at/above hi).
    """
    x_eval = np.asarray(x_eval, dtype=float)
    x_clipped = np.clip(x_eval, lo, hi)

    out = np.zeros((len(x_eval), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        b = BSpline(t, c, degree, extrapolate=False)
        b_int = b.antiderivative()
        denom = t[i + degree + 1] - t[i]
        if denom <= 0:
            continue
        norm = (degree + 1) / denom
        base_at_lo = b_int(lo)
        vals = (b_int(x_clipped) - base_at_lo) * norm
        vals = np.nan_to_num(vals, nan=0.0)
        # I-splines with clamped boundary knots saturate to 1 beyond their
        # support; clip defensively for numerical safety.
        out[:, i] = np.clip(vals, 0.0, 1.0)
    return out


class IsplinePredictor:
    """Holds the fitted knot vector for one environmental predictor and can
    evaluate its I-spline basis at arbitrary values (sites or raster cells)."""

    def __init__(self, site_values, n_basis=3, degree=3):
        self.n_basis = n_basis
        self.t, self.lo, self.hi, self.degree = _build_knot_vector(site_values, n_basis, degree)

    def basis(self, x_eval):
        return _ispline_basis_matrix(x_eval, self.t, self.degree, self.n_basis, self.lo, self.hi)


# --------------------------------------------------------------------------
# Dissimilarity
# --------------------------------------------------------------------------

def _bray_curtis(a, b):
    num = np.sum(np.abs(a - b))
    den = np.sum(a + b)
    if den == 0:
        return 0.0
    return num / den


# --------------------------------------------------------------------------
# IRLS fit with non-negativity-constrained spline coefficients
# --------------------------------------------------------------------------

def _fit_gdm(X_spline, y, max_iter=100, tol=1e-7, inner_iters=5, eps=1e-6):
    """
    Fit  y_hat = 1 - exp(-(intercept + X_spline @ beta)),  beta >= 0,
    intercept unconstrained, via IRLS + block-coordinate NNLS.
    """
    n, p = X_spline.shape
    beta = np.zeros(p)
    intercept = -np.log(1 - np.clip(np.mean(y), 0, 0.999))

    prev_dev = np.inf
    for _ in range(max_iter):
        eta = intercept + X_spline @ beta
        eta = np.clip(eta, 1e-6, 30.0)
        mu = 1 - np.exp(-eta)
        mu = np.clip(mu, eps, 1 - eps)

        dmu_deta = np.exp(-eta)
        dmu_deta = np.clip(dmu_deta, eps, None)
        var = mu * (1 - mu)
        var = np.clip(var, eps, None)

        z = eta + (y - mu) / dmu_deta
        w = (dmu_deta ** 2) / var

        sw = np.sqrt(w)
        for _ in range(inner_iters):
            resid = z - X_spline @ beta
            intercept = np.sum(w * resid) / np.sum(w)
            resid2 = z - intercept
            Xw = X_spline * sw[:, None]
            yw = resid2 * sw
            beta, _ = nnls(Xw, yw)

        eta_new = np.clip(intercept + X_spline @ beta, 1e-6, 30.0)
        mu_new = np.clip(1 - np.exp(-eta_new), eps, 1 - eps)
        dev = -2 * np.sum(y * np.log(mu_new) + (1 - y) * np.log(1 - mu_new))
        if np.abs(prev_dev - dev) < tol:
            break
        prev_dev = dev

    return intercept, beta


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_GDM(site_table, env_raster, abund=True, geo=False, n_splines=3,
            curve_points=200, max_iter=100):
    """
    Port of the core GDM workflow (geo=FALSE case).

    Parameters
    ----------
    site_table : pandas.DataFrame
        Columns: 'site' (int id), 'x', 'y' (lon/lat), 'spp0'..'sppN'
        (abundances or presence/absence).
    env_raster : xarray.DataArray
        3-D raster with one non-spatial dimension (the environmental
        variable/band dimension) and spatial dims named 'x' and 'y'
        (coordinates should be in the same units/CRS as site_table's x/y).
    abund : bool
        If True, spp columns are treated as abundances (Bray-Curtis).
        If False, treated as presence/absence (same formula = Sorensen).
    geo : bool
        Must be False; geographic distance as a predictor is not implemented
        (matches the requested default).
    n_splines : int
        Number of I-spline basis functions per environmental predictor
        (gdm default is 3).
    curve_points : int
        Number of points used to trace each fitted isplineExtract curve.
    max_iter : int
        Max IRLS iterations.

    Returns
    -------
    ispline_df : pandas.DataFrame
        Columns 'x.env_rast_i' / 'y.env_rast_i' for i = 1..n_bands, giving the
        fitted I-spline transform curve for each environmental raster band
        (equivalent of R's isplineExtract()).
    pca_raster : xarray.DataArray
        3-band raster (dim 'component' = PC1, PC2, PC3) of the top 3 principal
        components of the GDM-transformed environmental raster
        (equivalent of gdm.transform() followed by terra::prcomp()).
    """
    if geo:
        raise NotImplementedError("This port only implements the geo=False case.")

    site_table = site_table.reset_index(drop=True)
    spp_cols = sorted(
        [c for c in site_table.columns if c.startswith("spp")],
        key=lambda c: int(c[3:])
    )
    if not spp_cols:
        raise ValueError("No columns starting with 'spp' found in site_table.")

    n_sites = len(site_table)
    bio = site_table[spp_cols].to_numpy(dtype=float)
    if not abund:
        bio = (bio > 0).astype(float)

    # --- identify band dimension of the raster ---
    band_dim = [d for d in env_raster.dims if d not in ("x", "y")]
    if len(band_dim) != 1:
        raise ValueError("env_raster must have exactly one non-'x'/'y' dimension.")
    band_dim = band_dim[0]
    band_coord = env_raster[band_dim].values if band_dim in env_raster.coords else \
        np.arange(env_raster.sizes[band_dim])
    n_bands = env_raster.sizes[band_dim]

    # --- sample raster at site coordinates ---
    site_x = xr.DataArray(site_table["x"].values, dims="site")
    site_y = xr.DataArray(site_table["y"].values, dims="site")
    env_at_sites = env_raster.sel(x=site_x, y=site_y, method="nearest")
    env_at_sites = env_at_sites.transpose(band_dim, "site")
    env_vals = env_at_sites.values  # (n_bands, n_sites)

    # --- build per-predictor I-spline objects ---
    predictors = [
        IsplinePredictor(env_vals[j, :], n_basis=n_splines, degree=3)
        for j in range(n_bands)
    ]

    # --- transform each site's environmental values into I-spline space ---
    # site_basis[j] has shape (n_sites, n_splines)
    site_basis = [predictors[j].basis(env_vals[j, :]) for j in range(n_bands)]

    # --- build pairwise design matrix and response ---
    pairs = list(combinations(range(n_sites), 2))
    y = np.array([_bray_curtis(bio[a], bio[b]) for a, b in pairs])

    X_blocks = []
    for j in range(n_bands):
        sb = site_basis[j]
        diffs = np.abs(sb[[a for a, b in pairs], :] - sb[[b for a, b in pairs], :])
        X_blocks.append(diffs)
    X_spline = np.concatenate(X_blocks, axis=1)  # (n_pairs, n_bands*n_splines)

    # --- fit ---
    intercept, beta = _fit_gdm(X_spline, y, max_iter=max_iter)
    beta_blocks = np.split(beta, n_bands)

    # --- isplineExtract equivalent ---
    ispline_cols = {}
    for j in range(n_bands):
        lo, hi = predictors[j].lo, predictors[j].hi
        xs = np.linspace(lo, hi, curve_points)
        basis_vals = predictors[j].basis(xs)
        ys = basis_vals @ beta_blocks[j]
        ys = ys - ys.min()  # anchor curve at 0, matching isplineExtract convention
        label = j + 1
        ispline_cols[f"x.env_rast_{label}"] = xs
        ispline_cols[f"y.env_rast_{label}"] = ys
    ispline_df = pd.DataFrame(ispline_cols)

    # --- gdm.transform equivalent: apply fitted transforms to full raster ---
    raster_vals = env_raster.transpose(band_dim, "y", "x").values  # (n_bands, ny, nx)
    ny, nx = raster_vals.shape[1], raster_vals.shape[2]
    transformed = np.full((n_bands, ny, nx), np.nan)

    for j in range(n_bands):
        flat = raster_vals[j].ravel()
        valid = np.isfinite(flat)
        out = np.full(flat.shape, np.nan)
        if valid.any():
            basis_vals = predictors[j].basis(flat[valid])
            out[valid] = basis_vals @ beta_blocks[j]
        transformed[j] = out.reshape(ny, nx)

    # --- PCA (terra::prcomp equivalent) on transformed raster stack ---
    flat_stack = transformed.reshape(n_bands, -1).T  # (n_pixels, n_bands)
    valid_mask = np.all(np.isfinite(flat_stack), axis=1)

    pca = PCA(n_components=min(3, n_bands))
    scores_valid = pca.fit_transform(flat_stack[valid_mask])

    n_comp = scores_valid.shape[1]
    scores_full = np.full((flat_stack.shape[0], n_comp), np.nan)
    scores_full[valid_mask] = scores_valid

    pca_array = scores_full.T.reshape(n_comp, ny, nx)
    if n_comp < 3:
        pad = np.full((3 - n_comp, ny, nx), np.nan)
        pca_array = np.concatenate([pca_array, pad], axis=0)

    pca_raster = xr.DataArray(
        pca_array[:3],
        dims=("component", "y", "x"),
        coords={
            "component": ["PC1", "PC2", "PC3"],
            "y": env_raster["y"].values,
            "x": env_raster["x"].values,
        },
        name="gdm_pca",
    )

    return ispline_df, pca_raster
