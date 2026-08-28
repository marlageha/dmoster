import numpy as np
import emcee
from scipy.optimize import curve_fit

from dmost import dmost_EW, dmost_coadd_emcee

########################################################################
# CaT LINE-PROFILE MODELS: PRODUCTION'S Gauss+Lorentzian-SUM MODEL, WITH
# THE LORENTZIAN WIDTH ALLOWED TO VARY SLIGHTLY PER LINE (gamma-varies,
# "GL-gvary"). PHYSICAL MOTIVATION: THE GAUSSIAN WIDTH (THERMAL DOPPLER +
# INSTRUMENTAL) IS SHARED/FIXED ACROSS THE 3 LINES, BUT THE LORENTZIAN
# WIDTH (RADIATIVE/PRESSURE DAMPING) IS SET BY EACH TRANSITION'S OWN
# DAMPING CONSTANT, SO IT IS ALLOWED A SMALL PER-LINE DEVIATION FROM THE
# 8542 ANCHOR VIA A TIGHT HIERARCHICAL PRIOR. KEEPS TWO INDEPENDENT DEPTH
# PARAMETERS PER LINE (GAUSSIAN DEPTH + LORENTZIAN DEPTH).
#
# ALSO INCLUDES THE DECOUPLED-WIDTH MODEL: EW1 (8498) GETS ITS OWN
# GAUSSIAN WIDTH INSTEAD OF SHARING p2 WITH EW2/EW3, VIA ITS OWN WIDTH
# DEVIATION PARAMETERS (ds1/dg1). p2/p3 ARE FREE PARAMETERS IN THE SAME
# JOINT emcee FIT AS EVERYTHING ELSE (STAGE 1, "SOFT-ANCHOR" DESIGN,
# ADOPTED 2026-08-27 -- SEE fit_decoupled_stage BELOW), GOVERNED BY AN
# INFORMATIVE PRIOR CENTERED ON A CHEAP EW2+EW3-ONLY MAP PRE-FIT RATHER
# THAN HELD FIXED AT THAT PRE-FIT'S POINT ESTIMATE (THE ORIGINAL, NOW
# SUPERSEDED, TWO-STEP DESIGN). WHY THE PRIOR EXISTS AT ALL: EW1'S
# SHARED-WIDTH FIT WAS TOO BROAD (TUNED BY EW2/EW3, WHICH ARE WIDER
# LINES); A FULLY FREE ds1 WITH NO ANCHOR INFORMATION LET p2 DRIFT AND
# HURT EW2/EW3 TOO. THE SOFT PRIOR (RATHER THAN A HARD FIX) LETS p2/p3'S
# OWN POSTERIOR UNCERTAINTY PROPAGATE INTO THE REPORTED cat_err.
#
# THIS IS THE dmoster PRODUCTION VERSION OF THE gl_gvary_test_flatprior.py
# RESEARCH MODULE (CaT_GL) -- SEE THAT FILE'S HISTORY FOR THE FULL
# DEVELOPMENT/VALIDATION NARRATIVE. THE STANDALONE COMPARISON DRIVER
# (__main__ BLOCK) WAS DROPPED HERE; ONLY THE FITTING API IS KEPT.
########################################################################

CENTER = dmost_EW.CAT_LINE_CENTER          # 8542.09
R1, R3 = 0.994841, 1.01405
GAMMA_DEV_SCALE = 0.05     # gamma_dev_scale_sweep.py: HIGH-S/N chi2 IMPROVEMENT WAS FLAT
                            # ACROSS SCALE (0.02-0.15), WHILE EW BIAS AT S/N>=100 NEARLY
                            # DOUBLED (0.076 -> 0.141 AA) FOR NO chi2 BENEFIT -- 0.05 ADOPTED
GAMMA_DEV_HARDCAP = 1.0

# HIERARCHICAL EW-RATIO PRIOR SCATTER
EW1_SCATTER = 0.2
EW3_SCATTER = 0.2

# FLAT, ZERO-INTERCEPT EW RATIOS (MEASURED DIRECTLY FROM MODEL-FREE DIRECT
# INTEGRATION AT S/N>25, n=10894 SLITS) -- REPLACES THE HEIGER+24
# SLOPE+INTERCEPT RELATION (EW1=0.41*EW2+0.14, EW3=0.74*EW2+0.16). THE
# LOW/HIGH EW2 SPLIT TEST FOUND BOTH GROUPS CONVERGE TO THE SAME HIGH-S/N
# RATIO REGARDLESS OF EW2, IN TENSION WITH THE INTERCEPT TERM.
FLAT_EW1_OVER_EW2 = 0.3958
FLAT_EW3_OVER_EW2 = 1.0 / 1.2876   # i.e. EW2/EW3 = 1.2876


def pct_err(x):
    p16, p50, p84 = np.percentile(x, [15.8, 50, 84])
    return p50, (p84-p16)/2.


########################################################################
# DECOUPLED-WIDTH MODEL: EW1 (8498) GETS ITS OWN GAUSSIAN WIDTH INSTEAD OF
# SHARING p2 WITH EW2/EW3. SEE MODULE DOCSTRING ABOVE.
########################################################################
GAMMA_DEV_SCALE_1 = 0.15       # dg1 prior scale (looser than dg3's 0.05)
SIGMA_DEV_HARDCAP = 1.5        # ds1 hard cap
SIGMA_DEV_SCALE_1 = 0.5        # ds1 prior scale
ANCHOR_MAX = 4.0                # p2/p3 upper bound in the anchor fit
SIGMA1_ABS_MAX = 3.0            # EW1's own width can't exceed this
GAMMA1_ABS_MAX = 3.0            # EW1's own Lorentz width can't exceed this
RATIO_MIN = 0.15                # gamma/sigma floor (per line)
RATIO_LOGNORMAL_MU, RATIO_LOGNORMAL_SIGMA = 0.0, 0.5   # push away from RATIO_MIN


def _ln_ratio_prior(ratio):
    # keeps gamma/sigma away from 0: hard floor + push toward ~1
    if ratio < RATIO_MIN:
        return -np.inf
    return dmost_EW._ln_lognormal(ratio, RATIO_LOGNORMAL_MU, RATIO_LOGNORMAL_SIGMA)


def anchor_model(x, p0, p1, p2, p3, dg3, p5, p6, p8, p9):
    # EW2 (anchor) + EW3 only -- no EW1 term
    norm = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p5*norm*np.exp(-0.5*((x-p1)/p2)**2) + \
            p6*norm*np.exp(-0.5*((x-p1*R3)/p2)**2)
    gamma3 = p3*np.exp(dg3)
    norm2_2, norm2_3 = p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p8*norm2_2/((x-p1)   **2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-p1*R3)**2 + (gamma3/2.)**2))
    return p0 * (1. - gauss - lorentz)


def anchor_lnprior(theta, center_margin=1.0):
    p0, p1, p2, p3, dg3, p5, p6, p8, p9 = theta

    if p0 <= 0:
        return -np.inf
    if (p1 < CENTER-center_margin) | (p1 > CENTER+center_margin):
        return -np.inf
    if (p2 < 0.4) | (p2 > ANCHOR_MAX):
        return -np.inf
    if (p3 < 0.4) | (p3 > ANCHOR_MAX):
        return -np.inf
    if abs(dg3) > GAMMA_DEV_HARDCAP:
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p0, 1.0, 0.1)
    lnp += dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_lognormal(p2, 0.0, 0.5)
    lnp += dmost_EW._ln_lognormal(p3, 0.0, 0.5)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, GAMMA_DEV_SCALE)
    lnp += _ln_ratio_prior(p3/p2)
    if not np.isfinite(lnp):
        return -np.inf

    if (p5 < 0) | (p6 < 0) | (p8 < 0) | (p9 < 0):
        return -np.inf
    EW2, EW3 = p5+p8, p6+p9
    if (EW2 <= 0) | (EW3 <= 0):
        return -np.inf

    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_gauss(EW3, FLAT_EW3_OVER_EW2*EW2, EW3_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p8, EW2)
    lnp += dmost_EW._ln_beta_frac(p9, EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def anchor_lnlike(theta, wvl, spec, ivar, mw23):
    model = anchor_model(wvl[mw23], *theta)
    return -0.5*np.sum((spec[mw23]-model)**2 * ivar[mw23])


def fit_anchor_stage1(wvl, spec, ivar, mw23, center_margin=1.0):
    '''EW2+EW3-only MAP anchor fit (point estimate only). Thin wrapper
    around fit_map_anchor_with_sigma (below), which does the identical
    fit and also returns the Hessian-derived uncertainty needed by
    Stage 1's soft-anchor prior -- this entry point is for callers (the
    missing-coverage path) that only need the point estimate.'''
    p, sigma = fit_map_anchor_with_sigma(wvl, spec, ivar, mw23, center_margin)
    return p   # p0,p1,p2,p3,dg3,p5,p6,p8,p9


# p2/p3 for the functions below are set once per slit (as module globals,
# read by CaT_GL_gvary_decoupled1) by fit_decoupled_stage and run_emcee_reduced
P2_FIXED, P3_FIXED = None, None


def CaT_GL_gvary_decoupled1(x, *p):
    # p0,p1,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9 -- p2,p3 are the fixed anchor
    p0, p1, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9 = p
    p2, p3 = P2_FIXED, P3_FIXED

    sigma1 = p2 * np.exp(ds1)   # EW1's own Gaussian width
    norm1 = 1./(np.sqrt(2*np.pi) * sigma1)
    norm  = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p4*norm1*np.exp(-0.5*((x-p1*R1)/sigma1)**2) + \
            p5*norm *np.exp(-0.5*((x-p1)   /p2   )**2) + \
            p6*norm *np.exp(-0.5*((x-p1*R3)/p2   )**2)

    gamma1, gamma3 = p3*np.exp(dg1), p3*np.exp(dg3)
    norm2_1, norm2_2, norm2_3 = gamma1/(2.*np.pi), p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p7*norm2_1/((x-p1*R1)**2 + (gamma1/2.)**2)) + \
              (p8*norm2_2/((x-p1)   **2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-p1*R3)**2 + (gamma3/2.)**2))

    return p0 * (1. - gauss - lorentz)


########################################################################
# p0 (LOCAL CONTINUUM LEVEL) FIXED AT 1.0 -- ADOPTED 2026-08-24, REPLACES
# FREE p0 AS THE PRODUCTION DEFAULT (fit_decoupled_stage BELOW).
#
# RATIONALE: p0 was previously a free parameter fit JOINTLY with the line
# shape, constrained only by off-line pixels WITHIN the 3 CaT windows
# (mw1|mw23), NOT by the actual continuum bands used in the upstream
# whole-spectrum normalization (dmost_continuum.CaII_normalize_weighted_
# looflag). Posterior-chain check (CaT_GL_syserr_Feh research_log_2026-
# 08-24.html) found: (a) the posterior std(p0) is ~60x tighter than its
# own prior (data-dominated, prior does ~nothing), but (b) a real, if
# weak, correlation exists between p0 and total EW in the joint posterior
# (corr ~0.2-0.4) -- a genuine degeneracy, just a narrow one. A 50-slit
# ablation (fixing p0=1.0 vs free, same normalized spectrum both ways)
# found: median |Delta CaT| = 0.079 A (not negligible), corr(Delta CaT,
# 1-p0_free) = 0.73 (the effect tracks tightly with how far the free fit
# pulled p0 from 1), and chi2 ESSENTIALLY UNCHANGED (median 1.83 both
# ways) -- i.e. letting p0 float bought ~no fit-quality improvement while
# introducing this correlated wobble. Decision: trust the upstream
# continuum normalization (which already does the weighted, bad-band-
# dropping fit in the correct bands) completely, and fix p0=1.0 during
# line fitting instead of re-deriving a second, locally-different
# estimate of it inside each CaT window.
#
# CaT_GL_gvary_decoupled1 (the general 11-param model, p0 free) is KEPT
# UNCHANGED above -- reconstructing/plotting a stored cat_theta (this run
# or any legacy pre-2026-08-24 run) must keep working exactly as before.
# Only the FIT ITSELF (curve_fit seed, emcee dimensionality, prior,
# likelihood) drops p0 as a free parameter; the returned theta13 still
# has p0 in slot 0 for layout compatibility, just always =1.0.
########################################################################

def CaT_GL_gvary_decoupled1_fixedp0(x, p1, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9):
    return CaT_GL_gvary_decoupled1(x, 1.0, p1, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9)


########################################################################
# STAGE 1 FIT -- SOFT-ANCHOR DESIGN, ADOPTED 2026-08-27, REPLACES THE
# ORIGINAL TWO-STEP DESIGN (anchor MAP fit p2/p3, THEN FREEZE THEM AS
# CONSTANTS for a separate full emcee fit). See research_log_2026-08-26
# .html and CaT_GL_syserr_Feh/status.md for the full development
# narrative. Two real problems with the two-step design motivated this:
# (1) p2/p3's own uncertainty never reached the reported cat_err, since
# they were never sampled at all; (2) it's a needlessly complex Methods
# description for a paper (two separate fitting procedures chained
# together) when it doesn't have to be.
#
# NEW DESIGN: p2/p3 are free parameters in ONE joint emcee pass, same as
# everything else -- but with an INFORMATIVE Gaussian prior centered on
# the EW2+EW3-only anchor MAP point (fit_anchor_stage1, UNCHANGED, still
# used, still a cheap deterministic pre-fit -- just no longer freezes
# its result). Prior width comes from that MAP fit's own inverse-Hessian
# diagonal, floored at 0.08 and CAPPED at 0.4 -- the cap matters: the
# raw Hessian-derived width often came out ~1.0-1.8 (near-uninformative
# over the [0.4,4.0] range), letting p2/p3 drift wider than the old
# two-step result at moderate S/N (visually confirmed too-broad, shallow
# line wings undershooting real narrow features in 2 flagged slits,
# fixed by the cap with no regression elsewhere).
#
# A second small MAP fit, restricted to the EW1 window alone with p2/p3
# held at the anchor's point estimate, fills in the emcee seed for
# EW1's own ds1/dg1/depths (the anchor step never touches EW1 at all).
# This replaced an earlier curve_fit-based seed: curve_fit is
# unconstrained least-squares and could land on a start point where
# every parameter individually satisfies its own bound but the
# COMBINATION still violates the model's joint constraint
# (p2*exp(ds1)<=SIGMA1_ABS_MAX), stalling the sampler outright (0%
# acceptance) on some slits.
#
# Reported theta/curve/chi2 come from the single highest-log-probability
# sample in the post-burnin chain, not per-parameter marginal medians --
# for correlated parameters (p2/p3/ds1/dg1 all trade off against each
# other), an independently-chosen marginal-median composite can land
# somewhere the sampler never actually visited, violating the model's
# own hard constraints (confirmed: produced a visibly wrong, 3.6x-over-
# cap gamma1 on one slit before this fix).
########################################################################

SOFTANCHOR_SIGMA_FLOOR = 0.08
SOFTANCHOR_SIGMA_CAP = 0.4


def fit_map_anchor_with_sigma(wvl, spec, ivar, mw23, center_margin=1.0):
    '''MAP anchor fit (EW2+EW3 only), also returning the local inverse-
    Hessian-derived uncertainty on p2/p3 -- used to set the Stage 1
    soft-anchor prior width below. fit_anchor_stage1 (the missing-
    coverage path's entry point, which only needs the point estimate)
    wraps this and discards sigma.'''
    from scipy.optimize import minimize

    def neg_lnpost(theta):
        lp = anchor_lnprior(theta, center_margin)
        if not np.isfinite(lp):
            return 1e10
        ll = anchor_lnlike(theta, wvl, spec, ivar, mw23)
        return -(lp + ll) if np.isfinite(ll) else 1e10

    seed = [1.0, CENTER, 1.5, 1.2, 0.0, 0.3, 0.3, 1.5, 1.5]
    bounds = [(0.8, 1.2), (CENTER-center_margin, CENTER+center_margin), (0.4, ANCHOR_MAX), (0.4, ANCHOR_MAX),
              (-GAMMA_DEV_HARDCAP, GAMMA_DEV_HARDCAP), (0.05, 4.0), (0.05, 4.0), (0.0, 5.0), (0.0, 5.0)]
    res = minimize(neg_lnpost, seed, bounds=bounds, method='L-BFGS-B')

    try:
        hess_diag = np.diag(res.hess_inv.todense())
        sigma = np.sqrt(np.clip(hess_diag, SOFTANCHOR_SIGMA_FLOOR**2, SOFTANCHOR_SIGMA_CAP**2))
    except Exception:
        sigma = np.full(9, SOFTANCHOR_SIGMA_FLOOR)
    return res.x, sigma


def _ew1only_model(x, p1, ds1, dg1, p4, p7, p2_fixed, p3_fixed):
    sigma1 = p2_fixed * np.exp(ds1)
    gamma1 = p3_fixed * np.exp(dg1)
    norm1 = 1./(np.sqrt(2*np.pi) * sigma1)
    gauss1 = p4*norm1*np.exp(-0.5*((x-p1*R1)/sigma1)**2)
    norm2_1 = gamma1/(2.*np.pi)
    lorentz1 = p7*norm2_1/((x-p1*R1)**2 + (gamma1/2.)**2)
    return 1.0 * (1. - gauss1 - lorentz1)


def _lnprior_ew1only(theta, center_margin, p2_fixed, p3_fixed, EW2_anchor):
    p1, ds1, dg1, p4, p7 = theta
    if (p1 < CENTER-center_margin) | (p1 > CENTER+center_margin):
        return -np.inf
    if abs(dg1) > GAMMA_DEV_HARDCAP:
        return -np.inf
    if abs(ds1) > SIGMA_DEV_HARDCAP:
        return -np.inf
    if p2_fixed * np.exp(ds1) > SIGMA1_ABS_MAX:
        return -np.inf
    if p3_fixed * np.exp(dg1) > GAMMA1_ABS_MAX:
        return -np.inf
    if (p4 < 0) | (p7 < 0):
        return -np.inf
    EW1 = p4 + p7
    if EW1 <= 0:
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE_1)
    lnp += dmost_EW._ln_gauss(ds1, 0.0, SIGMA_DEV_SCALE_1)
    ratio1 = (p3_fixed*np.exp(dg1)) / (p2_fixed*np.exp(ds1))
    lnp += _ln_ratio_prior(ratio1)
    lnp += dmost_EW._ln_gauss(EW1, FLAT_EW1_OVER_EW2*EW2_anchor, EW1_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p7, EW1)
    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def fit_map_ew1(wvl, spec, ivar, mw1, p2_fixed, p3_fixed, EW2_anchor, center_margin=1.0):
    '''MAP fit restricted to the EW1 window alone, p2/p3 held at the
    anchor's point estimate -- fills in the Stage 1 emcee seed for
    ds1/dg1/EW1's two depths, the parameters fit_anchor_stage1 never
    touches. Same "MAP, not curve_fit" reasoning as the anchor step
    itself: respects the model's joint constraints by construction.'''
    from scipy.optimize import minimize

    seed = [CENTER, -0.5, -0.3, 0.2, 0.2]   # p1,ds1,dg1,p4,p7
    ds1_cap = min(SIGMA_DEV_HARDCAP, np.log(SIGMA1_ABS_MAX/p2_fixed))
    dg1_floor = np.log(RATIO_MIN * p2_fixed * np.exp(seed[1]) / p3_fixed)
    dg1_cap = min(GAMMA_DEV_HARDCAP, np.log(GAMMA1_ABS_MAX/p3_fixed))
    bounds = [(CENTER-center_margin, CENTER+center_margin), (-SIGMA_DEV_HARDCAP, ds1_cap),
              (max(-GAMMA_DEV_HARDCAP, dg1_floor), dg1_cap), (0.0, 5.0), (0.0, 5.0)]

    def neg_lnpost(theta):
        lp = _lnprior_ew1only(theta, center_margin, p2_fixed, p3_fixed, EW2_anchor)
        if not np.isfinite(lp):
            return 1e10
        model = _ew1only_model(wvl[mw1], *theta, p2_fixed, p3_fixed)
        ll = -0.5*np.sum((spec[mw1]-model)**2 * ivar[mw1])
        return -(lp + ll) if np.isfinite(ll) else 1e10

    res = minimize(neg_lnpost, seed, bounds=bounds, method='L-BFGS-B')
    return res.x   # p1, ds1, dg1, p4, p7


def _lnprior_softanchor(theta, center_margin, p2_map, sigma_p2, p3_map, sigma_p3):
    p1, p2, p3, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9 = theta

    if (p1 < CENTER-center_margin) | (p1 > CENTER+center_margin):
        return -np.inf
    if (p2 < 0.4) | (p2 > ANCHOR_MAX) | (p3 < 0.4) | (p3 > ANCHOR_MAX):
        return -np.inf
    if (abs(dg1) > GAMMA_DEV_HARDCAP) | (abs(dg3) > GAMMA_DEV_HARDCAP):
        return -np.inf
    if abs(ds1) > SIGMA_DEV_HARDCAP:
        return -np.inf
    if p2 * np.exp(ds1) > SIGMA1_ABS_MAX:
        return -np.inf
    if p3 * np.exp(dg1) > GAMMA1_ABS_MAX:
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_gauss(p2, p2_map, sigma_p2)
    lnp += dmost_EW._ln_gauss(p3, p3_map, sigma_p3)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE_1)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, GAMMA_DEV_SCALE)
    lnp += dmost_EW._ln_gauss(ds1, 0.0, SIGMA_DEV_SCALE_1)

    ratio1 = (p3*np.exp(dg1)) / (p2*np.exp(ds1))
    ratio3 = (p3*np.exp(dg3)) / p2
    lnp += _ln_ratio_prior(ratio1)
    lnp += _ln_ratio_prior(ratio3)
    if not np.isfinite(lnp):
        return -np.inf

    depths = np.array([p4, p5, p6, p7, p8, p9])
    if np.any(depths < 0):
        return -np.inf
    EW1, EW2, EW3 = p4+p7, p5+p8, p6+p9
    if (EW1 <= 0) | (EW2 <= 0) | (EW3 <= 0):
        return -np.inf

    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_gauss(EW1, FLAT_EW1_OVER_EW2*EW2, EW1_SCATTER)
    lnp += dmost_EW._ln_gauss(EW3, FLAT_EW3_OVER_EW2*EW2, EW3_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p7, EW1)
    lnp += dmost_EW._ln_beta_frac(p8, EW2)
    lnp += dmost_EW._ln_beta_frac(p9, EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def _model_softanchor(x, theta):
    # theta = p1,p2,p3,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9 -- p2/p3 explicit
    # here (unlike CaT_GL_gvary_decoupled1_fixedp0, which takes them via
    # the P2_FIXED/P3_FIXED globals) since they're free parameters now
    p1, p2, p3, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9 = theta
    sigma1 = p2 * np.exp(ds1)
    norm1 = 1./(np.sqrt(2*np.pi) * sigma1)
    norm  = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p4*norm1*np.exp(-0.5*((x-p1*R1)/sigma1)**2) + \
            p5*norm *np.exp(-0.5*((x-p1)   /p2   )**2) + \
            p6*norm *np.exp(-0.5*((x-p1*R3)/p2   )**2)
    gamma1, gamma3 = p3*np.exp(dg1), p3*np.exp(dg3)
    norm2_1, norm2_2, norm2_3 = gamma1/(2.*np.pi), p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p7*norm2_1/((x-p1*R1)**2 + (gamma1/2.)**2)) + \
              (p8*norm2_2/((x-p1)   **2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-p1*R3)**2 + (gamma3/2.)**2))
    return 1.0 * (1. - gauss - lorentz)


def _lnlike_softanchor(theta, wvl, spec, ivar, mw):
    model = _model_softanchor(wvl[mw], theta)
    return -0.5*np.sum((spec[mw]-model)**2 * ivar[mw])


def _lnprob_softanchor(theta, wvl, spec, ivar, mw, center_margin, p2_map, sigma_p2, p3_map, sigma_p3):
    lp = _lnprior_softanchor(theta, center_margin, p2_map, sigma_p2, p3_map, sigma_p3)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike_softanchor(theta, wvl, spec, ivar, mw)


def _clip_joint_softanchor(theta, center_margin=1.0, margin=0.02):
    '''Vectorized clip for the full 12-param Stage 1 model, rows=
    (candidate) walkers. Unlike a plain per-parameter np.clip, this
    respects the model's JOINT constraints (p2*exp(ds1)<=SIGMA1_ABS_MAX,
    p3*exp(dg1)<=GAMMA1_ABS_MAX, gamma/sigma ratio floors) -- needed
    because a plain per-parameter clip can leave individually in-bounds
    values that combine into a joint-constraint violation (curve_fit's
    own unconstrained result once put ds1 at its box bound with p2=4.0,
    giving p2*exp(ds1)=17.5, 6x over SIGMA1_ABS_MAX, undetected by
    per-parameter clipping alone -- part of why curve_fit seeding was
    dropped in favor of the EW1-only MAP above).'''
    theta = np.array(theta, dtype=float)
    theta[:, 0] = np.clip(theta[:, 0], CENTER-center_margin+margin, CENTER+center_margin-margin)
    theta[:, 1] = np.clip(theta[:, 1], 0.4+margin, ANCHOR_MAX-margin)
    theta[:, 2] = np.clip(theta[:, 2], 0.4+margin, ANCHOR_MAX-margin)
    p2, p3 = theta[:, 1], theta[:, 2]

    ds1_cap = np.minimum(SIGMA_DEV_HARDCAP-margin, np.log(SIGMA1_ABS_MAX/p2)-margin)
    theta[:, 5] = np.clip(theta[:, 5], -SIGMA_DEV_HARDCAP+margin, ds1_cap)
    ds1 = theta[:, 5]

    dg1_floor = np.log(RATIO_MIN * p2 * np.exp(ds1) / p3) + margin
    dg1_cap = np.minimum(GAMMA_DEV_HARDCAP-margin, np.log(GAMMA1_ABS_MAX/p3)-margin)
    theta[:, 3] = np.clip(theta[:, 3], np.maximum(-GAMMA_DEV_HARDCAP+margin, dg1_floor), dg1_cap)

    dg3_floor = np.log(RATIO_MIN * p2 / p3) + margin
    theta[:, 4] = np.clip(theta[:, 4], np.maximum(-GAMMA_DEV_HARDCAP+margin, dg3_floor), GAMMA_DEV_HARDCAP-margin)

    theta[:, 6:12] = np.clip(theta[:, 6:12], 0.05, None)
    return theta


def fit_decoupled_stage(wvl, spec, ivar, mw1, mw23, center_margin=1.0, max_n=3000):
    '''Stage 1 fit: one joint emcee pass over p1,p2,p3,dg1,dg3,ds1 + 6
    depths (EW1 stays decoupled via ds1/dg1). p2/p3 are free, governed
    by an informative prior from the EW2+EW3 anchor MAP fit (see module
    docstring above) rather than held fixed. Returns the same theta13
    layout as before (p0,p1,p2,p3,dg1,dg3,p4..p9,ds1) for compatibility
    with reconstruction code and the missing-coverage path.'''
    mw = mw1 | mw23
    anchor_x, anchor_sigma = fit_map_anchor_with_sigma(wvl, spec, ivar, mw23, center_margin)
    p2_map, p3_map = anchor_x[2], anchor_x[3]
    sigma_p2, sigma_p3 = anchor_sigma[2], anchor_sigma[3]
    p1_a, dg3_a, p5_a, p6_a, p8_a, p9_a = anchor_x[1], anchor_x[4], anchor_x[5], anchor_x[6], anchor_x[7], anchor_x[8]
    EW2_anchor = p5_a + p8_a

    ew1_x = fit_map_ew1(wvl, spec, ivar, mw1, p2_map, p3_map, EW2_anchor, center_margin)
    ds1_e, dg1_e, p4_e, p7_e = ew1_x[1], ew1_x[2], ew1_x[3], ew1_x[4]

    guess = np.array([p1_a, p2_map, p3_map, dg1_e, dg3_a, ds1_e, p4_e, p5_a, p6_a, p7_e, p8_a, p9_a])
    guess = _clip_joint_softanchor(guess[None, :], center_margin)[0]

    ndim, nwalkers = 12, 32
    jitter = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.03,0.03,0.03, 0.03,0.03,0.03])
    p0w = guess + jitter*np.random.randn(nwalkers, ndim)
    p0w[:, [1,2,6,7,8,9,10,11]] = np.abs(p0w[:, [1,2,6,7,8,9,10,11]])
    p0w = _clip_joint_softanchor(p0w, center_margin)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob_softanchor,
                                     args=(wvl, spec, ivar, mw, center_margin, p2_map, sigma_p2, p3_map, sigma_p3))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0w, max_n)
    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    lnprob_chain = sampler.lnprobability[:, burnin:].reshape(-1)

    theta_best = chain[np.argmax(lnprob_chain)]
    p1,p2,p3,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9 = theta_best

    EW1 = chain[:,6] + chain[:,9]
    EW2 = chain[:,7] + chain[:,10]
    EW3 = chain[:,8] + chain[:,11]

    theta13 = np.array([1.0,p1,p2,p3,dg1,dg3,p4,p5,p6,p7,p8,p9,ds1])
    global P2_FIXED, P3_FIXED
    P2_FIXED, P3_FIXED = p2, p3   # CaT_GL_gvary_decoupled1 reads p2/p3 from these
    fit = CaT_GL_gvary_decoupled1(wvl, 1.0,p1,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9)
    chi2 = dmost_EW.calc_chi2_ew(wvl, spec, ivar, mw, fit)

    return {
        'theta_med': theta_best, 'theta13': theta13, 'p2_fixed': p2, 'p3_fixed': p3,
        'cat': pct_err(EW1+EW2+EW3),
        'ew1': pct_err(EW1), 'ew2': pct_err(EW2), 'ew3': pct_err(EW3),
        'facc': facc, 'convg': convg, 'burnin': burnin,
        'fit': fit, 'chi2': chi2,
    }




########################################################################
# MISSING-LINE-COVERAGE HANDLING -- ADOPTED 2026-08-24, CONSOLIDATED
# 2026-08-26.
#
# SOME SLITS HAVE NO USABLE PIXELS IN THE EW1 (8498) OR EW3 (8662) WINDOW
# AT ALL (VIGNETTING/DEAD-CHIP EDGE, OVERWHELMINGLY THE RED EDGE --
# FULL-DATABASE SCAN OF THE 677-MASK RUN: 1032 SLITS MISSING EW3, 21
# MISSING EW1, 2 MISSING BOTH, OUT OF THE QUALITY-CUT SAMPLE --
# missing_ew_coverage_scan.csv, CaT_GL_syserr_Feh research_log_2026-08-
# 24.html). LETTING THE NORMAL FIT SAMPLE THAT LINE'S OWN DEPTH/WIDTH
# PARAMETERS ANYWAY MEANS THOSE PARAMETERS ARE PURE PRIOR (NO DATA TO
# CONSTRAIN THEM), WHICH PRODUCES GENUINE MCMC MULTIMODALITY/INSTABILITY
# (CONFIRMED ON TWO REAL SLITS BY MULTI-SEED RERUNS -- NOT A SMALL-NUMBER
# EFFECT, A REAL SAMPLER PATHOLOGY THAT CAN HIT EITHER OF TWO DIFFERENT
# FIT CONFIGURATIONS UNPREDICTABLY).
#
# FIX: DETECT PER-LINE COVERAGE (>=90% OF EXPECTED PIXELS IN THE WINDOW
# HAVE USABLE ivar -- SAME CONVENTION AS dmost_continuum's
# MIN_COVERAGE_FRAC), DROP THE MISSING LINE'S OWN PARAMETERS FROM THE
# SAMPLED VECTOR ENTIRELY (A GENUINELY REDUCED-DIMENSION FIT, NOT JUST A
# WIDE PRIOR ON AN UNCONSTRAINED PARAMETER), THEN ANALYTICALLY SUBSTITUTE
# ITS EW FROM THE SAME FLAT EW-RATIO PRIOR USED EVERYWHERE ELSE IN THIS
# MODEL, CONDITIONED ON THE FITTED EW2 POSTERIOR: PER POSTERIOR SAMPLE,
# EW_missing = ratio*EW2 + N(0, intrinsic_scatter) -- SO BOTH THE EW2
# POSTERIOR WIDTH AND THE INTRINSIC RATIO SCATTER PROPAGATE INTO THE
# REPORTED ERROR THROUGH THE USUAL pct_err PERCENTILE SPREAD, NO SEPARATE
# QUADRATURE BOOKKEEPING NEEDED. VALIDATED ON THE SAME TWO SLITS: CLEAN,
# DETERMINISTIC VALUES REPLACING THE UNSTABLE MCMC-DERIVED ONES.
#
# A SLIT FLAGGED HERE IS TERMINAL AT "STAGE 0 + SUBSTITUTION" -- IT NEVER
# ESCALATES TO STAGE A/B (dmost_cat_fit.fit_adaptive_GL_gvary). STAGE A
# ONLY WIDENS THE CENTER PRIOR AND STAGE B ADDS FREE PER-LINE CENTERS --
# NEITHER HELPS A LINE WITH ZERO DATA PIXELS, AND THE r23/anchor_stuck
# ESCALATION TRIGGERS ARE MEANINGLESS HERE ANYWAY (EW3/EW1 IS BY
# CONSTRUCTION PINNED TO THE FIXED PRIOR RATIO, NOT A DATA-DRIVEN
# MEASUREMENT THAT COULD LOOK ANOMALOUS).
#
# ONE GENERIC "REDUCED" MODEL REPLACES THE ORIGINAL THREE HAND-DUPLICATED
# missing1/missing3/missingboth IMPLEMENTATIONS (2026-08-26 cleanup, no
# behavior change -- mechanically verified against the three originals
# param-for-param, bound-for-bound, seed-for-seed). A missing line's
# contribution to CaT_GL_gvary_decoupled1_fixedp0 is IDENTICALLY ZERO
# once its own depth parameters are held at zero (the Gaussian/Lorentzian
# terms are linear in depth) -- so "drop that line's free parameters" is
# just "evaluate the same 10-parameter model with those slots pinned to
# 0", not a reason to hand-write a separate model per missing-line case.
########################################################################

EW1_WIN = (8484., 8513.)
EW2_WIN = (8522., 8562.)
EW3_WIN = (8642., 8682.)
MIN_LINE_COVERAGE = 0.90


def _line_coverage_frac(nwave, nivar, wlo, whi):
    dwave = float(np.median(np.diff(np.sort(nwave))))
    expected = (whi - wlo) / dwave
    actual = np.sum((nwave > wlo) & (nwave < whi) & (nivar > 0))
    return actual / expected if expected > 0 else 0.0


def check_missing_lines(nwave, nivar):
    '''>=90% expected-pixel coverage required in each CaT window (same
    threshold convention as dmost_continuum.MIN_COVERAGE_FRAC), else that
    line has no usable data to constrain its own fit parameters at all.
    Returns (missing1, missing3) booleans. Pairs with check_ew2_detected
    below -- this function alone doesn't check EW2's own coverage, since
    EW2 is always assumed present as the substitution anchor.'''
    return (_line_coverage_frac(nwave, nivar, *EW1_WIN) < MIN_LINE_COVERAGE,
            _line_coverage_frac(nwave, nivar, *EW3_WIN) < MIN_LINE_COVERAGE)


def check_ew2_detected(nwave, nivar):
    '''EW2 (8542, the CaT anchor line) must itself clear the same 90%
    coverage bar as EW1/EW3 before it can be used as the substitution
    anchor when EW1 and/or EW3 are missing (dmost_cat_fit.py). Found
    2026-08-27: a slit can have missing1/missing3 both True with EW2
    ALSO essentially uncovered (a very-low-S/N slit can have zero real
    data anywhere), or missing3 alone True with EW2 only ~40% covered --
    in both real cases (N147_5, S/N~2-3), the reduced model still ran,
    conditioning the substituted EW(s) on an "EW2 posterior" that was
    really just the prior with no data behind it, producing nonsense
    (cat=40 A in one case). check_missing_lines never checked this since
    EW2 was assumed always present as the anchor -- it usually is, but
    not always at extreme low S/N.'''
    return _line_coverage_frac(nwave, nivar, *EW2_WIN) >= MIN_LINE_COVERAGE


# full decoupled1_fixedp0 parameter order (p0 excluded -- always 1.0)
_FULL_NAMES = ['p1', 'dg1', 'dg3', 'ds1', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']


def _active_param_names(missing1, missing3):
    '''Which of the 10 decoupled1_fixedp0 params stay free when a line's
    coverage is missing: EW1's shape (dg1, ds1, p4, p7) drops out if
    missing1, EW3's (dg3, p6, p9) drops out if missing3. EW2 (p5, p8) is
    always free -- check_missing_lines never flags it.'''
    drop = set()
    if missing1:
        drop |= {'dg1', 'ds1', 'p4', 'p7'}
    if missing3:
        drop |= {'dg3', 'p6', 'p9'}
    return [n for n in _FULL_NAMES if n not in drop]


def CaT_GL_gvary_reduced(x, active, *vals):
    '''Evaluate the full decoupled1_fixedp0 model with any parameter not
    in `active` held at zero -- see module note above for why this is
    exactly equivalent to a dedicated reduced-dimension model.'''
    full = dict.fromkeys(_FULL_NAMES, 0.0)
    full.update(zip(active, vals))
    return CaT_GL_gvary_decoupled1_fixedp0(x, *[full[n] for n in _FULL_NAMES])


def lnprior_reduced(theta, active, center_margin=1.0):
    v = dict(zip(active, theta))
    p1 = v['p1']
    if (p1 < CENTER-center_margin) | (p1 > CENTER+center_margin):
        return -np.inf

    has1, has3 = ('dg1' in v), ('dg3' in v)

    if has1:
        dg1, ds1 = v['dg1'], v['ds1']
        if (abs(dg1) > GAMMA_DEV_HARDCAP) | (abs(ds1) > SIGMA_DEV_HARDCAP):
            return -np.inf
        if P2_FIXED * np.exp(ds1) > SIGMA1_ABS_MAX:
            return -np.inf
        if P3_FIXED * np.exp(dg1) > GAMMA1_ABS_MAX:
            return -np.inf
    if has3:
        if abs(v['dg3']) > GAMMA_DEV_HARDCAP:
            return -np.inf

    lnp = dmost_EW._ln_gauss(p1, CENTER, 0.5)
    if has1:
        lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE_1)
        lnp += dmost_EW._ln_gauss(ds1, 0.0, SIGMA_DEV_SCALE_1)
        lnp += _ln_ratio_prior((P3_FIXED*np.exp(dg1)) / (P2_FIXED*np.exp(ds1)))
    if has3:
        lnp += dmost_EW._ln_gauss(v['dg3'], 0.0, GAMMA_DEV_SCALE)
        lnp += _ln_ratio_prior((P3_FIXED*np.exp(v['dg3'])) / P2_FIXED)
    if not np.isfinite(lnp):
        return -np.inf

    depth_names = [n for n in ('p4', 'p5', 'p6', 'p7', 'p8', 'p9') if n in v]
    if any(v[n] < 0 for n in depth_names):
        return -np.inf

    EW2 = v['p5'] + v['p8']
    if EW2 <= 0:
        return -np.inf
    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_beta_frac(v['p8'], EW2)

    if has1:
        EW1 = v['p4'] + v['p7']
        if EW1 <= 0:
            return -np.inf
        lnp += dmost_EW._ln_gauss(EW1, FLAT_EW1_OVER_EW2*EW2, EW1_SCATTER)
        lnp += dmost_EW._ln_beta_frac(v['p7'], EW1)
    if has3:
        EW3 = v['p6'] + v['p9']
        if EW3 <= 0:
            return -np.inf
        lnp += dmost_EW._ln_gauss(EW3, FLAT_EW3_OVER_EW2*EW2, EW3_SCATTER)
        lnp += dmost_EW._ln_beta_frac(v['p9'], EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def lnlike_reduced(theta, active, wvl, spec, ivar, mw):
    model = CaT_GL_gvary_reduced(wvl[mw], active, *theta)
    return -0.5*np.sum((spec[mw]-model)**2 * ivar[mw])


def lnprob_reduced(theta, active, wvl, spec, ivar, mw, center_margin=1.0):
    lp = lnprior_reduced(theta, active, center_margin)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_reduced(theta, active, wvl, spec, ivar, mw)


_SEED = {'p1': None, 'dg1': -0.3, 'dg3': 0.0, 'ds1': -0.5,
          'p4': 0.2, 'p5': 0.2, 'p6': 0.2, 'p7': 0.2, 'p8': 0.2, 'p9': 0.2}
_JITTER = {'p1': 0.1, 'dg1': 0.05, 'dg3': 0.05, 'ds1': 0.1,
           'p4': 0.03, 'p5': 0.03, 'p6': 0.03, 'p7': 0.03, 'p8': 0.03, 'p9': 0.03}
_DEPTH_NAMES = {'p4', 'p5', 'p6', 'p7', 'p8', 'p9'}


def _guess_reduced(active):
    return [CENTER if n == 'p1' else _SEED[n] for n in active]


def _bounds_reduced(active, center_margin=1.0):
    ds1_seed = _SEED['ds1'] if 'ds1' in active else 0.0
    lo, hi = {'p1': CENTER-center_margin}, {'p1': CENTER+center_margin}
    if 'ds1' in active:
        lo['ds1'], hi['ds1'] = -SIGMA_DEV_HARDCAP, SIGMA_DEV_HARDCAP
    if 'dg1' in active:
        dg1_floor = np.log(RATIO_MIN * P2_FIXED * np.exp(ds1_seed) / P3_FIXED)
        lo['dg1'], hi['dg1'] = max(-GAMMA_DEV_HARDCAP, dg1_floor), GAMMA_DEV_HARDCAP
    if 'dg3' in active:
        dg3_floor = np.log(RATIO_MIN * P2_FIXED / P3_FIXED)
        lo['dg3'], hi['dg3'] = max(-GAMMA_DEV_HARDCAP, dg3_floor), GAMMA_DEV_HARDCAP
    for n in _DEPTH_NAMES:
        if n in active:
            lo[n], hi[n] = 0, 5
    return [lo[n] for n in active], [hi[n] for n in active]


def clip_seed_reduced(p, active, margin=0.02, depth_floor=0.05):
    v = dict(zip(active, np.array(p, dtype=float)))
    v['p1'] = np.clip(v['p1'], CENTER-0.98, CENTER+0.98)
    if 'ds1' in v:
        v['ds1'] = np.clip(v['ds1'], -SIGMA_DEV_HARDCAP+margin, np.log(SIGMA1_ABS_MAX/P2_FIXED)-margin)
    if 'dg1' in v:
        ds1_val = v['ds1'] if 'ds1' in v else 0.0
        dg1_floor = np.log(RATIO_MIN * P2_FIXED * np.exp(ds1_val) / P3_FIXED) + margin
        dg1_cap = np.log(GAMMA1_ABS_MAX / P3_FIXED) - margin
        v['dg1'] = np.clip(v['dg1'], dg1_floor, dg1_cap)
    if 'dg3' in v:
        dg3_floor = np.log(RATIO_MIN * P2_FIXED / P3_FIXED) + margin
        v['dg3'] = np.clip(v['dg3'], dg3_floor, None)
    for n in _DEPTH_NAMES:
        if n in v:
            v[n] = max(v[n], depth_floor)
    return np.array([v[n] for n in active])


def curve_fit_seed_reduced(wvl, spec, ivar, mw, active, center_margin=1.0):
    seed = _guess_reduced(active)
    errors = 1./np.sqrt(ivar[mw])
    bounds = _bounds_reduced(active, center_margin)
    try:
        p, pcov = curve_fit(lambda x, *vals: CaT_GL_gvary_reduced(x, active, *vals),
                             wvl[mw], spec[mw], sigma=errors, p0=seed, bounds=bounds, maxfev=20000)
        return np.array(p)
    except Exception:
        return np.array(seed)


def run_emcee_reduced(wvl, spec, ivar, mw, missing1, missing3, p2_fixed, p3_fixed, center_margin=1.0, max_n=3000):
    global P2_FIXED, P3_FIXED
    P2_FIXED, P3_FIXED = p2_fixed, p3_fixed
    active = _active_param_names(missing1, missing3)

    guess = clip_seed_reduced(curve_fit_seed_reduced(wvl, spec, ivar, mw, active, center_margin), active)
    ndim = len(active)
    nwalkers = 32
    jitter = np.array([_JITTER[n] for n in active])
    depth_idx = [i for i, n in enumerate(active) if n in _DEPTH_NAMES]
    p0 = guess + jitter*np.random.randn(nwalkers, ndim)
    p0[:, depth_idx] = np.abs(p0[:, depth_idx])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_reduced, args=(active, wvl, spec, ivar, mw, center_margin))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0, max_n)
    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    idx = {n: i for i, n in enumerate(active)}
    EW2 = chain[:, idx['p5']] + chain[:, idx['p8']]
    EW1 = (chain[:, idx['p4']] + chain[:, idx['p7']] if 'p4' in idx
           else FLAT_EW1_OVER_EW2*EW2 + np.random.normal(0.0, EW1_SCATTER, size=EW2.shape))
    EW3 = (chain[:, idx['p6']] + chain[:, idx['p9']] if 'p6' in idx
           else FLAT_EW3_OVER_EW2*EW2 + np.random.normal(0.0, EW3_SCATTER, size=EW2.shape))

    theta_med = np.median(chain, axis=0)
    fit  = CaT_GL_gvary_reduced(wvl, active, *theta_med)
    chi2 = dmost_EW.calc_chi2_ew(wvl, spec, ivar, mw, fit)

    return {
        'theta_med': theta_med, 'active': active, 'p2_fixed': p2_fixed, 'p3_fixed': p3_fixed,
        'cat': pct_err(EW1+EW2+EW3),
        'ew1': pct_err(EW1), 'ew2': pct_err(EW2), 'ew3': pct_err(EW3),
        'facc': facc, 'convg': convg, 'burnin': burnin,
        'fit': fit, 'chi2': chi2,
    }


def fit_decoupled_stage_missing(wvl, spec, ivar, mw1, mw23, missing1, missing3, center_margin=1.0):
    '''Same contract/return-dict shape as fit_decoupled_stage, for slits
    with no usable data in the EW1 and/or EW3 window (see
    check_missing_lines). The missing line's own parameters are dropped
    from the sampled vector entirely (a genuinely reduced-dimension
    fit, not just a wide prior on an unconstrained parameter) -- and its
    EW/error are substituted analytically from the flat EW-ratio prior
    conditioned on the fitted EW2 posterior (see module docstring above).
    Returns the same theta13 layout as fit_decoupled_stage; the missing
    line's own depth params are back-filled with a nominal 50/50
    gauss/lorentz split of its substituted EW purely so the curve still
    reconstructs sensibly for QA overlays -- the reported EW/err never
    come from that split, only from the substitution itself.'''
    mw = mw1 | mw23
    p_anchor = fit_anchor_stage1(wvl, spec, ivar, mw23, center_margin)
    p2f, p3f = p_anchor[2], p_anchor[3]

    result = run_emcee_reduced(wvl, spec, ivar, mw, missing1, missing3, p2f, p3f, center_margin)
    v = dict(zip(result['active'], result['theta_med']))
    p1 = v['p1']
    dg1, ds1 = v.get('dg1', 0.0), v.get('ds1', 0.0)
    dg3 = v.get('dg3', 0.0)
    p5, p8 = v['p5'], v['p8']
    p4, p7 = (v['p4'], v['p7']) if 'p4' in v else (result['ew1'][0]/2., result['ew1'][0]/2.)
    p6, p9 = (v['p6'], v['p9']) if 'p6' in v else (result['ew3'][0]/2., result['ew3'][0]/2.)

    theta13 = np.array([1.0,p1,p2f,p3f,dg1,dg3,p4,p5,p6,p7,p8,p9,ds1])
    fit = CaT_GL_gvary_decoupled1(wvl, 1.0,p1,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9)
    result['theta13'] = theta13
    result['fit'] = fit
    return result
