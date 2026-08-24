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
# GAUSSIAN WIDTH INSTEAD OF SHARING p2 WITH EW2/EW3. FIT IN TWO STEPS:
#   1. anchor fit: p2/p3 FROM EW2+EW3 ONLY (EW1 EXCLUDED)
#   2. full fit: p2/p3 HELD FIXED, EW1'S OWN WIDTH FREE VIA ds1
# WHY: EW1'S SHARED-WIDTH FIT WAS TOO BROAD (TUNED BY EW2/EW3, WHICH ARE
# WIDER LINES); A FULLY FREE ds1 WITHOUT A FIXED ANCHOR LET p2 DRIFT AND
# HURT EW2/EW3 TOO. FIXING p2/p3 FIRST AVOIDS THAT.
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


def CaT_GL_gvary(x, *p):
    # p[0] = CONTINUUM
    # p[1] = LINE POSITION (8542 anchor)
    # p[2] = GAUSSIAN WIDTH -- SHARED ACROSS ALL 3 LINES
    # p[3] = LORENTZIAN WIDTH -- 8542 (ANCHOR) LINE
    # p[4] = dg1 -- LOG-SPACE DEVIATION OF 8498's LORENTZIAN WIDTH FROM THE ANCHOR
    # p[5] = dg3 -- LOG-SPACE DEVIATION OF 8662's LORENTZIAN WIDTH FROM THE ANCHOR
    # p[6],p[7],p[8]   = GAUSSIAN DEPTH FOR 8498/8542/8662
    # p[9],p[10],p[11] = LORENTZIAN DEPTH FOR 8498/8542/8662
    p0, p1, p2, p3, dg1, dg3, p4, p5, p6, p7, p8, p9 = p

    norm  = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p4*norm*np.exp(-0.5*((x-p1*R1)/p2)**2) + \
            p5*norm*np.exp(-0.5*((x-p1)/p2)**2) + \
            p6*norm*np.exp(-0.5*((x-p1*R3)/p2)**2)

    gamma1, gamma3 = p3*np.exp(dg1), p3*np.exp(dg3)
    norm2_1, norm2_2, norm2_3 = gamma1/(2.*np.pi), p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p7*norm2_1/((x-p1*R1)**2 + (gamma1/2.)**2)) + \
              (p8*norm2_2/((x-p1)   **2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-p1*R3)**2 + (gamma3/2.)**2))

    return p0 * (1. - gauss - lorentz)


########################################################################
def lnprior_cat_GL_gvary(theta):

    p0, p1, p2, p3, dg1, dg3, p4, p5, p6, p7, p8, p9 = theta

    if p0 <= 0:
        return -np.inf
    if (p1 < 8541.09) | (p1 > 8543.09):
        return -np.inf
    if (p2 < 0.4) | (p2 > 2.5):
        return -np.inf
    if (p3 < 0.4) | (p3 > 2.5):
        return -np.inf
    if (abs(dg1) > GAMMA_DEV_HARDCAP) | (abs(dg3) > GAMMA_DEV_HARDCAP):
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p0, 1.0, 0.1)
    lnp += dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_lognormal(p2, 0.0, 0.5)
    lnp += dmost_EW._ln_lognormal(p3, 0.0, 0.5)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, GAMMA_DEV_SCALE)
    if not np.isfinite(lnp):
        return -np.inf

    depths = np.array([p4, p5, p6, p7, p8, p9])
    if np.any(depths < 0):
        return -np.inf

    EW_8498 = p4 + p7
    EW_8542 = p5 + p8
    EW_8662 = p6 + p9
    if (EW_8498 <= 0) | (EW_8542 <= 0) | (EW_8662 <= 0):
        return -np.inf

    lnp += -0.5*(EW_8542/3.0)**2
    lnp += dmost_EW._ln_gauss(EW_8498, dmost_EW.EW1_SLOPE*EW_8542 + dmost_EW.EW1_INT, EW1_SCATTER)
    lnp += dmost_EW._ln_gauss(EW_8662, dmost_EW.EW3_SLOPE*EW_8542 + dmost_EW.EW3_INT, EW3_SCATTER)

    lnp += dmost_EW._ln_beta_frac(p7, EW_8498)
    lnp += dmost_EW._ln_beta_frac(p8, EW_8542)
    lnp += dmost_EW._ln_beta_frac(p9, EW_8662)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def lnlike_cat_GL_gvary(theta, wvl, spec, ivar, mw):
    model = CaT_GL_gvary(wvl[mw], *theta)
    chi2  = (spec[mw]-model)**2 * ivar[mw]
    return -0.5*np.sum(chi2)


def lnprob_cat_GL_gvary(theta, wvl, spec, ivar, mw):
    lp = lnprior_cat_GL_gvary(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_cat_GL_gvary(theta, wvl, spec, ivar, mw)


########################################################################
def CaT_GL_gvary_guess():
    Ng, sg = 0.2, 1.5
    return [1., CENTER, sg, 0.8*sg, 0.0, 0.0, Ng, Ng, Ng, Ng, Ng, Ng]


def curve_fit_seed_GL_gvary(wvl, spec, ivar, mw):
    p0 = CaT_GL_gvary_guess()
    errors = 1./np.sqrt(ivar[mw])
    try:
        p, pcov = curve_fit(CaT_GL_gvary, wvl[mw], spec[mw], sigma=errors, p0=p0,
            bounds=((0.9, 8539.5, 0.5, 0.5, -GAMMA_DEV_HARDCAP, -GAMMA_DEV_HARDCAP, 0.1,0.1,0.1, 0,0,0),
                    (1.5, 8544.5, 3.0, 3.0,  GAMMA_DEV_HARDCAP,  GAMMA_DEV_HARDCAP, 4,4,4, 3,3,3)))
        return np.array(p)
    except Exception:
        return np.array(p0)


def clip_seed_to_prior_GL_gvary(p, margin=0.02, depth_floor=0.05):
    p = np.array(p, dtype=float)
    p[1] = np.clip(p[1], 8541.09+margin, 8543.09-margin)
    p[2] = np.clip(p[2], 0.4+margin, 2.5-margin)
    p[3] = np.clip(p[3], 0.4+margin, 2.5-margin)
    p[4] = np.clip(p[4], -GAMMA_DEV_HARDCAP+margin, GAMMA_DEV_HARDCAP-margin)
    p[5] = np.clip(p[5], -GAMMA_DEV_HARDCAP+margin, GAMMA_DEV_HARDCAP-margin)
    p[6:12] = np.clip(p[6:12], depth_floor, None)
    return p


def initialize_walkers_GL_gvary(guess, nwalkers=32):
    ndim = len(guess)
    jitter = np.array([0.02, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03,0.03,0.03, 0.03,0.03,0.03])
    p0 = guess + jitter*np.random.randn(nwalkers, ndim)
    # WIDTHS (2,3) AND DEPTHS (6-11) MUST STAY POSITIVE -- dg1/dg3 (4,5) CAN BE NEGATIVE
    p0[:, [2,3,6,7,8,9,10,11]] = np.abs(p0[:, [2,3,6,7,8,9,10,11]])
    return ndim, nwalkers, p0


def pct_err(x):
    p16, p50, p84 = np.percentile(x, [15.8, 50, 84])
    return p50, (p84-p16)/2.


def run_emcee_cat_GL_gvary(wvl, spec, ivar, mw, max_n=3000):
    guess = clip_seed_to_prior_GL_gvary(curve_fit_seed_GL_gvary(wvl, spec, ivar, mw))
    ndim, nwalkers, p0 = initialize_walkers_GL_gvary(guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_cat_GL_gvary, args=(wvl, spec, ivar, mw))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0, max_n)

    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    EW1 = chain[:,6] + chain[:,9]
    EW2 = chain[:,7] + chain[:,10]
    EW3 = chain[:,8] + chain[:,11]

    theta_med = np.median(chain, axis=0)
    fit  = CaT_GL_gvary(wvl, *theta_med)
    chi2 = dmost_EW.calc_chi2_ew(wvl, spec, ivar, mw, fit)

    return {
        'theta_med': theta_med,
        'cat': pct_err(EW1+EW2+EW3),
        'ew1': pct_err(EW1), 'ew2': pct_err(EW2), 'ew3': pct_err(EW3),
        'facc': facc, 'convg': convg, 'burnin': burnin,
        'fit': fit, 'chi2': chi2,
    }


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
    # MAP fit (prior + likelihood), not a bare curve_fit -- a bare
    # least-squares fit can run away to the bound when the data can't
    # break the width/depth degeneracy (mainly at low S/N).
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
    return np.array(res.x)   # p0,p1,p2,p3,dg3,p5,p6,p8,p9


# p2/p3 for the functions below are set once per slit by run_emcee_decoupled1
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


def lnprior_decoupled1(theta, center_margin=1.0):
    p0, p1, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9 = theta

    if p0 <= 0:
        return -np.inf
    if (p1 < CENTER-center_margin) | (p1 > CENTER+center_margin):
        return -np.inf
    if (abs(dg1) > GAMMA_DEV_HARDCAP) | (abs(dg3) > GAMMA_DEV_HARDCAP):
        return -np.inf
    if abs(ds1) > SIGMA_DEV_HARDCAP:
        return -np.inf
    if P2_FIXED * np.exp(ds1) > SIGMA1_ABS_MAX:
        return -np.inf
    if P3_FIXED * np.exp(dg1) > GAMMA1_ABS_MAX:
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p0, 1.0, 0.1)
    lnp += dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE_1)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, GAMMA_DEV_SCALE)
    lnp += dmost_EW._ln_gauss(ds1, 0.0, SIGMA_DEV_SCALE_1)

    ratio1 = (P3_FIXED*np.exp(dg1)) / (P2_FIXED*np.exp(ds1))
    ratio3 = (P3_FIXED*np.exp(dg3)) / P2_FIXED
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


def lnlike_decoupled1(theta, wvl, spec, ivar, mw):
    model = CaT_GL_gvary_decoupled1(wvl[mw], *theta)
    return -0.5*np.sum((spec[mw]-model)**2 * ivar[mw])


def lnprob_decoupled1(theta, wvl, spec, ivar, mw, center_margin=1.0):
    lp = lnprior_decoupled1(theta, center_margin)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_decoupled1(theta, wvl, spec, ivar, mw)


def guess_decoupled1():
    Ng = 0.2
    return [1., CENTER, 0.0, 0.0, 0.0, Ng, Ng, Ng, Ng, Ng, Ng]


def curve_fit_seed_decoupled1(wvl, spec, ivar, mw, center_margin=1.0):
    # raw MLE (no priors) to give emcee a good starting point
    p0 = guess_decoupled1()
    p0[2], p0[4] = -0.3, -0.5   # dg1, ds1 -- seed narrower, not at zero
    errors = 1./np.sqrt(ivar[mw])
    dg3_floor = np.log(RATIO_MIN * P2_FIXED / P3_FIXED)
    dg1_floor = np.log(RATIO_MIN * P2_FIXED * np.exp(p0[4]) / P3_FIXED)
    bounds = ([0.8, CENTER-center_margin, max(-GAMMA_DEV_HARDCAP,dg1_floor), max(-GAMMA_DEV_HARDCAP,dg3_floor), -SIGMA_DEV_HARDCAP, 0,0,0, 0,0,0],
              [1.2, CENTER+center_margin,  GAMMA_DEV_HARDCAP, GAMMA_DEV_HARDCAP,  SIGMA_DEV_HARDCAP, 5,5,5, 5,5,5])
    try:
        p, pcov = curve_fit(CaT_GL_gvary_decoupled1, wvl[mw], spec[mw], sigma=errors, p0=p0,
                             bounds=bounds, maxfev=20000)
        return np.array(p)
    except Exception:
        return np.array(p0)


def clip_seed_to_prior_decoupled1(p, margin=0.02, depth_floor=0.05):
    p = np.array(p, dtype=float)
    p[1] = np.clip(p[1], CENTER-0.98, CENTER+0.98)
    p[4] = np.clip(p[4], -SIGMA_DEV_HARDCAP+margin, np.log(SIGMA1_ABS_MAX/P2_FIXED)-margin)
    # ratio floor (needed so the seed itself isn't already invalid)
    dg3_floor = np.log(RATIO_MIN * P2_FIXED / P3_FIXED) + margin
    dg1_floor = np.log(RATIO_MIN * P2_FIXED * np.exp(p[4]) / P3_FIXED) + margin
    dg1_cap = np.log(GAMMA1_ABS_MAX / P3_FIXED) - margin
    p[3] = np.clip(p[3], dg3_floor, None)
    p[2] = np.clip(p[2], dg1_floor, dg1_cap)
    p[5:11] = np.clip(p[5:11], depth_floor, None)
    return p


def initialize_walkers_decoupled1(guess, nwalkers=32):
    ndim = len(guess)
    jitter = np.array([0.02, 0.1, 0.05, 0.05, 0.1, 0.03,0.03,0.03, 0.03,0.03,0.03])
    p0 = guess + jitter*np.random.randn(nwalkers, ndim)
    p0[:, [5,6,7,8,9,10]] = np.abs(p0[:, [5,6,7,8,9,10]])
    return ndim, nwalkers, p0


def run_emcee_decoupled1(wvl, spec, ivar, mw, p2_fixed, p3_fixed, center_margin=1.0, max_n=3000):
    global P2_FIXED, P3_FIXED
    P2_FIXED, P3_FIXED = p2_fixed, p3_fixed

    guess = clip_seed_to_prior_decoupled1(curve_fit_seed_decoupled1(wvl, spec, ivar, mw, center_margin))
    ndim, nwalkers, p0 = initialize_walkers_decoupled1(guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_decoupled1, args=(wvl, spec, ivar, mw, center_margin))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0, max_n)

    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    EW1 = chain[:,5] + chain[:,8]
    EW2 = chain[:,6] + chain[:,9]
    EW3 = chain[:,7] + chain[:,10]

    theta_med = np.median(chain, axis=0)
    fit  = CaT_GL_gvary_decoupled1(wvl, *theta_med)
    chi2 = dmost_EW.calc_chi2_ew(wvl, spec, ivar, mw, fit)

    return {
        'theta_med': theta_med, 'p2_fixed': p2_fixed, 'p3_fixed': p3_fixed,
        'cat': pct_err(EW1+EW2+EW3),
        'ew1': pct_err(EW1), 'ew2': pct_err(EW2), 'ew3': pct_err(EW3),
        'facc': facc, 'convg': convg, 'burnin': burnin,
        'fit': fit, 'chi2': chi2,
    }


def fit_decoupled_stage(wvl, spec, ivar, mw1, mw23, center_margin=1.0):
    '''One-call wrapper: anchor fit (EW2+EW3) then full fit (p2/p3 fixed).
    Returns theta as 13 values: p0,p1,p2,p3,dg1,dg3,p4..p9,ds1 -- same
    order as the old 12-param model with ds1 appended, so old code
    reading theta[:12] still works.'''
    mw = mw1 | mw23
    p_anchor = fit_anchor_stage1(wvl, spec, ivar, mw23, center_margin)
    p2f, p3f = p_anchor[2], p_anchor[3]
    result = run_emcee_decoupled1(wvl, spec, ivar, mw, p2f, p3f, center_margin)
    p0,p1,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9 = result['theta_med']
    theta13 = np.array([p0,p1,p2f,p3f,dg1,dg3,p4,p5,p6,p7,p8,p9,ds1])
    fit = CaT_GL_gvary_decoupled1(wvl, p0,p1,dg1,dg3,ds1,p4,p5,p6,p7,p8,p9)
    result['theta13'] = theta13
    result['fit'] = fit
    return result
