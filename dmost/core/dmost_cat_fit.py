import numpy as np

from dmost import dmost_EW, dmost_coadd_emcee
from dmost.core import dmost_cat_model as g

########################################################################
# ADAPTIVE STAGE A/B WRAPPER AROUND THE GL-gvary MODEL (dmost_cat_model.py).
#
# STAGE 0/A: THE DECOUPLED-WIDTH GL-gvary MODEL (g.fit_decoupled_stage),
#   JUST A WIDER HARD BOUND ON THE CENTER FOR STAGE A (SOFT PRIOR
#   UNCHANGED, ONLY THE HARD CUTOFF WIDENS).
# STAGE B: ADDS FREE PER-LINE CENTER OFFSETS (d1, d3) ON TOP OF THE
#   EXISTING PER-LINE GAMMA DEVIATIONS (dg1, dg3) -- 14 PARAMS TOTAL,
#   SHARED-WIDTH MODEL (NOT DECOUPLED).
#
# THIS IS THE dmoster PRODUCTION VERSION OF THE
# gl_gvary_adaptive_test_flatprior.py RESEARCH MODULE (CaT_GL) -- SEE
# THAT FILE'S HISTORY FOR THE FULL DEVELOPMENT/VALIDATION NARRATIVE,
# INCLUDING THE LOW-CaT TRIGGER'S DESIGN VALIDATION (135-slit forced-
# Stage-B labeled test set: 77/135 (57%) trigger, 96% recall of slits
# that show real improvement at 69% precision, vs. 41% precision for
# blanket escalation).
########################################################################

CENTER = g.CENTER
R1, R3 = g.R1, g.R3
GAMMA_DEV_SCALE, GAMMA_DEV_HARDCAP = g.GAMMA_DEV_SCALE, g.GAMMA_DEV_HARDCAP
EW1_SCATTER, EW3_SCATTER = g.EW1_SCATTER, g.EW3_SCATTER

R23_TRIGGER_A = 2.2
R23_TRIGGER_B = 3.0
CENTER_MARGIN_A = 3.5
D1_MARGIN_B = 1.0
D3_MARGIN_B = 5.0
MAX_N = 3000

########################################################################
# LOW-CaT TRIGGER: catches slits with a confidently-near-zero total CaT
# EW that the r23/anchor-stuck triggers above miss entirely (r23 is
# undefined/uninformative when the lines are this weak). Physical
# motivation: CaT EW vs M_V should follow the well-behaved Navabi+26
# relation, which predicts EW>~1 A for essentially all RGB stars.
# `p0_offset` (fitted continuum-level nuisance param) was also tested as
# a candidate veto for the bad-continuum failure mode and found to be
# uncorrelated with outcome (Spearman r=0.004) -- NOT used. A simple
# chi2_0 ceiling is the useful veto instead (slits with chi2_0>5 reliably
# fail to improve under Stage B).
CAT_LOWCAT_TRIGGER  = 1.0    # A, physical floor
CAT_ERR_LOWCAT_MAX  = 0.15   # A, "confidently low" -- small reported error
MISSED_SIG_TRIGGER  = 3.0    # sigma, 3-pixel-smoothed Stage-0 residual
CHI2_0_SANITY_MAX   = 5.0    # veto -- Stage 0 already catastrophically bad


def _window_missed_sig(nwave, nspec, nivar, model, wlo, whi):
    '''Deepest 3-pixel-smoothed (data-below-model) residual, in sigma,
    within one CaT window -- the Stage-0-visible symptom of a real line
    the fixed-center model isn't reaching.'''
    m = (nwave > wlo) & (nwave < whi) & (nivar > 0)
    if np.sum(m) < 3:
        return 0.0
    resid_sig = (nspec[m] - model[m]) * np.sqrt(nivar[m])
    smoothed = np.convolve(resid_sig, np.ones(3) / 3., mode='valid') if len(resid_sig) >= 3 else resid_sig
    return float(-np.min(smoothed))


def max_missed_sig(nwave, nspec, nivar, model):
    windows = [(8484., 8513.), (8522., 8562.), (8642., 8682.)]
    return max(_window_missed_sig(nwave, nspec, nivar, model, wlo, whi) for wlo, whi in windows)


########################################################################
def lnprior_cat_GL_gvary_stageA(theta):
    p0, p1, p2, p3, dg1, dg3, p4, p5, p6, p7, p8, p9 = theta

    if p0 <= 0:
        return -np.inf
    if (p1 < CENTER-CENTER_MARGIN_A) | (p1 > CENTER+CENTER_MARGIN_A):
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

    EW1, EW2, EW3 = p4+p7, p5+p8, p6+p9
    if (EW1 <= 0) | (EW2 <= 0) | (EW3 <= 0):
        return -np.inf

    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_gauss(EW1, g.FLAT_EW1_OVER_EW2*EW2, EW1_SCATTER)
    lnp += dmost_EW._ln_gauss(EW3, g.FLAT_EW3_OVER_EW2*EW2, EW3_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p7, EW1)
    lnp += dmost_EW._ln_beta_frac(p8, EW2)
    lnp += dmost_EW._ln_beta_frac(p9, EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def lnprob_stageA(theta, wvl, spec, ivar, mw):
    lp = lnprior_cat_GL_gvary_stageA(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + g.lnlike_cat_GL_gvary(theta, wvl, spec, ivar, mw)


########################################################################
def CaT_GL_gvary_freecenters(x, p0, p1, p2, p3, dg1, dg3, p4, p5, p6, p7, p8, p9, d1, d3):
    c1, c2, c3 = p1*R1 + d1, p1, p1*R3 + d3

    norm  = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p4*norm*np.exp(-0.5*((x-c1)/p2)**2) + \
            p5*norm*np.exp(-0.5*((x-c2)/p2)**2) + \
            p6*norm*np.exp(-0.5*((x-c3)/p2)**2)

    gamma1, gamma3 = p3*np.exp(dg1), p3*np.exp(dg3)
    norm2_1, norm2_2, norm2_3 = gamma1/(2.*np.pi), p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p7*norm2_1/((x-c1)**2 + (gamma1/2.)**2)) + \
              (p8*norm2_2/((x-c2)**2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-c3)**2 + (gamma3/2.)**2))

    return p0 * (1. - gauss - lorentz)


def lnprior_cat_GL_gvary_stageB(theta):
    p0, p1, p2, p3, dg1, dg3, p4, p5, p6, p7, p8, p9, d1, d3 = theta

    if p0 <= 0:
        return -np.inf
    if (p1 < CENTER-CENTER_MARGIN_A) | (p1 > CENTER+CENTER_MARGIN_A):
        return -np.inf
    if (d1 < -D1_MARGIN_B) | (d1 > D1_MARGIN_B):
        return -np.inf
    if (d3 < -D3_MARGIN_B) | (d3 > D3_MARGIN_B):
        return -np.inf
    if (p2 < 0.4) | (p2 > 2.5):
        return -np.inf
    if (p3 < 0.4) | (p3 > 2.5):
        return -np.inf
    if (abs(dg1) > GAMMA_DEV_HARDCAP) | (abs(dg3) > GAMMA_DEV_HARDCAP):
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p0, 1.0, 0.1)
    lnp += dmost_EW._ln_gauss(p1, CENTER, 0.5)
    lnp += dmost_EW._ln_gauss(d1, 0.0, 0.3)
    lnp += dmost_EW._ln_gauss(d3, 0.0, 0.3)
    lnp += dmost_EW._ln_lognormal(p2, 0.0, 0.5)
    lnp += dmost_EW._ln_lognormal(p3, 0.0, 0.5)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, GAMMA_DEV_SCALE)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, GAMMA_DEV_SCALE)
    if not np.isfinite(lnp):
        return -np.inf

    depths = np.array([p4, p5, p6, p7, p8, p9])
    if np.any(depths < 0):
        return -np.inf

    EW1, EW2, EW3 = p4+p7, p5+p8, p6+p9
    if (EW1 <= 0) | (EW2 <= 0) | (EW3 <= 0):
        return -np.inf

    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_gauss(EW1, g.FLAT_EW1_OVER_EW2*EW2, EW1_SCATTER)
    lnp += dmost_EW._ln_gauss(EW3, g.FLAT_EW3_OVER_EW2*EW2, EW3_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p7, EW1)
    lnp += dmost_EW._ln_beta_frac(p8, EW2)
    lnp += dmost_EW._ln_beta_frac(p9, EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def lnlike_stageB(theta, wvl, spec, ivar, mw):
    model = CaT_GL_gvary_freecenters(wvl[mw], *theta)
    chi2  = (spec[mw]-model)**2 * ivar[mw]
    return -0.5*np.sum(chi2)


def lnprob_stageB(theta, wvl, spec, ivar, mw):
    lp = lnprior_cat_GL_gvary_stageB(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_stageB(theta, wvl, spec, ivar, mw)


########################################################################
JITTER_12 = np.array([0.02, 0.1, 0.05, 0.05, 0.05, 0.05, 0.03,0.03,0.03, 0.03,0.03,0.03])
JITTER_14 = np.concatenate([JITTER_12, [0.15, 0.15]])
POS_IDX_12 = [2, 3, 6, 7, 8, 9, 10, 11]   # WIDTHS + DEPTHS -- dg1/dg3 (4,5) STAY FREE-SIGNED
POS_IDX_14 = POS_IDX_12                    # d1/d3 (12,13) ALSO STAY FREE-SIGNED


def run_fit(lnprob_fn, ndim, seed, wvl, spec, ivar, mw, jitter, pos_idx, nwalkers=32):
    import emcee
    p0 = seed + jitter*np.random.randn(nwalkers, ndim)
    p0[:, pos_idx] = np.abs(p0[:, pos_idx])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_fn, args=(wvl, spec, ivar, mw))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0, MAX_N)
    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    return chain, facc, convg


def pct_err(x):
    p16, p50, p84 = np.percentile(x, [15.8, 50, 84])
    return p50, (p84-p16)/2.


########################################################################
def fit_adaptive_GL_gvary(nwave, nspec, nivar, mw):
    '''
    STAGE 0 -> A -> B.
    Stage 0/A use the decoupled-width fit (EW1 gets its own Gaussian
    width) -- 13-value theta (old 12 + ds1 appended, so theta[:12] still
    matches the old layout). Stage B (rare, free centers) is unchanged --
    still the shared-width model, seeded from thetaA[:12] (drops ds1,
    which Stage B doesn't use).
    RETURNS A DICT: theta, fit (curve), chi2, cat, cat_err, ew (LIST OF 3),
    ew_err (LIST OF 3), stage (0/1/2), facc, convg, escalation_reason.
    '''
    mw1  = (nwave > 8484) & (nwave < 8513)
    mw23 = ((nwave > 8522) & (nwave < 8562)) | ((nwave > 8642) & (nwave < 8682))

    result0 = g.fit_decoupled_stage(nwave, nspec, nivar, mw1, mw23, center_margin=1.0)
    theta0 = result0['theta13']
    ew1_0, ew2_0, ew3_0 = theta0[6]+theta0[9], theta0[7]+theta0[10], theta0[8]+theta0[11]
    chi2_0 = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, result0['fit'])
    result = dict(theta=theta0, fit=result0['fit'], chi2=chi2_0,
                  cat=result0['cat'][0], cat_err=result0['cat'][1], stage=0,
                  ew=[result0['ew1'][0], result0['ew2'][0], result0['ew3'][0]],
                  ew_err=[result0['ew1'][1], result0['ew2'][1], result0['ew3'][1]],
                  facc=result0['facc'], convg=result0['convg'],
                  escalation_reason=None)

    # anchor pinned at its bound -> Stage 1 was forced into a bad compromise
    # fit (usually a mis-centered line it can't reach with a fixed center),
    # which can also make r23 look falsely low. Escalate anyway in that case.
    anchor_stuck0 = (result0['p2_fixed'] >= g.ANCHOR_MAX-1e-6) or (result0['p3_fixed'] >= g.ANCHOR_MAX-1e-6)

    # low-CaT trigger -- see constants/helpers above. Computed unconditionally
    # (cheap) since r23 is undefined/uninformative exactly when this fires.
    lowcat_trigger0 = ((result0['cat'][0] < CAT_LOWCAT_TRIGGER) and (result0['cat'][1] < CAT_ERR_LOWCAT_MAX)
                        and (chi2_0 < CHI2_0_SANITY_MAX)
                        and (max_missed_sig(nwave, nspec, nivar, result0['fit']) > MISSED_SIG_TRIGGER))

    if (ew3_0 <= 0) or (ew2_0 <= 0):
        if not lowcat_trigger0:
            return result
        r23_std = None
    else:
        r23_std = ew2_0/ew3_0
        if (r23_std <= R23_TRIGGER_A) and not anchor_stuck0 and not lowcat_trigger0:
            return result

    if anchor_stuck0 or lowcat_trigger0:
        # Stage A only widens the center prior -- it adds no new per-line
        # freedom, so if the anchor is already stuck at Stage 0 it reliably
        # saturates the same way at Stage A too. Skip that redundant
        # anchor-refit + full emcee fit and go straight to Stage B, which is
        # the stage that can actually fix this. The low-CaT trigger was
        # validated the same way (forced straight to Stage B).
        theta_for_seedB = theta0
        esc_reason = 'anchor_stuck' if anchor_stuck0 else 'lowcat'
        result['escalation_reason'] = esc_reason
    else:
        # ---- STAGE A (wider line-center bound) ----
        esc_reason = 'r23'
        resultA = g.fit_decoupled_stage(nwave, nspec, nivar, mw1, mw23, center_margin=CENTER_MARGIN_A)
        thetaA = resultA['theta13']
        ew1A, ew2A, ew3A = thetaA[6]+thetaA[9], thetaA[7]+thetaA[10], thetaA[8]+thetaA[11]
        chi2A = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, resultA['fit'])
        result = dict(theta=thetaA, fit=resultA['fit'], chi2=chi2A,
                      cat=resultA['cat'][0], cat_err=resultA['cat'][1], stage=1,
                      ew=[resultA['ew1'][0], resultA['ew2'][0], resultA['ew3'][0]],
                      ew_err=[resultA['ew1'][1], resultA['ew2'][1], resultA['ew3'][1]],
                      facc=resultA['facc'], convg=resultA['convg'],
                      escalation_reason=esc_reason)

        anchor_stuckA = (resultA['p2_fixed'] >= g.ANCHOR_MAX-1e-6) or (resultA['p3_fixed'] >= g.ANCHOR_MAX-1e-6)
        r23A = ew2A/ew3A if ew3A > 0 else np.nan
        if not np.isfinite(r23A):
            return result
        if (r23A <= R23_TRIGGER_B) and not anchor_stuckA:
            return result
        theta_for_seedB = thetaA

    # ---- STAGE B (shared-width model, free centers) ----
    # theta_for_seedB's p2/p3 (positions 2,3) can be up to g.ANCHOR_MAX (4.0,
    # the decoupled fit's bound), but Stage B's own prior still caps p2/p3
    # at 2.5 -- clip the seed so it starts inside Stage B's own valid
    # range, or its walkers start invalid and never recover (facc=0.0).
    seedB = np.concatenate([theta_for_seedB[:12], [0.0, 0.0]])
    seedB[2] = np.clip(seedB[2], 0.42, 2.48)
    seedB[3] = np.clip(seedB[3], 0.42, 2.48)
    chainB, faccB, convgB = run_fit(lnprob_stageB, 14, seedB, nwave, nspec, nivar, mw,
                                     JITTER_14, POS_IDX_14)
    thetaB = np.median(chainB, axis=0)
    fitB  = CaT_GL_gvary_freecenters(nwave, *thetaB)
    chi2B = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, fitB)
    catB, catB_err = pct_err(chainB[:,6]+chainB[:,9] + chainB[:,7]+chainB[:,10] + chainB[:,8]+chainB[:,11])
    ew1_Bm, ew1_Be = pct_err(chainB[:,6]+chainB[:,9])
    ew2_Bm, ew2_Be = pct_err(chainB[:,7]+chainB[:,10])
    ew3_Bm, ew3_Be = pct_err(chainB[:,8]+chainB[:,11])
    result = dict(theta=thetaB, fit=fitB, chi2=chi2B, cat=catB, cat_err=catB_err, stage=2,
                  ew=[ew1_Bm, ew2_Bm, ew3_Bm], ew_err=[ew1_Be, ew2_Be, ew3_Be],
                  facc=faccB, convg=convgB, escalation_reason=esc_reason)
    return result
