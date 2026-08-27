import numpy as np
import emcee

from dmost import dmost_EW, dmost_coadd_emcee
from dmost.core import dmost_cat_model as g
from dmost.core import dmost_chi2_criteria as dcc

########################################################################
# ADAPTIVE STAGE 1/STAGE 2 WRAPPER AROUND THE GL-gvary MODEL
# (dmost_cat_model.py).
#
# STAGE 1: THE SOFT-ANCHOR DECOUPLED-WIDTH GL-gvary MODEL
#   (g.fit_decoupled_stage) -- SEE THAT FILE'S MODULE DOCSTRING. RUNS ON
#   EVERY SLIT.
# STAGE 2: A SINGLE COMBINED MODEL THAT KEEPS STAGE 1'S EW1 WIDTH
#   DECOUPLING (ds1/dg1) *AND* ADDS FREE PER-LINE CENTER OFFSETS (d1, d3)
#   -- CaT_GL_gvary_decoupled_freecenters BELOW. ONLY RUN WHEN TRIGGERED.
#
# ADOPTED 2026-08-27, REPLACING THE ORIGINAL STAGE 0 -> A -> B THREE-RUNG
# LADDER (STAGE A -- JUST A WIDER CENTER BOUND, NO NEW FREEDOM -- IS
# GONE; STAGE B's SHARED-WIDTH MODEL IS REPLACED BY THE COMBINED MODEL
# BELOW, WHICH KEEPS EW1 DECOUPLING). STAGE B's OWN FREE-CENTER MACHINERY
# (2026-08-10) PREDATED EW1 WIDTH DECOUPLING (2026-08-24) BY ~2 WEEKS AND
# HAD NEVER BEEN REVISITED TO CARRY IT THROUGH -- COMBINING THE TWO WAS
# THE FIRST DIRECT TEST OF THAT, NOT A RE-LITIGATION OF A VALIDATED
# CHOICE. VALIDATED AT SCALE (144, 755, 1000+, AND 3 FULL MASKS) WITH NO
# SIGNIFICANT CHI2 REGRESSIONS ON THE SAME-CONTINUUM-DRAW COMPARISONS.
#
# `cat_adapt_stage` STILL STORES 0 (STAGE 1, NEVER ESCALATED) OR 2
# (STAGE 2, TRIGGERED) -- KEPT AS-IS RATHER THAN RENUMBERED TO 1/2, SO NO
# DOWNSTREAM CODE THAT FILTERS ON THIS COLUMN NEEDS TO CHANGE.
#
# TRIGGER: THREE CRITERIA, OR'd (SEE research_log_2026-08-26.html FOR THE
# FULL DEVELOPMENT/VALIDATION NARRATIVE, INCLUDING WHY A BARE CaT
# THRESHOLD ISN'T ENOUGH ON ITS OWN):
#   1. max_missed_sig(Stage 1 fit) > A S/N-NORMALIZED ENVELOPE (FLOOR +
#      b*SN^2 FORM, dcc.curve_form, FIT TO THE 90TH PERCENTILE OF THE
#      KNOWN-NORMAL, NON-ESCALATING POPULATION ONLY -- FITTING ON A MIXED
#      POPULATION INFLATES THE ENVELOPE AND KILLS RECALL).
#   2. r23 (EW2/EW3) > R23_TRIGGER_A -- KEPT AS ITS OWN CRITERION SINCE
#      max_missed_sig ALONE TOPS OUT AT ~60-66% RECALL FOR r23-DRIVEN
#      CASES (r23 IMBALANCE IS OFTEN A SUBTLER DEPTH/WIDTH
#      REDISTRIBUTION, NOT A SHARP LOCALIZED MISSED-RESIDUAL SIGNATURE).
#   3. THE LOW-CaT GATE BELOW -- CRITERIA 1+2 ALONE ONLY REACH 54-63%
#      RECALL AGAINST THE ORIGINAL 137-SLIT LOW-CaT-LABELED TEST SET
#      (VS. THE ORIGINAL DESIGN'S VALIDATED 96%). A BARE cat<X CUTOFF WAS
#      ALSO TESTED (AT THE ORIGINAL 2026-08-23 DESIGN TIME): 41%
#      PRECISION BLANKET-ESCALATING VS. 69% PRECISION/96% RECALL GATED --
#      max_missed_sig IS WHAT DISCRIMINATES "INTRINSICALLY WEAK-LINED,
#      FIT IS FINE" FROM "LOOKS WEAK BECAUSE A REAL FEATURE IS MISSED".
########################################################################

R23_TRIGGER_A = 2.2
MAX_N = 3000

# S/N-normalized max_missed_sig envelope (see module docstring above and
# dmost_chi2_criteria.curve_form) -- floor + b*SN^2, 90th percentile fit
# to the known-normal (non-escalating) population only.
MSIG_ENVELOPE_FLOOR = 3.636065188694636
MSIG_ENVELOPE_B     = 0.0008829425082795159

# low-CaT gate (criterion 3 above). Physical motivation for the CaT
# floor: CaT EW vs M_V should follow the well-behaved Navabi+26 relation,
# which predicts EW>~1 A for essentially all RGB stars. msig is the more
# effective lever than cat itself (widening cat past 1.3 plateaus, per
# the 2026-08-27 threshold sweep) -- cat/cat_err/chi2_1 mainly act as a
# sanity gate so this doesn't fire on a merely-noisy-but-fine Stage 1 fit.
CAT_LOWCAT_TRIGGER    = 1.3    # A
CAT_ERR_LOWCAT_MAX    = 0.15   # A, "confidently low" -- small reported error
CHI2_1_SANITY_MAX     = 5.0    # veto -- Stage 1 already catastrophically bad
MISSED_SIG_LOWCAT_MIN = 2.0    # sigma, 3-pixel-smoothed Stage 1 residual

# Stage 2's own internal seeding/priors -- see CaT_GL_gvary_decoupled_
# freecenters / lnprior_stage2 below for how each is used.
CENTER_MARGIN_STAGE2 = 3.5
D1_MARGIN_STAGE2 = 1.0
D3_MARGIN_STAGE2 = 5.0
P2_MAX_STAGE2 = g.ANCHOR_MAX   # 2026-08-27: widened from the old Stage B's
                                 # own 2.5 to match the decoupled anchor's
                                 # 4.0 -- 7/11 chi2 regressions in the first
                                 # combined-model pass had p2 or p3 pinned
                                 # right at 2.5 (all S/N>100), the same
                                 # too-tight-bound pattern already seen
                                 # forcing genuinely broad-lined stars into
                                 # a compromise fit.
DG3_SCALE_STAGE2 = 0.3
DG3_HARDCAP_STAGE2 = 2.0        # 2026-08-27: EW3's own Lorentzian-width
                                 # deviation gets a looser prior/cap than
                                 # Stage 1's default (g.GAMMA_DEV_SCALE=
                                 # 0.05, g.GAMMA_DEV_HARDCAP=1.0), STAGE 2
                                 # ONLY. Fixed 3 flagged slits where Stage
                                 # 2 correctly triggered but still couldn't
                                 # reach a visibly deep, narrow EW3 core --
                                 # the tight default (tuned for the
                                 # well-behaved bulk Stage 1 population)
                                 # was preventing gamma3 from shrinking
                                 # enough. No regression on a known-good
                                 # Stage 2 slit. A ratio3 floor (matching
                                 # Stage 1's own lnprior_decoupled1_
                                 # fixedp0, which applies _ln_ratio_prior
                                 # to both ratio1 and ratio3) was also
                                 # tried here and reverted: the only valid
                                 # counterexample (not junk/missing data)
                                 # showed it reintroduces the missing-
                                 # wings/too-narrow-core problem this dg3
                                 # loosening fixed -- looks like the known
                                 # Gauss+Lorentz-breaks-down-at-high-
                                 # optical-depth issue (status.md), which
                                 # a ratio floor can't distinguish from a
                                 # genuinely degenerate narrow fit.


def msig_envelope(sn):
    return dcc.curve_form(sn, MSIG_ENVELOPE_FLOOR, MSIG_ENVELOPE_B)


def _window_missed_sig(nwave, nspec, nivar, model, wlo, whi):
    '''Deepest 3-pixel-smoothed (data-below-model) residual, in sigma,
    within one CaT window -- the Stage-1-visible symptom of a real line
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
# STAGE 2 MODEL: KEEPS STAGE 1'S DECOUPLED EW1 WIDTH (ds1/dg1, SAME HARD
# CAPS/RATIO FLOOR AS dmost_cat_model.py) AND ADDS FREE PER-LINE CENTER
# OFFSETS d1 (EW1) / d3 (EW3) RELATIVE TO THE SHARED ANCHOR CENTER p1.
# p0 FIXED AT 1.0 (SAME RATIONALE AS dmost_cat_model.py's fit_decoupled_
# stage -- TRUST THE UPSTREAM CONTINUUM NORMALIZATION COMPLETELY).
########################################################################

def CaT_GL_gvary_decoupled_freecenters(x, p1, p2, p3, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9, d1, d3):
    c1, c2, c3 = p1*g.R1 + d1, p1, p1*g.R3 + d3
    sigma1 = p2 * np.exp(ds1)
    norm1 = 1./(np.sqrt(2*np.pi) * sigma1)
    norm  = 1./(np.sqrt(2*np.pi) * p2)
    gauss = p4*norm1*np.exp(-0.5*((x-c1)/sigma1)**2) + \
            p5*norm *np.exp(-0.5*((x-c2)/p2   )**2) + \
            p6*norm *np.exp(-0.5*((x-c3)/p2   )**2)

    gamma1, gamma3 = p3*np.exp(dg1), p3*np.exp(dg3)
    norm2_1, norm2_2, norm2_3 = gamma1/(2.*np.pi), p3/(2.*np.pi), gamma3/(2.*np.pi)
    lorentz = (p7*norm2_1/((x-c1)**2 + (gamma1/2.)**2)) + \
              (p8*norm2_2/((x-c2)**2 + (p3    /2.)**2)) + \
              (p9*norm2_3/((x-c3)**2 + (gamma3/2.)**2))
    return 1.0 * (1. - gauss - lorentz)


def lnprior_stage2(theta):
    p1, p2, p3, dg1, dg3, ds1, p4, p5, p6, p7, p8, p9, d1, d3 = theta

    if (p1 < g.CENTER-CENTER_MARGIN_STAGE2) | (p1 > g.CENTER+CENTER_MARGIN_STAGE2):
        return -np.inf
    if (d1 < -D1_MARGIN_STAGE2) | (d1 > D1_MARGIN_STAGE2) | (d3 < -D3_MARGIN_STAGE2) | (d3 > D3_MARGIN_STAGE2):
        return -np.inf
    if (p2 < 0.4) | (p2 > P2_MAX_STAGE2) | (p3 < 0.4) | (p3 > P2_MAX_STAGE2):
        return -np.inf
    if (abs(dg1) > g.GAMMA_DEV_HARDCAP) | (abs(dg3) > DG3_HARDCAP_STAGE2):
        return -np.inf
    if abs(ds1) > g.SIGMA_DEV_HARDCAP:
        return -np.inf
    if p2 * np.exp(ds1) > g.SIGMA1_ABS_MAX:
        return -np.inf
    if p3 * np.exp(dg1) > g.GAMMA1_ABS_MAX:
        return -np.inf

    lnp  = dmost_EW._ln_gauss(p1, g.CENTER, 0.5)
    lnp += dmost_EW._ln_gauss(d1, 0.0, 0.3)
    lnp += dmost_EW._ln_gauss(d3, 0.0, 0.3)
    lnp += dmost_EW._ln_lognormal(p2, 0.0, 0.5)
    lnp += dmost_EW._ln_lognormal(p3, 0.0, 0.5)
    lnp += dmost_EW._ln_gauss(dg1, 0.0, g.GAMMA_DEV_SCALE_1)
    lnp += dmost_EW._ln_gauss(dg3, 0.0, DG3_SCALE_STAGE2)
    lnp += dmost_EW._ln_gauss(ds1, 0.0, g.SIGMA_DEV_SCALE_1)
    ratio1 = (p3*np.exp(dg1)) / (p2*np.exp(ds1))
    lnp += g._ln_ratio_prior(ratio1)
    if not np.isfinite(lnp):
        return -np.inf

    depths = np.array([p4, p5, p6, p7, p8, p9])
    if np.any(depths < 0):
        return -np.inf
    EW1, EW2, EW3 = p4+p7, p5+p8, p6+p9
    if (EW1 <= 0) | (EW2 <= 0) | (EW3 <= 0):
        return -np.inf

    lnp += -0.5*(EW2/3.0)**2
    lnp += dmost_EW._ln_gauss(EW1, g.FLAT_EW1_OVER_EW2*EW2, g.EW1_SCATTER)
    lnp += dmost_EW._ln_gauss(EW3, g.FLAT_EW3_OVER_EW2*EW2, g.EW3_SCATTER)
    lnp += dmost_EW._ln_beta_frac(p7, EW1)
    lnp += dmost_EW._ln_beta_frac(p8, EW2)
    lnp += dmost_EW._ln_beta_frac(p9, EW3)

    if not np.isfinite(lnp):
        return -np.inf
    return lnp


def lnlike_stage2(theta, wvl, spec, ivar, mw):
    model = CaT_GL_gvary_decoupled_freecenters(wvl[mw], *theta)
    return -0.5*np.sum((spec[mw]-model)**2 * ivar[mw])


def lnprob_stage2(theta, wvl, spec, ivar, mw):
    lp = lnprior_stage2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_stage2(theta, wvl, spec, ivar, mw)


def fit_stage2(nwave, nspec, nivar, mw1, mw23, max_n=3000):
    '''Seeded exactly the way Stage 2 has always been seeded: straight
    from a fresh Stage 1 fit at Stage 2's own (wider) center margin,
    d1=d3=0 -- no curve_fit, no separate MAP call. Reported theta/curve/
    chi2 come from the single highest-log-probability chain sample, same
    reasoning as Stage 1 (dmost_cat_model.py module docstring).'''
    mw = mw1 | mw23
    result1w = g.fit_decoupled_stage(nwave, nspec, nivar, mw1, mw23, center_margin=CENTER_MARGIN_STAGE2)
    theta1w = result1w['theta13']   # p0,p1,p2,p3,dg1,dg3,p4..p9,ds1
    seed = np.array([theta1w[1], theta1w[2], theta1w[3], theta1w[4], theta1w[5], theta1w[12],
                      theta1w[6], theta1w[7], theta1w[8], theta1w[9], theta1w[10], theta1w[11], 0.0, 0.0])
    seed[1] = np.clip(seed[1], 0.42, P2_MAX_STAGE2-0.02)
    seed[2] = np.clip(seed[2], 0.42, P2_MAX_STAGE2-0.02)

    ndim, nwalkers = 14, 32
    jitter = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.03,0.03,0.03, 0.03,0.03,0.03, 0.15, 0.15])
    p0w = seed + jitter*np.random.randn(nwalkers, ndim)
    p0w[:, [1,2,6,7,8,9,10,11]] = np.abs(p0w[:, [1,2,6,7,8,9,10,11]])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_stage2, args=(nwave, nspec, nivar, mw))
    sampler, convg, burnin = dmost_coadd_emcee.run_sampler(sampler, p0w, max_n)
    facc  = np.mean(sampler.acceptance_fraction)
    chain = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    lnprob_chain = sampler.lnprobability[:, burnin:].reshape(-1)

    theta_best = chain[np.argmax(lnprob_chain)]
    EW1 = chain[:,6] + chain[:,9]
    EW2 = chain[:,7] + chain[:,10]
    EW3 = chain[:,8] + chain[:,11]

    fit  = CaT_GL_gvary_decoupled_freecenters(nwave, *theta_best)
    chi2 = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, fit)

    return {
        'theta_best': theta_best, 'p2': theta_best[1], 'p3': theta_best[2],
        'cat': g.pct_err(EW1+EW2+EW3),
        'ew1': g.pct_err(EW1), 'ew2': g.pct_err(EW2), 'ew3': g.pct_err(EW3),
        'facc': facc, 'convg': convg, 'fit': fit, 'chi2': chi2,
    }


########################################################################
def fit_adaptive_GL_gvary(nwave, nspec, nivar, mw, SN):
    '''
    STAGE 1 -> STAGE 2. See module docstring above for the full design.
    RETURNS A DICT: theta, fit (curve), chi2, cat, cat_err, ew (LIST OF 3),
    ew_err (LIST OF 3), stage (0=Stage 1/never escalated, 2=Stage 2
    triggered), facc, convg, escalation_reason, ew_flag (0=none, 1=EW1
    analytically substituted [no data coverage], 2=EW3 substituted,
    3=both -- see dmost_cat_model.py's missing-line-coverage section).
    '''
    mw1  = (nwave > 8484) & (nwave < 8513)
    mw23 = ((nwave > 8522) & (nwave < 8562)) | ((nwave > 8642) & (nwave < 8682))

    # MISSING-LINE-COVERAGE HANDLING (dmost_cat_model.py, ADOPTED 2026-08-24):
    # NO USABLE DATA AT ALL IN THE EW1 AND/OR EW3 WINDOW -- FIT A GENUINELY
    # REDUCED-DIMENSION MODEL (MISSING LINE'S OWN PARAMS DROPPED, NOT JUST
    # WIDE-PRIORED) AND SUBSTITUTE ITS EW ANALYTICALLY FROM THE EW2-
    # CONDITIONED PRIOR. TERMINAL HERE -- NEVER ESCALATES TO STAGE 2 (SEE
    # dmost_cat_model.py DOCSTRING FOR WHY). ew_flag: 0=NONE, 1=EW1
    # SUBSTITUTED, 2=EW3 SUBSTITUTED, 3=BOTH.
    missing1, missing3 = g.check_missing_lines(nwave, nivar)
    if missing1 or missing3:
        ew_flag = (1 if missing1 else 0) + (2 if missing3 else 0)

        # EW2 ITSELF MUST BE DETECTED TO SERVE AS THE SUBSTITUTION ANCHOR
        # (FOUND 2026-08-27, N147_5): WITHOUT THIS CHECK, A SLIT WHOSE EW2
        # WINDOW IS ALSO UNCOVERED (OR ONLY PARTIALLY COVERED) STILL GETS
        # EW1/EW3 "SUBSTITUTED" FROM AN "EW2 POSTERIOR" THAT'S REALLY JUST
        # THE PRIOR WITH NO REAL DATA BEHIND IT -- PRODUCES NONSENSE (E.G.
        # cat=40 A ON A S/N~2.9 SLIT). IF EW2 ISN'T DETECTED, REPORT NO
        # MEASUREMENT AT ALL RATHER THAN A FABRICATED VALUE (SAME -999
        # SENTINEL CONVENTION AS THE "FLAG EW DISASTERS" CHI2 VETO IN
        # dmost_EW.py).
        if not g.check_ew2_detected(nwave, nivar):
            return dict(theta=np.array([]), fit=np.ones_like(nwave), chi2=-999.,
                        cat=-999., cat_err=-999., stage=0,
                        ew=[-999., -999., -999.], ew_err=[-999., -999., -999.],
                        facc=0., convg=0, escalation_reason='no_ew2_detection', ew_flag=ew_flag)

        result0 = g.fit_decoupled_stage_missing(nwave, nspec, nivar, mw1, mw23, missing1, missing3, center_margin=1.0)
        theta0 = result0['theta13']
        chi2_0 = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, result0['fit'])
        esc_reason = 'missing_ew1' if (missing1 and not missing3) else ('missing_ew3' if (missing3 and not missing1) else 'missing_both')
        return dict(theta=theta0, fit=result0['fit'], chi2=chi2_0,
                    cat=result0['cat'][0], cat_err=result0['cat'][1], stage=0,
                    ew=[result0['ew1'][0], result0['ew2'][0], result0['ew3'][0]],
                    ew_err=[result0['ew1'][1], result0['ew2'][1], result0['ew3'][1]],
                    facc=result0['facc'], convg=result0['convg'],
                    escalation_reason=esc_reason, ew_flag=ew_flag)

    # ---- STAGE 1 ----
    result1 = g.fit_decoupled_stage(nwave, nspec, nivar, mw1, mw23, center_margin=1.0)
    theta1 = result1['theta13']
    ew1_1, ew2_1, ew3_1 = theta1[6]+theta1[9], theta1[7]+theta1[10], theta1[8]+theta1[11]
    chi2_1 = dmost_EW.calc_chi2_ew(nwave, nspec, nivar, mw, result1['fit'])
    cat1_val, cat1_err = result1['cat']
    result = dict(theta=theta1, fit=result1['fit'], chi2=chi2_1,
                  cat=cat1_val, cat_err=cat1_err, stage=0,
                  ew=[result1['ew1'][0], result1['ew2'][0], result1['ew3'][0]],
                  ew_err=[result1['ew1'][1], result1['ew2'][1], result1['ew3'][1]],
                  facc=result1['facc'], convg=result1['convg'],
                  escalation_reason=None, ew_flag=0)

    msig1 = max_missed_sig(nwave, nspec, nivar, result1['fit'])
    r23_1 = ew2_1/ew3_1 if ew3_1 > 0 else np.nan

    lowcat = ((cat1_val < CAT_LOWCAT_TRIGGER) and (cat1_err < CAT_ERR_LOWCAT_MAX)
              and (chi2_1 < CHI2_1_SANITY_MAX) and (msig1 > MISSED_SIG_LOWCAT_MIN))
    trig_reason = []
    if msig1 > msig_envelope(SN):
        trig_reason.append('msig')
    if np.isfinite(r23_1) and r23_1 > R23_TRIGGER_A:
        trig_reason.append('r23')
    if lowcat:
        trig_reason.append('lowcat')

    if not trig_reason:
        return result

    # ---- STAGE 2 ----
    result2 = fit_stage2(nwave, nspec, nivar, mw1, mw23)
    cat2_val, cat2_err = result2['cat']
    result = dict(theta=result2['theta_best'], fit=result2['fit'], chi2=result2['chi2'],
                  cat=cat2_val, cat_err=cat2_err, stage=2,
                  ew=[result2['ew1'][0], result2['ew2'][0], result2['ew3'][0]],
                  ew_err=[result2['ew1'][1], result2['ew2'][1], result2['ew3'][1]],
                  facc=result2['facc'], convg=result2['convg'],
                  escalation_reason=','.join(trig_reason), ew_flag=0)
    return result
