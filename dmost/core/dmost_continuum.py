import numpy as np

########################################################################
# CaT CONTINUUM NORMALIZATION -- ivar-WEIGHTED VERSION OF THE ORIGINAL
# dmost_EW.CaII_normalize (WHICH USED AN UNWEIGHTED POLYFIT, LETTING A
# NOISY OR BAD PIXEL IN A CONTINUUM BAND COUNT EQUALLY WITH A CLEAN,
# HIGH-S/N PIXEL). CaII_normalize_weighted_looflag (BELOW) IS THE
# PRODUCTION ENTRY POINT: LEAVE-ONE-OUT, BOOTSTRAP-chi2 BAND-DROPPING ON
# TOP OF THE ivar-WEIGHTED FIT. THE TWO SIMPLER VARIANTS ABOVE IT
# (CaII_normalize_weighted, CaII_normalize_weighted_bandflag) ARE KEPT AS
# THE DEVELOPMENT STEPS THAT MOTIVATE IT, NOT SEPARATELY WIRED IN
# ANYWHERE.
########################################################################

CONT1 = [8474.0, 8484.0]
CONT2 = [8563.0, 8577.0]
CONT3 = [8619.0, 8642.0]
CONT4 = [8700.0, 8725.0]
CONT5 = [8776.0, 8792.0]


def CaII_normalize_weighted(wave, spec, ivar):
    '''
    SAME 5 CENARRO (2001) CONTINUUM BANDS AS dmost_EW.CaII_normalize, BUT
    THE LINEAR CONTINUUM FIT IS INVERSE-VARIANCE WEIGHTED (numpy's
    polyfit w= EXPECTS 1/sigma, i.e. sqrt(ivar), NOT ivar ITSELF) INSTEAD
    OF UNWEIGHTED. PIXELS WITH NON-FINITE OR NON-POSITIVE ivar ARE
    DROPPED BEFORE THE FIT -- MATHEMATICALLY REDUNDANT WITH THE WEIGHTING
    FOR ivar=0 (WEIGHT WOULD ALREADY BE ZERO), BUT NEEDED AS A NUMERICAL
    GUARD AGAINST NaN/inf ivar POISONING polyfit.
    '''

    m1 = (wave > CONT1[0]) & (wave < CONT1[1])
    m2 = (wave > CONT2[0]) & (wave < CONT2[1])
    m3 = (wave > CONT3[0]) & (wave < CONT3[1])
    m4 = (wave > CONT4[0]) & (wave < CONT4[1])
    m5 = (wave > CONT5[0]) & (wave < CONT5[1])
    m = m1 | m2 | m3 | m4 | m5

    fwave, fspec, fivar = wave[m], spec[m], ivar[m]

    good = np.isfinite(fivar) & (fivar > 0)
    w = np.sqrt(fivar[good])

    z = np.polyfit(fwave[good], fspec[good], 1, w=w)
    p = np.poly1d(z)
    fit = p(wave)

    nwave = wave
    nspec = spec / fit
    nivar = ivar * fit**2

    return nwave, nspec, nivar


########################################################################
# ADDS A cont4 SUPPRESSION FLAG ON TOP OF CaII_normalize_weighted --
# MOTIVATED BY scl6/1009811 (research_log_2026-08-19.html). NOT A SIMPLE
# ratio4 = cont4/mean(cont1,2,3) CUT -- THAT PRODUCED FALSE POSITIVES ON
# SLITS WITH A NORMAL CONTINUUM SLOPE (A REAL SED SLOPE ALONE MAKES
# cont4 READ DIFFERENT FROM A FLAT AVERAGE OF cont1-3, EVEN WITH NO
# PROBLEM AT ALL). INSTEAD: FIT THE LINEAR CONTINUUM THROUGH THE OTHER
# 4 BANDS (cont1,2,3,5 -- LEAVING cont4 OUT), EXTRAPOLATE THAT LINE TO
# cont4's WAVELENGTHS, AND COMPARE cont4's ACTUAL FLUX TO WHAT THE LINE
# PREDICTS THERE -- I.E. IS cont4 CONSISTENT WITH THE SAME LINEAR TREND
# THE OTHER 4 BANDS FOLLOW, OR IS IT A GENUINE, NON-LINEAR OUTLIER?
# VALIDATED ON 4 REAL SLITS: 3 WITH A NORMAL SLOPE CAME BACK AT
# frac_resid=+-1%, THE ONE GENUINE CASE (LM108/GaiaFill_70) AT -25%.
# THRESHOLD (15%) MATCHES THE HISTORICAL LEAVE-ONE-OUT BAND-EXCLUSION
# CRITERION FROM THE 2026-08-06 SESSION (CaT_EW_pipeline_summary.md),
# SUPERSEDED THEN BY THE VIGNETTING FIX BUT APPARENTLY STILL NEEDED FOR
# A RESIDUAL SET OF SLITS.
#
# S/N FLOOR ADDED AFTER REVIEWING 15 REAL TRIGGERED SLITS: NEITHER
# frac_resid NOR AN ivar- OR EMPIRICAL-SCATTER-BASED z-SCORE CLEANLY
# SEPARATED GENUINE CASES FROM FALSE POSITIVES -- WITH ONLY ~10-20
# PIXELS PER CONTINUUM BAND, THE PER-SLIT FIT IS JUST TOO NOISY BELOW
# S/N~30 TO TRUST REGARDLESS OF HOW THE SIGNIFICANCE IS COMPUTED. THE
# REVIEWED EXAMPLES SPLIT CLEANLY BY S/N ALONE: EVERY FALSE POSITIVE WAS
# S/N<25, EVERY CONFIRMED REAL CASE WAS S/N>=35.8. SN_MIN=30 CHOSEN TO
# SIT IN THAT GAP.
########################################################################

CONT4_RESID_THRESHOLD = 0.15
CONT4_FLAG_SN_MIN = 30.


def CaII_normalize_weighted_bandflag(wave, spec, ivar, SN=None,
                                      resid_threshold=CONT4_RESID_THRESHOLD,
                                      sn_min=CONT4_FLAG_SN_MIN):

    band_masks = {
        'cont1': (wave > CONT1[0]) & (wave < CONT1[1]),
        'cont2': (wave > CONT2[0]) & (wave < CONT2[1]),
        'cont3': (wave > CONT3[0]) & (wave < CONT3[1]),
        'cont4': (wave > CONT4[0]) & (wave < CONT4[1]),
        'cont5': (wave > CONT5[0]) & (wave < CONT5[1]),
    }

    def fit_line(mask_):
        fwave, fspec, fivar = wave[mask_], spec[mask_], ivar[mask_]
        good = np.isfinite(fivar) & (fivar > 0)
        w = np.sqrt(fivar[good])
        z = np.polyfit(fwave[good], fspec[good], 1, w=w)
        return np.poly1d(z)

    # FIT THROUGH THE OTHER 4 BANDS ONLY, THEN CHECK cont4 AGAINST THE
    # EXTRAPOLATION -- THIS FIT IS ALSO THE FALLBACK IF cont4 GETS DROPPED
    m_other4 = band_masks['cont1'] | band_masks['cont2'] | band_masks['cont3'] | band_masks['cont5']
    p_other4 = fit_line(m_other4)

    m4 = band_masks['cont4'] & (ivar > 0) & np.isfinite(ivar)
    frac_resid = np.nan
    if np.sum(m4) > 3:
        predicted4 = p_other4(wave[m4])
        frac_resid = float(np.median((spec[m4] - predicted4) / predicted4))

    sn_ok = (SN is None) or (SN > sn_min)
    drop_cont4 = sn_ok and np.isfinite(frac_resid) and (abs(frac_resid) > resid_threshold)

    if drop_cont4:
        fit = p_other4(wave)
    else:
        p_all5 = fit_line(m_other4 | band_masks['cont4'])
        fit = p_all5(wave)

    nwave = wave
    nspec = spec / fit
    nivar = ivar * fit**2

    return nwave, nspec, nivar, drop_cont4, frac_resid


########################################################################
# GENERAL LEAVE-ONE-OUT, CHI-SQUARE VERSION -- REPLACES THE cont4-
# SPECIFIC, S/N-GATED VERSION ABOVE. TESTS ALL 5 BANDS (OR THE 4
# AVAILABLE ONES, IF cont5 OR ANOTHER BAND HAS NO DATA) FOR BEING AN
# OUTLIER FROM A SHARED LINEAR TREND, NOT JUST cont4.
#
# KEY FIX OVER THE EARLIER z-SCORE ATTEMPT: A NAIVE sqrt(N)-SCALED
# UNCERTAINTY (WHETHER FROM ivar OR FROM EMPIRICAL PIXEL SCATTER)
# BADLY UNDERESTIMATES THE TRUE ERROR ON A BAND'S MEAN, BECAUSE PIXELS
# WITHIN A BAND AREN'T INDEPENDENT (CORRELATED CONTINUUM WIGGLES, LSF,
# FLAT-FIELDING). INSTEAD, EACH BAND'S MEAN AND ITS UNCERTAINTY ARE
# ESTIMATED VIA A BLOCK BOOTSTRAP (RESAMPLING CONTIGUOUS CHUNKS OF
# PIXELS, NOT PIXEL-BY-PIXEL) -- THIS CAPTURES THE REAL CORRELATION
# STRUCTURE WITHOUT ASSUMING INDEPENDENCE, AND NEEDS NO EXTERNAL NOISE
# MODEL.
#
# FOR EACH AVAILABLE BAND: FIT A WEIGHTED LINE THROUGH THE *OTHER*
# BANDS' BOOTSTRAP MEANS, PREDICT THE HELD-OUT BAND, AND COMPUTE ITS
# chi2 = (actual-predicted)^2 / (bootstrap_var + fit_prediction_var).
# THE WORST (LARGEST chi2) BAND IS FLAGGED ONLY IF IT CLEARS A
# SIGNIFICANCE THRESHOLD *AND* IS CLEARLY SEPARATED FROM THE SECOND-
# WORST BAND'S chi2 -- SAME TWO-PART STRUCTURE AS THE 2026-08-06
# SESSION'S ORIGINAL BAND-EXCLUSION CRITERION (15% AND 2x THE SECOND-
# WORST), NOW IN PROPERLY-CALIBRATED chi2 UNITS INSTEAD OF A RAW
# PERCENTAGE.
########################################################################

ALL_BANDS = {'cont1': CONT1, 'cont2': CONT2, 'cont3': CONT3, 'cont4': CONT4, 'cont5': CONT5}
MIN_PIX_PER_BAND = 4
N_BOOT = 400
BLOCK_SIZE = 4
CHI2_SIG_THRESHOLD = 50.   # RAISED FROM 6.63 (1-dof, p=0.01) -> 45 AFTER
# REVIEWING 25 REAL EXAMPLES (clean gap between chi2=31.6 false positive
# and chi2=66.3 confirmed real case) -> 50 AFTER REVIEWING A FURTHER
# 27-mask/11-trigger BATCH, USER PREFERENCE.
CHI2_SEPARATION_RATIO = 2.0


def _block_bootstrap_mean(x, block_size, n_boot, rng):
    '''BLOCK BOOTSTRAP: RESAMPLE x IN CONTIGUOUS CHUNKS OF block_size (WITH
    REPLACEMENT, WRAPPING AT THE EDGE) TO PRESERVE SHORT-RANGE CORRELATION,
    REBUILD AN ARRAY OF THE SAME LENGTH, AND RETURN THE MEAN OF EACH
    RESAMPLE -- THE SPREAD OF THESE MEANS IS THE EMPIRICAL UNCERTAINTY ON
    THE BAND'S OWN MEAN, PROPERLY ACCOUNTING FOR WITHIN-BAND CORRELATION
    (UNLIKE A PLAIN std(x)/sqrt(N), WHICH ASSUMES INDEPENDENT PIXELS).'''
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    starts_pool = np.arange(n)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
        resampled = x[idx.ravel()][:n]
        boot_means[i] = np.mean(resampled)
    return boot_means


MIN_COVERAGE_FRAC = 0.90   # cont4/cont5: DROP A BAND IF MORE THAN 10% OF
# ITS EXPECTED PIXELS (FROM THE SPECTRUM'S OWN MEDIAN DISPERSION) ARE
# MISSING -- EITHER BECAUSE THE WAVELENGTH GRID DOESN'T REACH PART OF THE
# BAND (VIGNETTING TRIM, CHIP GAP) OR BECAUSE PIXELS WITHIN IT ARE MASKED
# (ivar<=0). CAUGHT umi6/SERENDIP (chi2=49.4, just under the significance
# threshold): cont5 100% MISSING, cont4 21% MISSING -- THE LEAVE-ONE-OUT
# TEST WAS LIKELY REACTING TO cont4's PARTIAL, EDGE-TRUNCATED BAND MEAN,
# NOT A GENUINE LINEARITY VIOLATION.
MIN_COVERAGE_FRAC_INNER = 0.75   # cont1/cont2/cont3: LOOSER (25% MISSING
# ALLOWED) -- THESE BANDS AREN'T NEAR A VIGNETTED EDGE, SO A GAP THERE IS
# LESS LIKELY TO BE THE SAME SYSTEMATIC ROLL-OFF EFFECT; A TIGHTER CUT
# WAS DROPPING THEM UNNECESSARILY OFTEN (E.G. FROM AN ISOLATED MASKED
# PIXEL RUN OR A MINOR CHIP-GAP OVERLAP).
INNER_BANDS = ('cont1', 'cont2', 'cont3')


def band_level_stats(wave, spec, ivar, band_range, dwave=None, n_boot=N_BOOT,
                      block_size=BLOCK_SIZE, rng=None, min_coverage_frac=MIN_COVERAGE_FRAC):
    '''RETURNS (mean_wave, mean_flux, bootstrap_sigma, n_pixels) FOR ONE
    CONTINUUM BAND, OR None IF TOO FEW VALID PIXELS OR INSUFFICIENT
    WAVELENGTH COVERAGE (min_coverage_frac OF THE BAND'S EXPECTED PIXELS
    REQUIRED).'''
    if rng is None:
        rng = np.random.default_rng(12345)
    if dwave is None:
        dwave = float(np.median(np.diff(np.sort(wave))))

    m = (wave > band_range[0]) & (wave < band_range[1]) & (ivar > 0) & np.isfinite(ivar)
    n = int(np.sum(m))
    if n < MIN_PIX_PER_BAND:
        return None

    expected_n = (band_range[1] - band_range[0]) / dwave
    if n < min_coverage_frac * expected_n:
        return None

    bwave, bspec = wave[m], spec[m]
    mean_wave = float(np.mean(bwave))
    mean_flux = float(np.mean(bspec))

    boot_means = _block_bootstrap_mean(bspec, min(block_size, n), n_boot, rng)
    sigma = float(np.std(boot_means))
    if sigma <= 0 or not np.isfinite(sigma):
        return None

    return mean_wave, mean_flux, sigma, n


CANDIDATE_DROP_BANDS = ('cont4', 'cont5')


def CaII_normalize_weighted_looflag(wave, spec, ivar, rng=None,
                                     chi2_threshold=CHI2_SIG_THRESHOLD,
                                     separation_ratio=CHI2_SEPARATION_RATIO,
                                     candidate_bands=CANDIDATE_DROP_BANDS):
    '''LEAVE-ONE-OUT, BOOTSTRAP-chi2 VERSION -- SEE MODULE-LEVEL COMMENT
    ABOVE. ONLY cont4/cont5 ARE EVER CANDIDATES TO BE DROPPED (cont1-3
    ALWAYS TRUSTED/INCLUDED) -- TESTING ALL 5 BANDS AS CANDIDATES LET
    cont1/cont3 GET FLAGGED SPURIOUSLY, INCREASINGLY OFTEN AT HIGH S/N
    (A STRAIGHT LINE IS NEVER A PERFECT MODEL, SO GIVEN ENOUGH S/N ANY
    BAND CAN LOOK "SIGNIFICANT"; RESTRICTING TO THE TWO BANDS WITH AN
    ACTUAL DIAGNOSED PHYSICAL MECHANISM (VIGNETTING/THROUGHPUT ROLL-OFF
    NEAR THE RED EDGE) AVOIDS THAT). RETURNS (nwave, nspec, nivar,
    dropped_band_or_None, chi2_by_band dict, band_stats dict).'''
    if rng is None:
        rng = np.random.default_rng(12345)

    dwave = float(np.median(np.diff(np.sort(wave))))

    stats = {}
    for name, rng_ in ALL_BANDS.items():
        min_cov = MIN_COVERAGE_FRAC_INNER if name in INNER_BANDS else MIN_COVERAGE_FRAC
        s = band_level_stats(wave, spec, ivar, rng_, dwave=dwave, rng=rng, min_coverage_frac=min_cov)
        if s is not None:
            stats[name] = s   # (mean_wave, mean_flux, sigma, n)

    available = list(stats.keys())
    coverage_dropped = [name for name in ALL_BANDS if name not in available]
    candidates = [b for b in candidate_bands if b in available]
    dropped_band = None
    chi2_by_band = {}

    if len(available) >= 4:
        for leave_out in candidates:
            fit_bands = [b for b in available if b != leave_out]
            xs = np.array([stats[b][0] for b in fit_bands])
            ys = np.array([stats[b][1] for b in fit_bands])
            sigmas = np.array([stats[b][2] for b in fit_bands])
            w = 1. / sigmas

            (a, b_), cov = np.polyfit(xs, ys, 1, w=w, cov=True)

            # MODEL-ERROR FLOOR: A STRAIGHT LINE IS ONLY AN APPROXIMATION TO
            # THE TRUE CONTINUUM EVEN FOR A PERFECTLY NORMAL SLIT. USE THE
            # OTHER (TRUSTED) BANDS' OWN MEAN-SQUARED RESIDUAL ABOUT THIS
            # SAME FIT AS AN EMPIRICAL FLOOR ON HOW MUCH DEVIATION-FROM-
            # LINEAR IS "NORMAL" FOR THIS SPECIFIC SLIT -- WITHOUT THIS, A
            # LOW-PIXEL-COUNT BAND'S BOOTSTRAP sigma CAN BE ARTIFICIALLY
            # TIGHT (TOO FEW BLOCKS TO SAMPLE THE TRUE TAIL), MAKING chi2
            # BLOW UP EVEN FOR GENUINELY FINE SLITS.
            model_resid = ys - (a * xs + b_)
            model_var = float(np.mean(model_resid**2))

            x_out, y_out, sigma_out, _ = stats[leave_out]
            predicted = a * x_out + b_
            var_fit = cov[0, 0] * x_out**2 + cov[1, 1] + 2 * x_out * cov[0, 1]
            var_combined = sigma_out**2 + max(var_fit, 0) + model_var
            chi2_by_band[leave_out] = float((y_out - predicted)**2 / var_combined)

        if chi2_by_band:
            worst = max(chi2_by_band, key=chi2_by_band.get)
            worst_chi2 = chi2_by_band[worst]
            others = sorted([v for k, v in chi2_by_band.items() if k != worst])
            # WITH ONLY 2 POSSIBLE CANDIDATES, "others" IS EMPTY WHENEVER
            # cont5 (OR cont4) IS ALREADY MISSING -- SKIP THE SEPARATION
            # REQUIREMENT THEN, THERE'S NOTHING TO SEPARATE FROM
            passes_separation = (not others) or (worst_chi2 > separation_ratio * max(others[-1], 1e-6))

            if (worst_chi2 > chi2_threshold) and passes_separation:
                dropped_band = worst

    # BUILD THE FINAL FIT FROM PIXELS (NOT JUST BAND-LEVEL POINTS) OF
    # WHICHEVER BANDS SURVIVE, SAME AS THE OTHER FUNCTIONS IN THIS MODULE
    band_masks = {name: (wave > lo) & (wave < hi) for name, (lo, hi) in ALL_BANDS.items()}
    use_bands = [b for b in available if b != dropped_band]
    m = np.zeros_like(wave, dtype=bool)
    for b in use_bands:
        m |= band_masks[b]

    fwave, fspec, fivar = wave[m], spec[m], ivar[m]
    good = np.isfinite(fivar) & (fivar > 0)
    wgt = np.sqrt(fivar[good])
    z = np.polyfit(fwave[good], fspec[good], 1, w=wgt)
    p = np.poly1d(z)
    fit = p(wave)

    nwave = wave
    nspec = spec / fit
    nivar = ivar * fit**2

    return nwave, nspec, nivar, dropped_band, chi2_by_band, stats, coverage_dropped
