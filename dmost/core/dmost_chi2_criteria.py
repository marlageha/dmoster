import numpy as np
from scipy.optimize import least_squares

########################################################################
# CHI2-VS-S/N OUTLIER FLAG FOR THE GL-gvary ADAPTIVE MODEL, SAME
# APPROACH AS criterion A IN chi2_flag_criteria_design.py (DESIGNED
# AGAINST PRODUCTION'S OWN Gauss+Lorentzian chi2 DISTRIBUTION) --
# REFIT HERE TO THIS MODEL'S OWN chi2-vs-S/N POPULATION RATHER THAN
# REUSING PRODUCTION'S CURVE, SINCE A DIFFERENT MODEL FIT TO THE SAME
# DATA CAN HAVE A DIFFERENT chi2-vs-S/N RELATION.
#
# FORM: chi2_envelope(SN) = floor + b*SN^2 (SAME PHYSICAL MOTIVATION --
# A FIXED FRACTIONAL TEMPLATE-MISMATCH TERM GROWING AS SN^2 ONCE PAST
# THE NOISE-DOMINATED REGIME), FIT TO THE 98TH PERCENTILE OF chi2 IN
# LOG-SPACED S/N BINS, IN LOG-SPACE (EQUAL RELATIVE WEIGHT PER BIN, SO
# THE SPARSE HIGH-S/N TAIL DOESN'T GET SWAMPED BY THE BULK AT LOW S/N).
########################################################################


def curve_form(sn, floor, b):
    return floor + b*sn**2


def fit_criterion_curve(sn, chi2, percentile=98, n_bins=20):
    '''Fit floor + b*SN^2 to the given percentile of chi2 in log-spaced S/N bins.'''
    logsn = np.log10(sn)
    bins = np.logspace(logsn.min(), logsn.max(), n_bins)
    bin_centers, bin_pct = [], []
    for i in range(len(bins)-1):
        m = (sn >= bins[i]) & (sn < bins[i+1])
        if np.sum(m) > 5:
            bin_centers.append(np.sqrt(bins[i]*bins[i+1]))
            bin_pct.append(np.percentile(chi2[m], percentile))
    bin_centers, bin_pct = np.array(bin_centers), np.array(bin_pct)

    def resid_log(p, x, y):
        floor, b = p
        model = curve_form(x, floor, b)
        return np.log(y) - np.log(np.maximum(model, 1e-6))

    fit = least_squares(resid_log, x0=[2.0, 1e-4], args=(bin_centers, bin_pct),
                         bounds=([0, 0], [np.inf, np.inf]))
    return fit.x, bin_centers, bin_pct
