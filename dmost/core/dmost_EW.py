#!/usr/bin/env python

import numpy as np
import os,sys
import time


import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from astropy.table import Table, Column
from astropy import units as u
from astropy.io import ascii,fits

import glob
import warnings

from dmost import dmost_utils
from dmost.core import dmost_continuum, dmost_chi2_criteria

import scipy.ndimage as scipynd
from scipy.optimize import curve_fit


DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')

# CHI2-VS-S/N ENVELOPE FOR THE ADAPTIVE GL-gvary CaT FIT (dmost_chi2_criteria.
# curve_form: floor + b*SN^2), FIT TO THE FULL FLATPRIOR DATABASE'S OWN
# chi2-vs-S/N POPULATION -- FIXED HERE, NOT REFIT PER MASK.
CHI2_ENVELOPE_FLOOR = 3.448
CHI2_ENVELOPE_B     = 0.008564


######################################################
def mk_EW_plots(pdf, this_slit, nwave,nspec, nawave, naspec, cat_fit, mg_fit, na_fit,p_na,p_mg,CaT_chi2,cat_stage):

    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4,figsize=(22,5))

    ax1.plot(nwave,nspec,label = 'objname= {}'.format(this_slit['objname']))
    ax1.set_xlim(8484, 8560)
    ax1.plot(nwave,cat_fit,'r')
    ax1.set_title('SN= {:0.1f} v = {:0.1f}'.format(this_slit['collate1d_SN'], this_slit['dmost_v']))
    ax1.legend(loc=3,title='CaT Chi2 = {:0.1f}, stage = {}'.format(CaT_chi2,cat_stage))

    ax2.plot(nwave,nspec)
    ax2.set_xlim(8630,8680)
    tt = 'teff = {}  feh = {}'.format(this_slit['chi2_teff'],this_slit['chi2_feh'])
    ax2.plot(nwave,cat_fit,'r',label=tt)
    ax2.set_title('CaT EW= {:0.2f}  err={:0.2f}'.format(this_slit['cat'],this_slit['cat_err']))
    ax2.legend(loc=3)


    mg_label = 'sig = {:0.2f}, cen = {:0.1f}'.format(p_mg[2],p_mg[3])
    ax3.plot(nwave,nspec,label=mg_label)
    ax3.plot(nwave,mg_fit,'r')
    ax3.set_title(' MgI EW = {:0.2f}  err={:0.2f}'.format(this_slit['mgI'],this_slit['mgI_err']))
    ax3.set_xlim(8802,8811)
    ax3.legend(loc=3)


    na_label = 'sig = {:0.2f}, ratio= {:0.1f}, cen = {:0.1f}'.format(p_na[2],p_na[3],p_na[1])
    ax4.plot(nawave,naspec,label=na_label)
    ax4.set_xlim(8150,8220)
    ax4.plot(nwave,na_fit,'r')
    ax4.set_title('Na1 EW={:0.2f} err={:0.2f}'.format(this_slit['naI'],this_slit['naI_err']))
    ax4.legend(loc=3)

    ymax = 1.2
    if this_slit['collate1d_SN'] < 20:
        ymax = 1.5

    ax1.set_ylim(0,ymax)
    ax2.set_ylim(0,ymax)
    ax3.set_ylim(0,ymax)
    ax4.set_ylim(0,ymax)

    pdf.savefig()
    plt.close('all')


    return pdf

########################################
def NaI_normalize(wave,spec,ivar):

    # 21AA window centered on 8190AA
    wred  = [8203., 8228.]
    wblue = [8155., 8175.]
    waver = (wred[0] + wred[1])/2.
    waveb = (wblue[0] + wblue[1])/2.

    mred = (wave > wred[0]) & (wave < wred[1])
    mblue = (wave > wblue[0]) & (wave < wblue[1])

    # DETERMINE WEIGHTED MEAN OF BLUE/RED PSEUDO-CONTINUUM BAND
    # DON"T CALCULATE IF DATA DOESN"T EXIST

    fit = 0
    if (np.sum(mblue) != 0) & (np.sum(mred) != 0):

        m = (spec > np.percentile(spec,20)) & (spec < np.percentile(spec,95))
        sum1 = np.sum(spec[mblue&m] * ivar[mblue&m]**2 )
        sum2 = np.sum( ivar[mblue&m]**2 )
        bcont = sum1 / sum2

        sum1 = np.sum(spec[mred&m] * ivar[mred&m]**2 )
        sum2 = np.sum( ivar[mred&m]**2 )
        rcont = sum1 / sum2


        # DEFINE CONTINUUM LINE BETWEEN RED/BLUE PASSBAND (y=mx+b)
        mline = (rcont - bcont) / (waver - waveb)
        bline = rcont - (mline * waver)
        fit   = (mline * wave) + bline


    # NORMALIZE SPECTRUM
    nwave = wave
    nspec = spec/fit
    nivar = ivar*fit**2

    return nwave,nspec,nivar


######################################
def NaI_double_gauss(x,*p):
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Central value                : p[1]
    #   Standard deviatio            : p[2]
    #   Relative height              : p[3]
    # 8183.3, 8194.8
    return p[4]-p[0]*np.exp(-1.*(x-p[1])**2/(2.*p[2]**2)) \
               -p[3]*p[0]*np.exp(-1.*(x-(p[1]+11.54))**2/(2.*p[2]**2))

########################################
def NaI_guess(x,y):

    N_guess   = np.max(y) - np.min(y)
    wv_guess  = 8183.256
    sig_guess = 0.6
    p0 = [N_guess,wv_guess,sig_guess,1.,1.]

    return p0


########################################
def NaI_fit_EW(wvl,spec,ivar,SN):

    wline = [8172., 8210.5]

    Na1_EW,Na1_EW_err     = -999., -999.
    gfit       = -999*wvl
    p0 = [-999.,-999.,-999.,-999.,-999.]


    mw  = (wvl > wline[0]) & (wvl < wline[1])
    mzero = spec[mw] == 0

    if (np.sum(mzero) < 10):

        # GAUSSIAN FIT
        p0 = NaI_guess(wvl[mw],spec[mw])
        errors = 1./np.sqrt(ivar[mw])

        try:
            p, pcov = curve_fit(NaI_double_gauss,wvl[mw],spec[mw],sigma = errors,p0=p0,\
                   bounds=((0, 8182, 0.4, 1.0,0.95), (2, 8185, 1., 1.6,1.05)))

            perr = np.sqrt(np.diag(pcov))

            # INTEGRATE PROFILE
            Na1_EW1 = (p[0])*(p[2]*np.sqrt(2.*np.pi))
            Na1_EW2 = (p[3])*p[0]*(p[2]*np.sqrt(2.*np.pi))
            Na1_EW  = Na1_EW1+Na1_EW2

            # CALCUALTE ERROR
            tmp1 = p[0] * perr[2]* np.sqrt(2*np.pi)
            tmp2 = p[2] * perr[0]* np.sqrt(2*np.pi)


            tmp3 = p[3] *p[0] * perr[2]* np.sqrt(2*np.pi)
            tmp4 = p[0] *p[2] * perr[3]* np.sqrt(2*np.pi)
            tmp5 = p[3] *p[2] * perr[0]* np.sqrt(2*np.pi)


            Na1_EW_err = np.sqrt(tmp1**2 + tmp2**2 + tmp3**2 + tmp4**2 + tmp5**2)

            # CREATE FIT FOR PLOTTING
            gfit = NaI_double_gauss(wvl,*p)

            if (Na1_EW > 5) | (Na1_EW_err > 5) | (Na1_EW_err == 0):
                Na1_EW     = -999.
                Na1_EW_err = -999.
                p=p0
        except:
            p, pcov = p0, None
            perr = p0


    return Na1_EW,Na1_EW_err,gfit,p

########################################
def MgI_gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Standard deviation           : p[2]
    return p[0]-p[1]*np.exp(-1.*(x-p[3])**2/(2.*p[2]**2))

########################################
def MgI_guess(x,y):
    N_guess   = np.max(y) - np.min(y)
    sig_guess = 0.6
    p0 = [1.,N_guess,sig_guess, 8806.8]

    return p0


########################################
def mgI_EW_fit(wvl,spec,ivar,SN):

    # CALCULATE in +/- 5A of MgI line
    # there is a line at 8804.6 (need to deal with this?)
    mgI_line = 8806.8
    wline = [mgI_line-5.,mgI_line+5.]
    mw    = (wvl > wline[0]) & (wvl < wline[1])

    mg1_EW, mg1_EW_err    = -999., -999.
    mgfit       = -999*wvl
    p0 =  [-999.,-999.,-999.,-999.]


    # GAUSSIAN FIT
    try:
        p0 = MgI_guess(wvl[mw],spec[mw])

        errors = 1./np.sqrt(ivar[mw])
        if SN > 5:
            p, pcov = curve_fit(MgI_gaussian,wvl[mw],spec[mw],sigma = errors,p0=p0, \
                            bounds=((0.5, 0.0, 0.45, 8805.8), (2, 2, 0.9,8807.8)))
        if SN < 5:
            p, pcov = curve_fit(MgI_gaussian,wvl[mw],spec[mw],sigma = errors,p0=p0, \
                            bounds=((0.5, 0.0, 0.45, 8806.2), (2, 2, 0.8,8807.4)))
        perr = np.sqrt(np.diag(pcov))

        # INTEGRATE PROFILE
        mg1_EW = (p[1])*(p[2]*np.sqrt(2.*np.pi))

        # CALCUALTE ERROR
        tmp1 = p[1] * perr[2]* np.sqrt(2*np.pi)
        tmp2 = p[2] * perr[1]* np.sqrt(2*np.pi)
        mg1_EW_err = np.sqrt(tmp1**2 + tmp2**2)

        # CREATE FIT FOR PLOTTING
        mgfit = MgI_gaussian(wvl,*p)
        p3=p[2]

        if (np.abs(mg1_EW) > 10) | (mg1_EW_err == 0.) | (mg1_EW_err > 10.):
            mg1_EW     = -999.
            mg1_EW_err = -999.
    except:
        p=p0


    return mg1_EW,mg1_EW_err,mgfit, p


########################################################################
# SHARED CaT MODEL BUILDING BLOCKS -- USED BY dmost_cat_model.py AND
# dmost_cat_fit.py (THE ADAPTIVE GL-gvary FIT), NOT JUST HERE. KEPT IN
# THIS MODULE SINCE calc_chi2_ew (BELOW) ALREADY LIVED HERE AND THESE ARE
# THE SAME KIND OF LOW-LEVEL, MODEL-AGNOSTIC MATH HELPER.
########################################################################

CAT_LINE_CENTER = 8542.09

# EMPIRICAL CaT LINE-EW RELATIONS (Heiger et al. 2024, ApJ 961, 234, Eqns 2 & 7), 8542 AS ANCHOR.
# USED AS A FALLBACK/COMPARISON REFERENCE -- THE PRODUCTION GL-gvary PRIOR
# ITSELF USES THE FLAT, ZERO-INTERCEPT RATIOS IN dmost_cat_model.py.
EW1_SLOPE, EW1_INT, EW1_SCATTER = 0.41, 0.14, 0.3   # 8498 | 8542
EW3_SLOPE, EW3_INT, EW3_SCATTER = 0.74, 0.16, 0.3   # 8662 | 8542


def _ln_gauss(x, mu, sigma):
    return -0.5*((x-mu)/sigma)**2

def _ln_lognormal(x, mu, sigma):
    if x <= 0:
        return -np.inf
    return -0.5*((np.log(x)-mu)/sigma)**2 - np.log(x)

def _ln_beta_frac(numer, denom, a=2., b=2.):
    # LN BETA(a,b) DENSITY OF f = numer/denom, f IN (0,1)
    if (numer <= 0) | (numer >= denom):
        return -np.inf
    f = numer/denom
    return (a-1.)*np.log(f) + (b-1.)*np.log(1.-f)


########################################
def calc_chi2_ew(wave,spec,ivar,mwindow, fit):

    model = fit[mwindow]
    data  = spec[mwindow]
    ivar  = ivar[mwindow]

    chi2 = np.sum((data - model)**2 * ivar)/np.size(data)

    return chi2


######################################################

def calc_all_EW(data_dir, slits, mask, arg, pdf, rng):

    warnings.simplefilter('ignore')#, OptimizeWarning)

    # DEFERRED IMPORT -- dmost_cat_fit (AND dmost_cat_model, TRANSITIVELY)
    # IMPORT THIS MODULE FOR THE SHARED HELPERS ABOVE, SO IT CAN'T BE
    # IMPORTED AT THIS MODULE'S TOP LEVEL WITHOUT A CIRCULAR IMPORT.
    from dmost.core import dmost_cat_fit

    # READ COADDED DATA
    jhdu = fits.open(data_dir+'/collate1d_flex/'+slits['collate1d_filename'][arg])

    jwave,jflux,jivar, SN = dmost_utils.load_coadd_collate1d(slits[arg],jhdu)
    wave_lims = dmost_utils.vignetting_limits(slits[arg],0,jwave)

    wvl  = jwave[wave_lims]
    flux = jflux[wave_lims]
    ivar = jivar[wave_lims]

    redshift = slits['dmost_v'][arg] / 299792.
    wave     = wvl / (1.0+redshift)

    wlims = (wvl > 8100) & (wvl < 8700)
    if (np.sum(flux[wlims] > 0) > 1200) & (np.max(wave) > 8562):


        #####################
        # CaT CONTINUUM: ivar-WEIGHTED, LEAVE-ONE-OUT BAND-DROPPING FIT
        nwave, nspec, nivar, dropped_band, chi2_by_band, band_stats, coverage_dropped = \
            dmost_continuum.CaII_normalize_weighted_looflag(wave, flux, ivar, rng=rng)

        # CaT WINDOWS, CENARRO (2001) TABLE 4: lines = [8498.02,8542.09,8662.14]
        mw1 = (nwave > 8484) & (nwave < 8513)
        mw2 = (nwave > 8522) & (nwave < 8562)
        mw3 = (nwave > 8642) & (nwave < 8682)
        mw  = mw1 | mw2 | mw3

        # ADAPTIVE STAGE 0/A/B GL-gvary FIT (DECOUPLED WIDTHS, FREE
        # CENTERS AND LOW-CaT ESCALATION AS NEEDED -- SEE dmost_cat_fit.py)
        result = dmost_cat_fit.fit_adaptive_GL_gvary(nwave, nspec, nivar, mw)

        CaT_fit  = result['fit']
        CaT_chi2 = result['chi2']

        theta14 = np.zeros(14)
        theta14[:len(result['theta'])] = result['theta']

        slits['cat'][arg]             = result['cat']
        slits['cat_err'][arg]         = result['cat_err']
        slits['cat_chi2'][arg]        = CaT_chi2
        slits['cat_theta'][arg]       = theta14
        slits['cat_all'][arg]         = result['ew']
        slits['cat_all_err'][arg]     = result['ew_err']
        slits['cat_adapt_stage'][arg] = result['stage']
        slits['cat_f_acc'][arg]       = result['facc']
        slits['cat_converge'][arg]    = result['convg']
        slits['cat_good'][arg]        = 1 if (result['convg'] > 0) and (0.2 < result['facc'] < 0.8) else 0

        # FLAG EW DISASTERS
        if (CaT_chi2 > 50) & (slits['collate1d_SN'][arg] < 200):
            slits['cat_err'][arg] = -999.
        if (CaT_chi2 > 30) & (slits['collate1d_SN'][arg] < 100):
            slits['cat_err'][arg] = -999.


        ##########################
        # CALCULATE MgI LINES -- SHARES THE (NEW) CaT CONTINUUM
        MgI_EW,MgI_EW_err, MgI_fit,p_mg  = mgI_EW_fit(nwave,nspec,nivar,SN)

        slits['mgI'][arg]     = MgI_EW
        slits['mgI_err'][arg] = MgI_EW_err

        #############################
        # NaI LINES -- OWN INDEPENDENT NORMALIZATION, UNCHANGED
        nawave,naspec,naivar            = NaI_normalize(wave,flux,ivar)
        NaI_EW,NaI_EW_err, NaI_fit,p_na = NaI_fit_EW(nawave,naspec,naivar,SN)
        slits['naI'][arg]     = NaI_EW
        slits['naI_err'][arg] = NaI_EW_err


        # ADD EW SYSTEMATIC ERRORS HERE
        slits[arg] = ew_sys_errors(slits[arg])

        mk_EW_plots(pdf, slits[arg], nwave, nspec, nawave, naspec, CaT_fit, MgI_fit, NaI_fit,p_na,p_mg, \
                    CaT_chi2, result['stage'])


    return slits

########################################################
def ew_sys_errors(slit):

    slit['naI_err_rand'] = slit['naI_err']
    slit['mgI_err_rand'] = slit['mgI_err']
    slit['cat_err_rand'] = slit['cat_err']

    # FOR NaI and MgI, multiplier only
    if (slit['naI_err'] > 0):
        slit['naI_err'] =  np.sqrt((0.7*slit['naI_err'])**2 + 0.05**2)


    if slit['mgI_err'] > 0:
        slit['mgI_err'] =  np.sqrt((0.7*slit['mgI_err'])**2 + 0.05**2)

    # CaT's systematic mult/floor correction is applied downstream, in the
    # combined-masks step (calibrated against the whole flatprior database's
    # own chi2-vs-S/N population) -- not here, per-mask.

    return slit

######################################################

def run_coadd_EW(data_dir, slits, mask):
    '''
    CALCUALTE EW USING COADDED SPECTRA
    '''

    logfile      = data_dir + mask['maskname'][0]+'_dmost.log'
    log          = open(logfile,'a')


    file  = data_dir+'/QA/ew_'+mask['maskname'][0]+'.pdf'
    pdf   = matplotlib.backends.backend_pdf.PdfPages(file)

    # SEEDED RNG FOR THE CONTINUUM'S BLOCK-BOOTSTRAP BAND STATS --
    # REPRODUCIBLE ACROSS RE-RUNS OF THE SAME MASK
    rng = np.random.default_rng(2026)

    # ENSURE ALL CaT/MgI/NaI COLUMNS EXIST -- BACKWARD COMPATIBILITY FOR
    # MASKFILES CREATED BEFORE THE 14-SLOT cat_theta / ADAPTIVE-STAGE
    # SCHEMA (dmost_create_maskfile.create_slits DEFINES THESE FOR NEW
    # MASKFILES ALREADY, SO THIS IS A NO-OP THERE)
    n = len(slits)
    for col, default in [('cat_all', -999.*np.ones((n, 3))), ('cat_all_err', -999.*np.ones((n, 3))),
                          ('cat_theta', -999.*np.ones((n, 14))), ('cat_err_rand', -999.),
                          ('mgI', -999.), ('mgI_err', -999.), ('mgI_err_rand', -999.),
                          ('naI', -999.), ('naI_err', -999.), ('naI_err_rand', -999.)]:
        if col not in slits.colnames:
            slits[col] = default
    for col in ['cat', 'cat_err', 'cat_chi2', 'cat_f_acc', 'cat_converge', 'cat_good',
                'cat_chi2_flag', 'cat_adapt_stage']:
        if col not in slits.colnames:
            slits[col] = -999.


    m = (slits['dmost_v_err'] > 0) & (slits['marz_flag'] < 3)
    dmost_utils.printlog(log,'{} EW estimates for {} slits '.format(mask['maskname'][0],np.sum(m)))

    # FOR EACH SLIT
    for ii,slt in enumerate(slits):


        if (slt['dmost_v_err'] > 0) & (slt['marz_flag'] < 2):


            # RUN ADAPTIVE CaT FIT + MgI/NaI ON COADD
            slits = calc_all_EW(data_dir, slits, mask, ii, pdf, rng)


    # FLAG SLITS WHERE cat_chi2 IS UNREASONABLY LARGE FOR THEIR S/N --
    # FIXED S/N-DEPENDENT ENVELOPE (dmost_chi2_criteria.curve_form),
    # NEEDS ALL SLITS ALREADY FIT SO IT RUNS ONCE HERE, MASK-LEVEL
    valid_chi2 = np.array(slits['cat_chi2']) > 0
    if np.sum(valid_chi2) > 0:
        sn_arr   = np.array(slits['collate1d_SN'])
        chi2_arr = np.array(slits['cat_chi2'])
        flagged  = np.zeros(len(slits), dtype=bool)
        flagged[valid_chi2] = chi2_arr[valid_chi2] > dmost_chi2_criteria.curve_form(
            sn_arr[valid_chi2], CHI2_ENVELOPE_FLOOR, CHI2_ENVELOPE_B)
        slits['cat_chi2_flag'][valid_chi2] = np.where(flagged[valid_chi2], 1.0, 0.0)
        slits['cat_err'][flagged] = -999.


    pdf.close()
    plt.close('all')
    log.close()



    return slits, mask
