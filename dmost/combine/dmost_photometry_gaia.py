#!/usr/bin/env python

import numpy as np
import os,sys
    
from astropy.coordinates import SkyCoord


from astropy.table import Table
from astropy import units as u
from astropy.io import ascii,fits

from sfdmap2 import sfdmap
from scipy.stats import truncnorm


###########################################
def get_ebv(allspec,pandas = 0, sdss = 0, ps1 = 0):

    DEIMOS_RAW  = os.getenv('DEIMOS_RAW')
    print(DEIMOS_RAW)

    # scaling=1.0 -> raw SFD98, no Schlafly & Finkbeiner 2011 recalibration.
    # DR11's own 'ebv' column is raw SFD98 (confirmed 2026-08-24, M. Geha,
    # against the Legacy Survey docs); sfdmap2's default scaling=0.86 applies
    # the SF11 recalibration and was silently mismatched with DR11 and with
    # the DECam coefficients below, which were calibrated for SFD98.
    sdf_ext = sfdmap.SFDMap(DEIMOS_RAW+'SFDmaps/', scaling=1.0)
    EBV     = sdf_ext.ebv(allspec['RA'],allspec['DEC'])

    allspec['EBV'] = EBV

    #https://www.legacysurvey.org/dr10/catalogs/#galactic-extinction-coefficients
    # table 6:   https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103/pdf

    # THESE ARE WHAT DELVE AND DES ARE USING
    #   Ar = 2.140 * allspec['EBV']
    #   Ag = 3.185 * allspec['EBV']

    # THESE ARE USED in ls_dr10, ls_dr11, and decaps -- DECam-native, SFD98 scale
    Ai = 1.58  * allspec['EBV']
    Ar = 2.165 * allspec['EBV']
    Ag = 3.214 * allspec['EBV']

    # From Ibata+ 2014 (PANDAS/CFHT MegaCam system), scaled by 0.86 (2026-
    # 08-25, M. Geha) -- Ibata+2014's coefficients were empirically
    # validated (2026-08-24) using SF11-recalibrated E(B-V), not the raw
    # SFD98 this function now returns everywhere. See
    # research_log_2026-08-25.html.
    if (pandas == 1):
        Ai = 2.080 * 0.86 * allspec['EBV']
        Ag = 3.793 * 0.86 * allspec['EBV']

    # SDSS native ugriz -- Schlafly & Finkbeiner 2011, Table 6, applied to
    # SFD98 E(B-V) (confirmed 2026-08-24 to match SDSS DR14's own
    # EXTINCTION_G/R/I columns exactly). Deredden SDSS mags in this native
    # system BEFORE any transform_sdss2decals color transform, not after --
    # see research_log_2026-08-24.html for justification.
    if (sdss == 1):
        Ai = 1.698 * allspec['EBV']
        Ar = 2.285 * allspec['EBV']
        Ag = 3.303 * allspec['EBV']

    # PS1 native grizy -- Schlafly & Finkbeiner 2011, Table 6 (2026-08-25,
    # M. Geha), applied to SFD98 E(B-V). Deredden PS1 mags in this native
    # system BEFORE transform_ps12decals, same reasoning as sdss=1 above.
    if (ps1 == 1):
        Ai = 1.682 * allspec['EBV']
        Ar = 2.271 * allspec['EBV']
        Ag = 3.172 * allspec['EBV']

    return allspec, Ar, Ag, Ai
    

###########################################
def calc_MV_star(allspec,obj):
    
    # DISTANCE MODULUS AND REDDENING
    dmod = 5.*np.log10(obj['Dist_kpc']*1e3) - 5.


    # SDSS Jordi 2006
    # V = allspec['gmag_o'] - 0.565*gr_o -0.016

    # Abbott et al. 2021 (DES DR2), ApJS 255, 20
    # https://iopscience.iop.org/article/10.3847/1538-4365/ac00b3
    # Appendix B, Eq. B8 -- piecewise transformation (confirmed directly
    # against the paper 2026-08-25, M. Geha; NOT Appendix A, which is an
    # unrelated weight-averaging appendix in that paper)

    gr_o =  allspec['gmag_o'] - allspec['rmag_o']
    V    = np.zeros(np.size(gr_o))

    m1    = gr_o <= 0.2
    V[m1] =  allspec['gmag_o'][m1] - 0.465*gr_o[m1] -0.02

    m2    = (gr_o > 0.2) & (gr_o <= 0.7)
    V[m2] =  allspec['gmag_o'][m2] - 0.496*gr_o[m2] -0.015

    m3    = gr_o > 0.7
    V[m3] = allspec['gmag_o'][m3] - 0.445*gr_o[m3] -0.062

    # Only update stars with matched photometry
    m = allspec['rmag_o'] > 0
    allspec['MV_o'][m] = V[m] - dmod


    # PANDAS stars without a real borrowed r (see borrow_r_for_pandas)
    # have no rmag_o and so were skipped above. Estimate (g-r) from
    # measured (g-i) via the dwarf-galaxy stellar locus (r-i =
    # 0.4271*(g-r) - 0.0612, fit 2026-08-24, research_log_2026-08-24.html)
    # and apply the same Abbott et al. 2021 g,r->V transform as above.
    # (2026-08-25 fix: previously this unconditionally overwrote every
    # PANDAS star's V using a Huxor+08 g,i relation that assumed gmag_o/
    # imag_o were still raw, untransformed PANDAS magnitudes -- they are
    # now DR11-system values (own g,i fits, research_log_2026-08-25.html),
    # so that formula no longer applies and stars with real borrowed r are
    # left to the correct g,r-based transform above instead.)
    if obj['Phot'] == 'pandas':
        m_needs_v = (allspec['gmag_o'] > 0) & (allspec['imag_o'] > 0) & (allspec['rmag_o'] < 0)

        if np.any(m_needs_v):
            gi_est = allspec['gmag_o'][m_needs_v] - allspec['imag_o'][m_needs_v]
            gr_est = (gi_est + 0.0612) / 1.4271

            V_est = np.zeros(np.sum(m_needs_v))
            e1 = gr_est <= 0.2
            V_est[e1] = allspec['gmag_o'][m_needs_v][e1] - 0.465*gr_est[e1] - 0.02

            e2 = (gr_est > 0.2) & (gr_est <= 0.7)
            V_est[e2] = allspec['gmag_o'][m_needs_v][e2] - 0.496*gr_est[e2] - 0.015

            e3 = gr_est > 0.7
            V_est[e3] = allspec['gmag_o'][m_needs_v][e3] - 0.445*gr_est[e3] - 0.062

            allspec['MV_o'][m_needs_v] = V_est - dmod


    
    ########## FIX MISSING PHOTOMETRY #############

   # GMAG ONLY
    mg1 = (allspec['rmag_err'] <= -999.0) & (allspec['gmag_err'] >0)
    mg2 = (allspec['rmag_o'] <= -999.0) & (allspec['gmag_o'] >0)
    V[mg1|mg2] =  allspec['gmag_o'][mg1|mg2]   # ASSUME gr = 0.3
    allspec['MV_o'][mg1|mg2] = V[mg1|mg2] - dmod

    # RMAG ONLY
    mr1 = (allspec['rmag_err'] > 0) & (allspec['gmag_err'] <= -999.0)
    mr2 = (allspec['rmag_o'] > 0)   & (allspec['gmag_o'] <= -999.0)
    V[mr1|mr2] =  allspec['rmag_o'][mr1|mr2]  + 0.3  # ASSUME gr = 0.3
    allspec['MV_o'][mr1|mr2] = V[mr1|mr2] - dmod
    
   # IF NO PHOTOMETRY EXISTS
    m = (allspec['rmag_err'] <= -999.0) & (allspec['gmag_err'] <= -999.0)
    allspec['MV_o'][m] = -999.0



    return allspec

###########################################
def calc_rproj(allspec,obj):

    sc_gal = SkyCoord(obj['RA'],obj['Dec'], unit=(u.deg, u.deg),distance = obj['Dist_kpc']*u.kpc)


    # CALCULATE STAR RADIUS FROM OBJECT CENTER
    sc_all = SkyCoord(allspec['RA'],allspec['DEC'], unit=(u.deg, u.deg),distance = obj['Dist_kpc']*u.kpc)

    sep = sc_all.separation(sc_gal)
    allspec['rproj_arcm'] = sep.arcmin

    sep3d = sc_all.separation_3d(sc_gal)
    allspec['rproj_kpc'] = sep3d.kpc 


    return allspec

###########################################
def truncated_normal(loc, scale, size, lowerbound = 0.1):
    a = (lowerbound - loc) / scale
    b = np.inf
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)


def calculate_FeH(V0, V0err, ew_cat, ew_cat_err, use_truncnorm = True):

    # MAG AND DISTANCE ERROR-- could improve this
    V0err     = np.sqrt(V0err**2 + 0.1**2)  # add dmod error
    Vmag0_abs = np.random.normal(loc=V0, scale=V0err, size=10000)
    
    
    # Distribute equivalent widths as Truncated Normal (cutoff at zero)
    ew_cat_distrib = truncated_normal(loc=ew_cat, scale=ew_cat_err, lowerbound = 0., size=10000)
    

    # #######Carrera 2013##########
    # [value, error]  using M_V
    #a = [-3.45,0.04]
    #b = [0.16,0.01]
    #c = [0.41,0.004]
    #d = [-0.53,0.11]
    #e = [0.019, 0.002]
    #FeH = a[0] + b[0]* mag + c[0]*CaT + d[0]*CaT**(-1.5) + e[0]*CaT*mag

    #a = np.random.normal(loc=-3.45, scale=0.04, size=5000)
    #b = np.random.normal(loc=0.16,  scale=0.01, size=5000)
    #c = np.random.normal(loc=0.41,  scale=0.004,size=5000)
    #d = np.random.normal(loc=-0.53, scale=0.11, size=5000)
    #e = np.random.normal(loc=0.019, scale=0.002,size=5000)


    # Update to Navabi+25 Coefficients
    a = np.random.normal(loc=-3.1, scale=0.05, size=10000)
    b = np.random.normal(loc=0.09, scale=0.02, size=10000)
    c = np.random.normal(loc=0.33, scale=0.01, size=10000)
    d = np.random.normal(loc=-1.01, scale=0.13, size=10000)
    e = np.random.normal(loc=0.02, scale=0.01, size=10000)

    # Calculate [Fe/H]
    FeH = a + (b * Vmag0_abs) + (c * ew_cat_distrib) + (d * ew_cat_distrib**(-1.5)) + \
                                (e * ew_cat_distrib * Vmag0_abs)
    
    masked_FeH = FeH.copy()
    masked_FeH[masked_FeH < -6] = np.nan
       
    feh_medians = np.nanpercentile(masked_FeH, [16,50,84])
    feh           = feh_medians[1]
    feh_err       = (feh_medians[2] - feh_medians[0])/2.
    feh_err_upper = feh_medians[2] - feh_medians[1]
    feh_err_lower = feh_medians[1] - feh_medians[0]

    # 0.1 dex systematic error due to scatter of the Carrera reln itself
    feh_err = np.sqrt(feh_err**2 + (0.1)**2)

    return feh,feh_err, feh_err_upper,feh_err_lower


# CALCULATE FeH FROM CaT EWs
def CaT_to_FeH(alldata):
    

    for ii,slt in enumerate(alldata): 

        alldata['ew_feh'][ii]      = -999.
        alldata['ew_feh_err'][ii]  = -999.
        if (slt['MV_o'] > -99) & (slt['ew_cat'] > 0.) & (slt['MV_o'] < 3):

            # MAGNITUDE ERRORS
            mag     = slt['MV_o']
            magerr  = slt['rmag_err']
            if magerr < 0: 
                magerr = 0.1

            # EW ERRORS    
            CaT     = slt['ew_cat']
            CaTerr  = slt['ew_cat_err']
            
            
            FeH, FeH_err, ferru,ferrl = calculate_FeH(mag, magerr, CaT, CaTerr)

            alldata['ew_feh'][ii]      = FeH
            alldata['ew_feh_err'][ii]  = FeH_err

    return alldata



###########################################
def match_gaia(obj,allspec):

    DEIMOS_RAW  = os.getenv('DEIMOS_RAW')
    gaia_file   = DEIMOS_RAW + '/Gaia_DR3/gaia_dr3_'+obj['Name2']+'.csv'

    
    if not os.path.isfile(gaia_file):
        print('NO GAIA FILE',gaia_file)

        
    if os.path.isfile(gaia_file):

        gaia = Table.read(gaia_file)
  

        cgaia   = SkyCoord(ra=gaia['ra']*u.degree, dec=gaia['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cgaia)  
        foo = np.arange(0,np.size(idx),1)

        mt  = foo[d2d < 1.25*u.arcsec]
        mt2 = idx[d2d < 1.25*u.arcsec]
        allspec['gaia_source_id'][mt]     = gaia['source_id'][mt2]
        allspec['gaia_pmra'][mt]          = gaia['pmra'][mt2] 
        allspec['gaia_pmra_err'][mt]      = gaia['pmra_error'][mt2]
        allspec['gaia_pmdec'][mt]         = gaia['pmdec'][mt2] 
        allspec['gaia_pmdec_err'][mt]     = gaia['pmdec_error'][mt2]
        allspec['gaia_pmra_pmdec_corr'][mt]  = gaia['pmra_pmdec_corr'][mt2]

        allspec['gaia_parallax'][mt]      = gaia['parallax'][mt2] 
        allspec['gaia_parallax_err'][mt]  = gaia['parallax_error'][mt2]
        allspec['gaia_phot_variable_flag'][mt] = gaia['phot_variable_flag'][mt2]
        allspec['gaia_rv'][mt]            = gaia['radial_velocity'][mt2] 
        allspec['gaia_rv_err'][mt]        = gaia['radial_velocity_error'][mt2] 
        allspec['gaia_grvs_mag'][mt]      = gaia['grvs_mag'][mt2] 

        allspec['gaia_aen'][mt]           = gaia['astrometric_excess_noise'][mt2] 
        allspec['gaia_aen_sig'][mt]       = gaia['astrometric_excess_noise_sig'][mt2]


        # SET NON_DETECTED BACK TO DEFAULT
        # GAIA DEFAULTS ARE ZERO (CONFUSING!)
        m = (allspec['gaia_pmra_err']  == -999.) | (allspec['gaia_pmra_err']  == 0.0)
        allspec['gaia_pmra'][m]        = -999.
        allspec['gaia_pmra_err'][m]    = -999.
        m = (allspec['gaia_pmdec_err'] == -999.) | (allspec['gaia_pmdec_err']  == 0.0)
        allspec['gaia_pmdec'][m]       = -999.
        allspec['gaia_pmdec_err'][m]   = -999.
        m = (allspec['gaia_parallax_err'] == -999.) | (allspec['gaia_parallax_err']  == 0.0)
        allspec['gaia_parallax'][m]    = -999.
        allspec['gaia_parallax_err'][m]     = -999.
        mrv = (allspec['gaia_rv']      == -999.) | (allspec['gaia_rv_err'] == 0)
        allspec['gaia_rv'][mrv]       = -999.
        allspec['gaia_rv_err'][mrv]   = -999.

        nrv = np.sum(allspec['gaia_rv'] > -999.)

        # SET GAIA FLAG 
        m = allspec['gaia_pmra'] > -999
        allspec['flag_gaia'][m] = 1

        print('GAIA: Matched {} stars and {} Gaia RVS'.format(np.size(mt),nrv))


    return allspec

###########################################
###########################################
# CALCULATE MAGNITUDE ERRORS FROM LEGACY FILES
def legacy_mag_err(flux, flux_ivar):

    # MINIMUM MAG ERROR
    mag_err    = np.zeros(np.size(flux))
    m          = (np.isfinite(flux_ivar)) & (flux_ivar >0)

    flux_err   = 1./np.sqrt(flux_ivar[m])
    mag_err[m] = (2.5/np.log(10.)) * (flux_err/flux[m])

    mag_err = np.sqrt(mag_err**2 + 0.02**2)

    return mag_err

def transform_sdss2decals(g_sdss, r_sdss):

    # TRANSFORMATION FROM Dey+ 2019, Appendix B
    gi = g_sdss - r_sdss +0.25

    g_decals = g_sdss + 0.0244 - 0.1183*gi + 0.0322*gi**2 - 0.0066*gi**3
    r_decals = r_sdss - 0.0005 - 0.0868*gi + 0.0287*gi**2 - 0.0092*gi**3

    return g_decals, r_decals


def transform_sdss2decam_des(g_sdss, r_sdss, i_sdss):

    # TRANSFORMATION FROM Abbott+ 2021 (DES DR2), Appendix B.1, Eq. B1,
    # SDSS -> DES, plus a flat +0.03 mag zero-point offset to bring DES
    # onto the Legacy Survey DR11 system (empirically calibrated against
    # DR11 using dwarf-field and sdss_gc crossmatches -- see
    # research_log_2026-08-25.html). Requires real measured g,r,i -- unlike
    # transform_sdss2decals, this is NOT valid with a (g-r)+0.25 proxy for
    # (g-i)/(r-i).
    gi = g_sdss - i_sdss
    ri = r_sdss - i_sdss

    g_decam = g_sdss - 0.061*gi + 0.008 - 0.03
    r_decam = r_sdss - 0.155*ri - 0.007 - 0.03
    i_decam = i_sdss - 0.166*ri + 0.032 - 0.03

    return g_decam, r_decam, i_decam


def borrow_sdss_i(ra, dec, Ai, name2):

    # Munoz g,r photometry has no i-band of its own. Where the same field
    # was also observed by SDSS, borrow real measured i for those stars so
    # transform_sdss2decam_des can be used instead of the (g-r)+0.25 proxy.
    # Ai is the per-star SDSS-native i-band extinction (already computed at
    # these positions via get_ebv(allspec, sdss=1)). Returns (i_dered,
    # i_err), both NaN where no sdss_dr14 file exists for this field, or no
    # good match/quality star is found within 1 arcsec.
    n = len(ra)
    i_dered = np.full(n, np.nan)
    i_err   = np.full(n, np.nan)

    DEIMOS_RAW = os.getenv('DEIMOS_RAW')
    path = None
    for ext in ['.fits.gz', '.fits']:
        candidate = DEIMOS_RAW + '/Photometry/sdss_dr14/sdss_dr14_' + name2 + ext
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        return i_dered, i_err

    try:
        sdss_i = Table.read(path)
    except Exception:
        return i_dered, i_err
    if len(sdss_i) == 0:
        return i_dered, i_err

    if 'PSFMAG_I' in sdss_i.colnames:
        clean = (np.array(sdss_i['clean']) == 1) & (np.array(sdss_i['PHOTPTYPE']) == 6)
        sdss_i = sdss_i[clean]
        if len(sdss_i) == 0:
            return i_dered, i_err
        imag_raw = np.array(sdss_i['PSFMAG_I'], dtype=float)
        imag_err = np.array(sdss_i['PSFMAGERR_I'], dtype=float)
        ra_col, dec_col = 'RA', 'DEC'
    elif 'i' in sdss_i.colnames:
        imag_raw = np.array(sdss_i['i'], dtype=float)
        imag_err = np.array(sdss_i['err_i'], dtype=float)
        ra_col, dec_col = 'ra', 'dec'
    else:
        return i_dered, i_err

    valid = np.isfinite(imag_raw) & np.isfinite(imag_err) & (imag_err > 0) & \
            (imag_err < 0.15) & (imag_raw > 10) & (imag_raw < 30)
    sdss_i = sdss_i[valid]
    imag_raw = imag_raw[valid]
    imag_err = imag_err[valid]
    if len(sdss_i) == 0:
        return i_dered, i_err

    c_target = SkyCoord(ra=np.asarray(ra)*u.degree, dec=np.asarray(dec)*u.degree)
    c_sdss   = SkyCoord(ra=np.array(sdss_i[ra_col])*u.degree, dec=np.array(sdss_i[dec_col])*u.degree)
    idx, sep, _ = c_target.match_to_catalog_sky(c_sdss)
    good = sep.arcsec < 1.0

    i_dered[good] = imag_raw[idx[good]] - Ai[good]
    i_err[good]   = imag_err[idx[good]]

    return i_dered, i_err


def borrow_r_for_pandas(ra, dec, name2):

    # PANDAS (Ibata+ 2014) has no r-band of its own. Try real DR11 r first
    # (already the target DECam system, no transform needed), then fall
    # back to SDSS DR14 r (transformed via transform_sdss2decam_des) for
    # stars DR11 doesn't cover. Returns (r_dered, r_err), NaN where neither
    # source has a good match. Coverage/consistency check: 2026-08-25,
    # research_log_2026-08-25.html -- DR11 alone covers ~7%, SDSS fallback
    # brings bright PANDAS stars to ~64% combined; where both exist they
    # agree to std~0.07 mag with no color trend.
    ra = np.asarray(ra); dec = np.asarray(dec)
    n = len(ra)
    r_dered = np.full(n, np.nan)
    r_err   = np.full(n, np.nan)
    c_target = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

    DEIMOS_RAW = os.getenv('DEIMOS_RAW')

    # --- try DR11 first ---
    dr11_path = DEIMOS_RAW + '/Photometry/legacy_DR11/ls_dr11_' + name2 + '.csv'
    if os.path.exists(dr11_path):
        try:
            d = ascii.read(dr11_path)
        except Exception:
            d = None
        if d is not None and len(d) > 0:
            good_d = (np.array(d['flux_ivar_r']) > 0) & np.isfinite(np.array(d['dered_mag_r']))
            d = d[good_d]
            if len(d) > 0:
                c_dr11 = SkyCoord(ra=np.array(d['ra'])*u.degree, dec=np.array(d['dec'])*u.degree)
                idx, sep, _ = c_target.match_to_catalog_sky(c_dr11)
                good = sep.arcsec < 0.5
                r_dered[good] = np.array(d['dered_mag_r'])[idx[good]]
                r_err[good] = legacy_mag_err(np.array(d['flux_r'])[idx[good]], np.array(d['flux_ivar_r'])[idx[good]])

    # --- SDSS DR14 fallback, only for stars DR11 didn't cover ---
    need = ~np.isfinite(r_dered)
    if np.any(need):
        sdss_path = None
        for ext in ['.fits.gz', '.fits']:
            candidate = DEIMOS_RAW + '/Photometry/sdss_dr14/sdss_dr14_' + name2 + ext
            if os.path.exists(candidate):
                sdss_path = candidate
                break
        if sdss_path is not None:
            try:
                s = Table.read(sdss_path)
            except Exception:
                s = None
            if s is not None and len(s) > 0:
                if 'PSFMAG_G' in s.colnames:
                    clean = (np.array(s['clean']) == 1) & (np.array(s['PHOTPTYPE']) == 6)
                    s = s[clean]
                    gcol,rcol,icol = 'PSFMAG_G','PSFMAG_R','PSFMAG_I'
                    gecol,recol,iecol = 'PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I'
                    ra_col,dec_col = 'RA','DEC'
                elif 'g' in s.colnames:
                    gcol,rcol,icol = 'g','r','i'
                    gecol,recol,iecol = 'err_g','err_r','err_i'
                    ra_col,dec_col = 'ra','dec'
                else:
                    s = None
                if s is not None and len(s) > 0:
                    g_raw = np.array(s[gcol], dtype=float); r_raw = np.array(s[rcol], dtype=float); i_raw = np.array(s[icol], dtype=float)
                    ge = np.array(s[gecol], dtype=float); re = np.array(s[recol], dtype=float); ie = np.array(s[iecol], dtype=float)
                    valid = np.isfinite(g_raw)&np.isfinite(r_raw)&np.isfinite(i_raw)&(ge<0.2)&(re<0.2)&(ie<0.2)
                    s = s[valid]; g_raw=g_raw[valid]; r_raw=r_raw[valid]; i_raw=i_raw[valid]; re=re[valid]
                    if len(s) > 0:
                        c_sdss = SkyCoord(ra=np.array(s[ra_col])*u.degree, dec=np.array(s[dec_col])*u.degree)
                        idx2, sep2, _ = c_target.match_to_catalog_sky(c_sdss)
                        good2 = need & (sep2.arcsec < 1.0)
                        if np.any(good2):
                            sdf_r = sfdmap.SFDMap(DEIMOS_RAW + '/SFDmaps/', scaling=1.0)
                            ebv2 = sdf_r.ebv(ra[good2], dec[good2])
                            g_s = g_raw[idx2[good2]] - 3.303*ebv2
                            r_s = r_raw[idx2[good2]] - 2.285*ebv2
                            i_s = i_raw[idx2[good2]] - 1.698*ebv2
                            _, r_decam, _ = transform_sdss2decam_des(g_s, r_s, i_s)
                            r_dered[good2] = r_decam
                            r_err[good2] = re[idx2[good2]]

    return r_dered, r_err


def transform_ps12decals(g_ps1, r_ps1, i_ps1):

    # TRANSFORMATION FROM Dey+ 2019, Eq 1+2. Defined for real measured
    # (g-i)_PS1, not a (g-r)+0.2 proxy (fixed 2026-08-25, M. Geha -- same
    # issue as transform_sdss2decals, see research_log_2026-08-25.html).
    gi = g_ps1 - i_ps1

    g_decals = g_ps1 + 0.00062 + 0.03604*gi + 0.01028*gi**2 - 0.00613*gi**3
    r_decals = r_ps1 + 0.00495 - 0.08435*gi + 0.03222*gi**2 - 0.01140*gi**3

    # i-band: no published PS1->DECaLS relation exists (Dey+2019 Eq 1+2
    # only cover g,r). Our own cubic fit, same form/predictor as g,r above,
    # from 14 GC/M31 fields overlapping DR11, 2026-08-25 (M. Geha) -- see
    # research_log_2026-08-25.html.
    i_decals = i_ps1 - (0.04783 + 0.04108*gi - 0.05507*gi**2 + 0.03224*gi**3)

    return g_decals, r_decals, i_decals


###########################################
###########################################
def match_photometry(obj,allspec):
    
    DEIMOS_RAW      = os.getenv('DEIMOS_RAW')

    # POPULATE EBV
    nall            = np.size(allspec)
    nobj            = np.sum(allspec['v_err'] >= 0)
    nstar           = np.sum(allspec['v_err'] > 0)

    # DEFINE MATCHING LENGTH
    dm = 1.25
    dm_serendip = 2.
 
  #####################
    ### PRIMARY SOURCE:   LEGACY DR11 (replaces DR10 -- DR11 has native
    #   griz for our targets, confirmed empirically: 5 spot-checked
    #   objects spanning dec -23 to +67 all had >=98% finite i-band rows,
    #   despite the Legacy Survey docs describing i-band as DECam(south)-
    #   only in general. https://www.legacysurvey.org/decamls/)
    #   Using dereddened AB magnitudes in DECAM system
    if obj['Phot'] == 'ls_dr11':
        file = DEIMOS_RAW + '/Photometry/legacy_DR11/ls_dr11_'+obj['Name2']+'.csv'
        ls_dr11 = ascii.read(file)

        ls_dr11.rename_column('dered_mag_g', 'gmag')
        ls_dr11.rename_column('dered_mag_r', 'rmag')
        ls_dr11.rename_column('dered_mag_i', 'imag')

        # REPLACE INF VALUES
        m = np.isfinite(ls_dr11['gmag'])
        ls_dr11['gmag'][~m] = -999.0
        m = np.isfinite(ls_dr11['rmag'])
        ls_dr11['rmag'][~m] = -999.0
        m = np.isfinite(ls_dr11['imag'])
        ls_dr11['imag'][~m] = -999.0


        # NO DEIMOS SOURCES THIS FAINT, CUT TO REDUCE MIS-MATCHING
        ls_dr11 = ls_dr11[ls_dr11['rmag'] < 25]
        ls_dr11 = ls_dr11[ls_dr11['imag'] < 25]


        # CORRECT BASS MAGNITUDES FOR NORTHERN FIELDS -- CARRIED OVER
        # UNCHANGED FROM THE DR10 BRANCH. LIKELY A BASS-vs-DECam r-FILTER
        # BANDPASS/COLOR-TERM EFFECT RATHER THAN A CALIBRATION ARTIFACT
        # (BASS DR3 IS ITSELF CALIBRATED TO PS1 TO <5 mmag, Zou et al.
        # 2019, ApJS 245, 4 -- SO A RESIDUAL BASS/DECam MISMATCH REFLECTS
        # THE UNDERLYING INSTRUMENTS, NOT THE PIPELINE), AND CONFIRMED
        # (2026-08-24, M. Geha) TO BE UNCHANGED IN DR11.
        if np.median(ls_dr11['dec'] > 34.):
            m = (ls_dr11['gmag'] > 0) & (ls_dr11['rmag'] > 0)
            ls_dr11['rmag'][m] =  -0.0382 * (ls_dr11['gmag'][m] - ls_dr11['rmag'][m]) + 0.0108 + ls_dr11['rmag'][m]


        cls_dr11 = SkyCoord(ra=ls_dr11['ra']*u.degree, dec=ls_dr11['dec']*u.degree)
        cdeimos  = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr11)
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < dm*u.arcsec]
        print(np.size(mt))

        allspec['gmag_o'][mt] = ls_dr11['gmag'][idx[d2d < dm*u.arcsec]]
        allspec['rmag_o'][mt] = ls_dr11['rmag'][idx[d2d < dm*u.arcsec]]
        allspec['imag_o'][mt] = ls_dr11['imag'][idx[d2d < dm*u.arcsec]]

        allspec['gmag_err'][mt] = legacy_mag_err(ls_dr11['flux_g'][idx[d2d < dm*u.arcsec]] , ls_dr11['flux_ivar_g'][idx[d2d < dm*u.arcsec]] )
        allspec['rmag_err'][mt] = legacy_mag_err(ls_dr11['flux_r'][idx[d2d < dm*u.arcsec]] , ls_dr11['flux_ivar_r'][idx[d2d < dm*u.arcsec]] )
        allspec['imag_err'][mt] = legacy_mag_err(ls_dr11['flux_i'][idx[d2d < dm*u.arcsec]] , ls_dr11['flux_ivar_i'][idx[d2d < dm*u.arcsec]] )

        allspec['phot_source'][mt] = 'ls_dr11'
        allspec['phot_type'][mt] = ls_dr11['type'][idx[d2d < dm*u.arcsec]]



        # INCREASE FOR SERENDIPS W/O Match
        sm  = (d2d < dm_serendip*u.arcsec) & (allspec['serendip'] == 1) & (allspec['rmag_o'] < 0)


        if np.sum(sm) > 0:
            mts = foo[sm]
            #print(allspec['rmag_o'][mts])

            allspec['gmag_o'][mts] = ls_dr11['gmag'][idx[sm]]
            allspec['rmag_o'][mts] = ls_dr11['rmag'][idx[sm]]
            allspec['imag_o'][mts] = ls_dr11['imag'][idx[sm]]

            allspec['gmag_err'][mts] = legacy_mag_err(ls_dr11['flux_g'][idx[sm]] , ls_dr11['flux_ivar_g'][idx[sm]] )
            allspec['rmag_err'][mts] = legacy_mag_err(ls_dr11['flux_r'][idx[sm]] , ls_dr11['flux_ivar_r'][idx[sm]] )
            allspec['imag_err'][mts] = legacy_mag_err(ls_dr11['flux_i'][idx[sm]] , ls_dr11['flux_ivar_i'][idx[sm]] )

            allspec['phot_source'][mts] = 'ls_dr11'
            allspec['phot_type'][mts] = ls_dr11['type'][idx[sm]]

            #for ob in allspec[mts]:
            #    print(ob['rmag_o'],ob['RA'],ob['DEC'],ob['SN'],ob['marz_flag'])


    #####################
    ### MUNOZ -- COMPLETE UNPUBLISHED CATALOGS
    if obj['Phot'] == 'munozf':
        
        file = DEIMOS_RAW + '/Photometry/munoz_full/final_'+obj['Name2']+'.phot'

        munozf = ascii.read(file)
        munozf.rename_column('col2', 'RA')
        munozf.rename_column('col3', 'DEC')
        munozf.rename_column('col4', 'g')
        munozf.rename_column('col5', 'gerr')
        munozf.rename_column('col6', 'r')
        munozf.rename_column('col7', 'rerr')

        
        if (obj['Name2'] == 'Eri') | (obj['Name2'] == 'K2')| (obj['Name2'] == 'Leo2')| \
           (obj['Name2'] == 'Seg2') | (obj['Name2'] == 'N2419')| (obj['Name2'] == 'Pal2'):
            munozf['RA'] *= 15.
        
        munozf  = munozf[munozf['r'] < 25.0]
        cmunf   = SkyCoord(ra=munozf['RA']*u.degree, dec=munozf['DEC']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
 
         # ADDING SYSTEMATIC MAG ERROR
        munozf['rerr'] = np.sqrt(munozf['rerr']**2 + 0.02**2)
        munozf['gerr'] = np.sqrt(munozf['gerr']**2 + 0.02**2)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)
        foo = np.arange(0,np.size(idx),1)

        # Get SDSS-native Ar/Ag, deredden BEFORE transform to DECaLS
        allspec, Ar, Ag, Ai = get_ebv(allspec, sdss=1)
        mt = foo[d2d < dm*u.arcsec]

        # TRANSFORM SDSS TO DECam (input already dereddened in native system).
        # Borrow real SDSS DR14 i where this field/star has it, and use the
        # full g,r,i DES transform there; fall back to the (g-r)+0.25 proxy
        # (Dey+2019) transform where no independent i is available.
        r_sdss = munozf['r'][idx[d2d < dm*u.arcsec]] - Ar[mt]
        g_sdss = munozf['g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        i_sdss, i_sdss_err = borrow_sdss_i(allspec['RA'][mt], allspec['DEC'][mt], Ai[mt], obj['Name2'])
        has_i = np.isfinite(i_sdss)

        g_decals = np.zeros(len(mt))
        r_decals = np.zeros(len(mt))
        if np.any(has_i):
            g_decals[has_i], r_decals[has_i], i_decam = transform_sdss2decam_des(g_sdss[has_i], r_sdss[has_i], i_sdss[has_i])
            allspec['imag_o'][mt[has_i]]   = i_decam
            allspec['imag_err'][mt[has_i]] = i_sdss_err[has_i]
        if np.any(~has_i):
            g_decals[~has_i], r_decals[~has_i] = transform_sdss2decals(g_sdss[~has_i], r_sdss[~has_i])

        allspec['rmag_o'][mt]   = r_decals
        allspec['gmag_o'][mt]   = g_decals
        allspec['rmag_err'][mt] = munozf['rerr'][idx[d2d < dm*u.arcsec]]
        allspec['gmag_err'][mt] = munozf['gerr'][idx[d2d < dm*u.arcsec]]

        allspec['phot_source'][mt] = 'munozf'


       # INCREASE FOR SERENDIPS W/O Match
        sm  = (d2d < dm_serendip*u.arcsec) & (allspec['serendip'] == 1) & (allspec['rmag_o'] < 0)

        if np.sum(sm) > 0:
            mts = foo[sm]

            r_sdss_s = munozf['r'][idx[sm]] - Ar[mts]
            g_sdss_s = munozf['g'][idx[sm]] - Ag[mts]
            i_sdss_s, i_sdss_err_s = borrow_sdss_i(allspec['RA'][mts], allspec['DEC'][mts], Ai[mts], obj['Name2'])
            has_i_s = np.isfinite(i_sdss_s)

            g_decals_s = np.zeros(len(mts))
            r_decals_s = np.zeros(len(mts))
            if np.any(has_i_s):
                g_decals_s[has_i_s], r_decals_s[has_i_s], i_decam_s = transform_sdss2decam_des(g_sdss_s[has_i_s], r_sdss_s[has_i_s], i_sdss_s[has_i_s])
                allspec['imag_o'][mts[has_i_s]]   = i_decam_s
                allspec['imag_err'][mts[has_i_s]] = i_sdss_err_s[has_i_s]
            if np.any(~has_i_s):
                g_decals_s[~has_i_s], r_decals_s[~has_i_s] = transform_sdss2decals(g_sdss_s[~has_i_s], r_sdss_s[~has_i_s])

            allspec['rmag_o'][mts]   = r_decals_s
            allspec['gmag_o'][mts]   = g_decals_s
            allspec['rmag_err'][mts] = munozf['rerr'][idx[sm]]
            allspec['gmag_err'][mts] = munozf['gerr'][idx[sm]]
            allspec['phot_source'][mts] = 'munozf'


     
    if obj['Phot'] == 'munoz18_2':
        
        file = DEIMOS_RAW + '/Photometry/munoz18/munoz18_secondary.txt'
        # Get SDSS-native Ar/Ag, deredden BEFORE transform to DECaLS
        allspec, Ar, Ag, Ai = get_ebv(allspec, sdss=1)

        munozf = ascii.read(file)

        cmunf   = SkyCoord(ra=munozf['ra']*u.degree, dec=munozf['dec']*u.degree)
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cmunf)
        foo = np.arange(0,np.size(idx),1)
        mt  = foo[d2d < dm*u.arcsec]

        r_sdss = munozf['r'][idx[d2d < dm*u.arcsec]] - Ar[mt]
        g_sdss = munozf['g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        i_sdss, i_sdss_err = borrow_sdss_i(allspec['RA'][mt], allspec['DEC'][mt], Ai[mt], obj['Name2'])
        has_i = np.isfinite(i_sdss)

        g_decals = np.zeros(len(mt))
        r_decals = np.zeros(len(mt))
        if np.any(has_i):
            g_decals[has_i], r_decals[has_i], i_decam = transform_sdss2decam_des(g_sdss[has_i], r_sdss[has_i], i_sdss[has_i])
            allspec['imag_o'][mt[has_i]]   = i_decam
            allspec['imag_err'][mt[has_i]] = i_sdss_err[has_i]
        if np.any(~has_i):
            g_decals[~has_i], r_decals[~has_i] = transform_sdss2decals(g_sdss[~has_i], r_sdss[~has_i])

        allspec['rmag_o'][mt]   = r_decals
        allspec['gmag_o'][mt]   = g_decals

        # ADDING SYSTEMATIC MAG ERROR
        allspec['rmag_err'][mt] = np.sqrt(munozf['rerr'][idx[d2d < dm*u.arcsec]]**2 + 0.02**2)
        allspec['gmag_err'][mt] = np.sqrt(munozf['gerr'][idx[d2d < dm*u.arcsec]]**2 + 0.02**2)

        allspec['phot_source'][mt] = 'munoz18_2'


    #####################
    ### GC SDSS PHOTOMETRY
    # http://classic.sdss.org/dr7/products/value_added/anjohnson08_clusterphotometry.html
    if obj['Phot'] == 'sdss_gc':

        file = DEIMOS_RAW + '/Photometry/sdss_gc/sdss_gc_'+obj['Name2']+'.phot'
        sdss = ascii.read(file)
        
        sdss.rename_column('col5', 'RA')
        sdss.rename_column('col6', 'DEC')
        sdss.rename_column('col16', 'gmag')
        sdss.rename_column('col17', 'gmag_err')
        sdss.rename_column('col23', 'rmag')
        sdss.rename_column('col24', 'rmag_err')
        sdss.rename_column('col30', 'imag')
        sdss.rename_column('col31', 'imag_err')




        csdss   = SkyCoord(ra=sdss['RA']*u.degree, dec=sdss['DEC']*u.degree)
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(csdss)
        foo = np.arange(0,np.size(idx),1)
        mt = foo[d2d < dm*u.arcsec]

        # Get SDSS-native Ar/Ag/Ai, deredden BEFORE transform to DECam
        allspec, Ar, Ag, Ai = get_ebv(allspec, sdss=1)

        r_sdss = sdss['rmag'][idx[d2d < dm*u.arcsec]] - Ar[mt]
        g_sdss = sdss['gmag'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        i_sdss = sdss['imag'][idx[d2d < dm*u.arcsec]] - Ai[mt]

        g_decam, r_decam, i_decam = transform_sdss2decam_des(g_sdss, r_sdss, i_sdss)
        allspec['rmag_o'][mt] =  r_decam
        allspec['gmag_o'][mt] =  g_decam
        allspec['imag_o'][mt] =  i_decam
        allspec['rmag_err'][mt] = sdss['rmag_err'][idx[d2d < dm*u.arcsec]]
        allspec['gmag_err'][mt] = sdss['gmag_err'][idx[d2d < dm*u.arcsec]]
        allspec['imag_err'][mt] = sdss['imag_err'][idx[d2d < dm*u.arcsec]]

        # FIX NULLS
        mr = (allspec['rmag_err'] > 9.0) | (allspec['rmag_o'] > 100)
        mg = (allspec['gmag_err'] > 9.0) | (allspec['gmag_o'] > 100)
        mi = (allspec['imag_err'] > 9.0) | (allspec['imag_o'] > 100)
        allspec['rmag_err'][mr] = -999.0
        allspec['gmag_err'][mg] = -999.0
        allspec['imag_err'][mi] = -999.0

        allspec['phot_source'][mt] = 'sdss_gc'


    #####################
    ### DECAPS
    if obj['Phot'] == 'decaps':
        file = DEIMOS_RAW + '/Photometry/decaps/decaps_'+obj['Name2']+'.csv'
        decaps = Table.read(file)
        
        
        cls_dr10= SkyCoord(ra=decaps['ra_ok']*u.degree, dec=decaps['dec_ok']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        # REPLACE INF VALUES
        m = np.isfinite(decaps['mean_mag_g'])
        decaps['mean_mag_g'][~m] = -999
        m = np.isfinite(decaps['mean_mag_r'])
        decaps['mean_mag_r'][~m] = -999

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cls_dr10)  
        foo = np.arange(0,np.size(idx),1)

        allspec, Ar, Ag, Ai = get_ebv(allspec)

        mt = foo[d2d < dm*u.arcsec]
        allspec['rmag_o'][mt] = decaps['mean_mag_r'][idx[d2d < dm*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt] = decaps['mean_mag_g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = 0.1
        allspec['gmag_err'][mt] = 0.1

        allspec['phot_source'][mt] = 'decaps'


  #####################
    ## PANSTARRS DR2
    #  https://catalogs.mast.stsci.edu/
    if obj['Phot'] == 'PanS':
        file = DEIMOS_RAW + '/Photometry/PanS/PanS_'+obj['Name2']+'.csv'
        pans = ascii.read(file)
        m=(pans['rMeanPSFMag'] != -999) & (pans['gMeanPSFMag'] != -999) & (pans['iMeanPSFMag'] != -999)
        pans=pans[m]

        cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree)
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)
        foo = np.arange(0,np.size(idx),1)

        # GET reddening -- PS1-native, deredden BEFORE transform to DECaLS
        allspec, Ar, Ag, Ai = get_ebv(allspec, ps1=1)

        # INCREASED TO 2" TO GET CENTRAL GLOBULAR CLUSTER MEMBERS
        ds = 2.0
        mt = foo[d2d < ds*u.arcsec]

        # TRANSFORM TO DECALS (real measured (g-i)_PS1, dereddened first)
        g_ps1 = pans['gMeanPSFMag'][idx[d2d < ds*u.arcsec]] - Ag[mt]
        r_ps1 = pans['rMeanPSFMag'][idx[d2d < ds*u.arcsec]] - Ar[mt]
        i_ps1 = pans['iMeanPSFMag'][idx[d2d < ds*u.arcsec]] - Ai[mt]
        g_decals, r_decals, i_decals = transform_ps12decals(g_ps1, r_ps1, i_ps1)

        allspec['rmag_o'][mt] = r_decals
        allspec['gmag_o'][mt] = g_decals
        allspec['imag_o'][mt] = i_decals

        allspec['rmag_err'][mt] = pans['rMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['gmag_err'][mt] = pans['gMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['imag_err'][mt] = pans['iMeanPSFMagErr'][idx[d2d < ds*u.arcsec]]

        allspec['phot_source'][mt] = 'PanS'


 #####################
    ## PANSTARRS DR2
    #  https://catalogs.mast.stsci.edu/
    if obj['Phot'] == 'PanS1':
        file = DEIMOS_RAW + '/Photometry/PanS/PanS1_'+obj['Name2']+'.csv'
        pans = ascii.read(file)
        m=(pans['rPSFMag'] != -999) & (pans['gPSFMag'] != -999) & (pans['iPSFMag'] != -999) & \
          (pans['rPSFMagErr'] < 0.5)& (pans['gPSFMagErr'] < 0.5)
        pans=pans[m]

        cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree)
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)
        foo = np.arange(0,np.size(idx),1)

        # GET reddening -- PS1-native, deredden BEFORE transform to DECaLS
        allspec, Ar, Ag, Ai = get_ebv(allspec, ps1=1)

        # INCREASED TO 2" TO GET CENTRAL GLOBULAR CLUSTER MEMBERS
        ds = 2.25
        mt = foo[d2d < ds*u.arcsec]

        # TRANSFORM TO DECALS (real measured (g-i)_PS1, dereddened first)
        g_ps1 = pans['gPSFMag'][idx[d2d < ds*u.arcsec]] - Ag[mt]
        r_ps1 = pans['rPSFMag'][idx[d2d < ds*u.arcsec]] - Ar[mt]
        i_ps1 = pans['iPSFMag'][idx[d2d < ds*u.arcsec]] - Ai[mt]
        g_decals, r_decals, i_decals = transform_ps12decals(g_ps1, r_ps1, i_ps1)

        allspec['rmag_o'][mt] = r_decals
        allspec['gmag_o'][mt] = g_decals
        allspec['imag_o'][mt] = i_decals

        allspec['rmag_err'][mt] = pans['rPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['gmag_err'][mt] = pans['gPSFMagErr'][idx[d2d < ds*u.arcsec]]
        allspec['imag_err'][mt] = pans['iPSFMagErr'][idx[d2d < ds*u.arcsec]]

        allspec['phot_source'][mt] = 'PanS'

    
    #####################
    ### USE GAIA IF THERE ARE NO OTHER OPTIONS
    if obj['Phot'] == 'gaia':
        file = DEIMOS_RAW + '/Gaia_DR3/gaia_dr3_'+obj['Name2']+'.csv'
        gaia = ascii.read(file)
        
        # TRANSFORM USING Table 5.7
        #https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html#Ch5.T8 

        G_BP_RP = gaia['bp_rp']
        G       = gaia['phot_g_mean_mag']

        Gr   = -0.09837 + 0.08592*G_BP_RP + 0.1907*G_BP_RP**2 - 0.1701*G_BP_RP**3 + 0.02263*G_BP_RP**4
        Gg   =  0.2199  - 0.6365*G_BP_RP  - 0.1548*G_BP_RP**2 + 0.0064*G_BP_RP**3
        rmag =  G - Gr
        gmag =  G - Gg
        err  = (2.5/np.log(10)) / gaia['phot_g_mean_flux_over_error']
        gaia_err  = np.sqrt(err**2 + 0.07**2)

        # TRANSFORMATION IS TOTALLY OFF
        rmag = gaia['phot_rp_mean_mag']
        gmag = gaia['phot_g_mean_mag']-0.3


        cgaia   = SkyCoord(ra=gaia['ra']*u.degree, dec=gaia['dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 

        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cgaia)  
        foo = np.arange(0,np.size(idx),1)

        allspec, Ar, Ag, Ai = get_ebv(allspec)

        mt = foo[d2d < dm*u.arcsec]
        allspec['rmag_o'][mt]   = rmag[idx[d2d < dm*u.arcsec]] - Ar[mt]
        allspec['gmag_o'][mt]   = gmag[idx[d2d < dm*u.arcsec]] - Ag[mt]
        allspec['rmag_err'][mt] = gaia_err[idx[d2d < dm*u.arcsec]] 
        allspec['gmag_err'][mt] = gaia_err[idx[d2d < dm*u.arcsec]] 

        allspec['phot_source'][mt] = 'gaia'

        

    if obj['Phot'] == 'sdss_dr14':
        
            file  =  DEIMOS_RAW + '/Photometry/sdss_dr14/sdss_dr14_'+obj['Name2']+'.fits'
            sdss  = Table.read(file)
            m=(sdss['err_g'] < 0.4) & (sdss['err_r'] < 0.4)
            sdss=sdss[m]
            allspec, Ar, Ag, Ai = get_ebv(allspec)


            cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
            csdss = SkyCoord(ra=sdss['ra']*u.degree, dec=sdss['dec']*u.degree) 
            idx, d2d, d3d = cdeimos.match_to_catalog_sky(csdss)  
            foo = np.arange(0,np.size(idx),1)

        
            mt = foo[d2d < dm*u.arcsec]
            allspec['imag_o'][mt] = sdss['i'][idx[d2d < dm*u.arcsec]] - Ai[mt] 
            allspec['rmag_o'][mt] = sdss['r'][idx[d2d < dm*u.arcsec]] - Ar[mt] 
            allspec['gmag_o'][mt] = sdss['g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
            allspec['imag_err'][mt] = sdss['err_i'][idx[d2d < dm*u.arcsec]] 
            allspec['rmag_err'][mt] = sdss['err_r'][idx[d2d < dm*u.arcsec]] 
            allspec['gmag_err'][mt] = sdss['err_g'][idx[d2d < dm*u.arcsec]] 
            allspec['phot_source'][mt] = 'sdss'
    

    if obj['Phot'] == 'pandas':

        # GET reddening 
        allspec, Ar, Ag, Ai = get_ebv(allspec)


        # FIRST SUPPLEMENT WITH OTHER PHOTOMETRY IF AVALABLE
        if obj['Phot2'] == 'sdss_dr14':
            file  =  DEIMOS_RAW + '/Photometry/sdss_dr14/sdss_dr14_'+obj['Name2']+'.fits'
            sdss  = Table.read(file)
            m=(sdss['err_g'] < 0.4) & (sdss['err_r'] < 0.4)
            sdss=sdss[m]


            cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
            csdss = SkyCoord(ra=sdss['ra']*u.degree, dec=sdss['dec']*u.degree) 
            idx, d2d, d3d = cdeimos.match_to_catalog_sky(csdss)  
            foo = np.arange(0,np.size(idx),1)

            mt = foo[d2d < dm*u.arcsec]
            allspec['imag_o'][mt] = sdss['i'][idx[d2d < dm*u.arcsec]] - Ai[mt] 
            allspec['rmag_o'][mt] = sdss['r'][idx[d2d < dm*u.arcsec]] - Ar[mt] 
            allspec['gmag_o'][mt] = sdss['g'][idx[d2d < dm*u.arcsec]] - Ag[mt]
            allspec['imag_err'][mt] = sdss['err_i'][idx[d2d < dm*u.arcsec]] 
            allspec['rmag_err'][mt] = sdss['err_r'][idx[d2d < dm*u.arcsec]] 
            allspec['gmag_err'][mt] = sdss['err_g'][idx[d2d < dm*u.arcsec]] 
            allspec['phot_source'][mt] = 'sdss'

        if obj['Phot2'] == 'PanS':
            file = DEIMOS_RAW + '/Photometry/PanS/PanS_'+obj['Name2']+'.csv'
            pans = ascii.read(file)
            m=(pans['rPSFMag'] != -999) & (pans['gPSFMag'] != -999) & (pans['iPSFMag'] != -999) & \
              (pans['rPSFMagErr'] < 0.5)& (pans['gPSFMagErr'] < 0.5)
            pans=pans[m]

            cpans   = SkyCoord(ra=pans['raMean']*u.degree, dec=pans['decMean']*u.degree)
            cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree)

            idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpans)
            foo = np.arange(0,np.size(idx),1)
            mt = foo[d2d < dm*u.arcsec]

            # PS1-native, deredden BEFORE transform to DECaLS (2026-08-25
            # fix, same issue as the primary PanS/PanS1 branches above --
            # this fallback previously applied DECam-scale extinction
            # directly to raw PS1 mags with no color transform at all).
            allspec, Ar_ps1, Ag_ps1, Ai_ps1 = get_ebv(allspec, ps1=1)
            g_ps1 = pans['gPSFMag'][idx[d2d < dm*u.arcsec]] - Ag_ps1[mt]
            r_ps1 = pans['rPSFMag'][idx[d2d < dm*u.arcsec]] - Ar_ps1[mt]
            i_ps1 = pans['iPSFMag'][idx[d2d < dm*u.arcsec]] - Ai_ps1[mt]
            g_decals, r_decals, i_decals = transform_ps12decals(g_ps1, r_ps1, i_ps1)

            allspec['imag_o'][mt] = i_decals
            allspec['rmag_o'][mt] = r_decals
            allspec['gmag_o'][mt] = g_decals
            allspec['imag_err'][mt] = pans['iPSFMagErr'][idx[d2d < dm*u.arcsec]]
            allspec['rmag_err'][mt] = pans['rPSFMagErr'][idx[d2d < dm*u.arcsec]]
            allspec['gmag_err'][mt] = pans['gPSFMagErr'][idx[d2d < dm*u.arcsec]]
            allspec['phot_source'][mt] = 'PanS'

        allspec, Ar, Ag, Ai = get_ebv(allspec,pandas=1)


        # OVERRIDE WITH PANDAS:   g and i-band photometry
        file = DEIMOS_RAW + '/Photometry/PANDAS/PANDAS_'+obj['Name2']+'.csv'
        pandas = ascii.read(file)
        pandas.rename_column('g', 'gmag')
        pandas.rename_column('i', 'imag')        
        pandas.rename_column('dg', 'gmag_err')
        pandas.rename_column('di', 'imag_err')



        cpandas = SkyCoord(ra=pandas['RA']*u.degree, dec=pandas['Dec']*u.degree) 
        cdeimos = SkyCoord(ra=allspec['RA']*u.degree, dec=allspec['DEC']*u.degree) 
        idx, d2d, d3d = cdeimos.match_to_catalog_sky(cpandas)  
        foo = np.arange(0,np.size(idx),1)

        mt = foo[d2d < dm*u.arcsec]

        # Deredden natively, then apply our own g,i fits (2026-08-25,
        # research_log_2026-08-25.html): g needs only a flat zero-point
        # offset (a fitted color term made things worse); i has a real
        # cubic color term, fit against SDSS-transformed-to-DR11 i since
        # DR11 has no real i coverage in the PANDAS (M31/M33) footprint.
        g_pandas_dered = pandas['gmag'][idx[d2d < dm*u.arcsec]] - Ag[mt]
        i_pandas_dered = pandas['imag'][idx[d2d < dm*u.arcsec]] - Ai[mt]
        gi_pandas = g_pandas_dered - i_pandas_dered

        allspec['gmag_o'][mt] = g_pandas_dered + 0.023
        allspec['imag_o'][mt] = i_pandas_dered - (0.00797 - 0.02110*gi_pandas - 0.00936*gi_pandas**2 - 0.00589*gi_pandas**3)
        allspec['imag_err'][mt] = np.sqrt(pandas['imag_err'][idx[d2d < dm*u.arcsec]]**2 + 0.05**2)
        allspec['gmag_err'][mt] = np.sqrt(pandas['gmag_err'][idx[d2d < dm*u.arcsec]]**2 + 0.05**2)
        allspec['phot_source'][mt] = 'pandas'

        # PANDAS has no r-band; borrow real r from DR11 (preferred) or
        # SDSS DR14 (fallback) where available -- see borrow_r_for_pandas.
        # Leaves rmag_o at its default/missing value otherwise, rather
        # than the previous crude r = i + 0.3 placeholder.
        r_borrowed, r_borrowed_err = borrow_r_for_pandas(allspec['RA'][mt], allspec['DEC'][mt], obj['Name2'])
        has_r = np.isfinite(r_borrowed)
        allspec['rmag_o'][mt[has_r]] = r_borrowed[has_r]
        allspec['rmag_err'][mt[has_r]] = r_borrowed_err[has_r]


    # REMOVE SERENDIP STARS WITHOUT PHOTOMETRY IN ANY BAND (2026-08-25,
    # M. Geha -- was r-band only, which started dropping serendips that
    # have real g,i but no r now that PANDAS r-band is genuinely missing
    # for a fraction of stars rather than always filled with a placeholder)
    m_serendip_nophot =  (allspec['serendip'] == 1) &  (allspec['rmag_o'] < 0) & \
                          (allspec['gmag_o'] < 0) & (allspec['imag_o'] < 0)
    allspec           = allspec[~m_serendip_nophot]



    # DETERMINE MV AND CONVERT CAT -> FEH
    m_miss_star = (allspec['rmag_o'] < 0)  & (allspec['v_err'] > 0)
    print('PHOT: Matched {} stars, missing {} star targets'.format(np.size(mt),np.sum(m_miss_star)))

    allspec = calc_rproj(allspec,obj)
    allspec = calc_MV_star(allspec,obj)
    allspec = CaT_to_FeH(allspec)


    return allspec

