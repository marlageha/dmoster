import numpy as np
import os
import glob

from astropy import table
from astropy.table import Table,Column
from astropy import units as u
from astropy.io import ascii,fits

from astropy.coordinates import SkyCoord

from dmost import dmost_utils
from dmost.combine import dmost_photometry_gaia, dmost_membership
from scipy import stats

# CaT SYSTEMATIC ERROR, PER-OBSERVATION: sqrt(CAT_SYS_MULT^2*cat_err^2 +
# CAT_SYS_FLOOR^2). Calibrated from cross-mask repeat-star pairs (dynesty
# joint-mixture fit, density-weighted MLE, and a binned cross-check all
# agree within noise; CaT_GL_syserr_Feh/research_log_2026-08-27.html,
# rechecked on the full Stage 1/Stage 2 database 2026-08-28).
CAT_SYS_MULT  = 1.7
CAT_SYS_FLOOR = 0.07



###################################
# CREATE ALLSPEC DATA STRUCTURE
def create_allstars(nmasks,nstars):

    cols = [filled_column('system_name','                  ',nstars),
            filled_column('objname','                  ',nstars),
            filled_column('RA',-999.,nstars),
            filled_column('DEC',-999.,nstars),


            # INDIVIDUAL MASK PROPERTIES
            filled_column('nmask',-999,nstars),
            filled_column('nexp',-999,nstars),
            filled_column('t_exp',-999.,nstars),
            filled_column('masknames','                                                              ',nstars),
            filled_column('slitwidth',-999.,nstars),
            filled_column('mean_mjd',-999.,nstars),
            filled_column('collate1d_filename','                                                  ',nstars),

         
            # SN and FLAGs
            filled_column('SN',-999.,nstars),
            filled_column('serendip',-999,nstars),

            # GALAXIES
            filled_column('marz_flag',-999,nstars),
            filled_column('marz_z',-999.,nstars),


            # COMBINED FINAL PROPERTIES
            filled_column('v',-999.,nstars),
            filled_column('v_err',-999.,nstars),
            filled_column('verr_rand',-999.,nstars),
            filled_column('v_chi2',-999.,nstars),


            # PHOTOMETRY
            filled_column('phot_source','       ',nstars),
            filled_column('phot_type','   ',nstars),
            filled_column('gmag_o',-999.,nstars),
            filled_column('rmag_o',-999.,nstars),
            filled_column('imag_o',-999.,nstars),
            filled_column('gmag_err',-999.,nstars),
            filled_column('rmag_err',-999.,nstars),
            filled_column('imag_err',-999.,nstars),
            filled_column('EBV',-999.,nstars),

            filled_column('MV_o',-999.,nstars),
            filled_column('rproj_arcm',-999.,nstars),
            filled_column('rproj_kpc',-999.,nstars),


            # EQUIVALENT WIDTHS
            filled_column('ew_naI',-999.,nstars),
            filled_column('ew_naI_err',-999.,nstars),

            filled_column('ew_mgI',-999.,nstars),
            filled_column('ew_mgI_err',-999.,nstars),

            filled_column('ew_cat',-999.,nstars),
            filled_column('ew_cat_err',-999.,nstars),
            
            filled_column('ew_feh',-999.,nstars),
            filled_column('ew_feh_err',-999.,nstars),

            filled_column('ew_w1',-999.,nstars),
            filled_column('ew_w2',-999.,nstars),
            filled_column('ew_w3',-999.,nstars),
            filled_column('ew_gl',-999.,nstars),

            filled_column('tmpl_teff',-999.,nstars),
            filled_column('tmpl_feh',-999.,nstars),


            # GAIA
            filled_column('gaia_source_id',-999,nstars),
            filled_column('gaia_pmra',-999.,nstars),
            filled_column('gaia_pmra_err',-999.,nstars),            
            filled_column('gaia_pmdec',-999.,nstars),
            filled_column('gaia_pmdec_err',-999.,nstars),
            filled_column('gaia_pmra_pmdec_corr',-999.,nstars),
            filled_column('gaia_parallax',-999.,nstars),
            filled_column('gaia_parallax_err',-999.,nstars),
            filled_column('gaia_aen',-999.,nstars),
            filled_column('gaia_aen_sig',-999.,nstars),
            filled_column('gaia_phot_variable_flag','                    ',nstars),
            filled_column('gaia_rv',-999.,nstars),
            filled_column('gaia_rv_err',-999.,nstars),
            filled_column('gaia_grvs_mag',-999.,nstars),



            # VARIABLE VELOCITY FLAGS
            filled_column('var_pval',-999.,nstars),
            filled_column('var_max_v',-999.,nstars),
            filled_column('var_max_t',-999.,nstars),
            filled_column('var_short_max_t',-999.,nstars),
            filled_column('flag_short_var',-999,nstars),

            # FLAGS AND MEMBERSHIPS 
            filled_column('flag_coadd',-999,nstars),
            filled_column('flag_var',-999,nstars),
            filled_column('flag_gaia',-999,nstars),
            filled_column('flag_HB',-999,nstars),
            filled_column('Pmem_cmd',-999.,nstars),
            filled_column('Pmem_EW',-999.,nstars),
            filled_column('Pmem_parallax',-999.,nstars),
            filled_column('Pmem_pm',-999.,nstars),
            filled_column('Pmem_feh',-999.,nstars),
            filled_column('Pmem_v',-999.,nstars),

            filled_column('Pmem',-999.,nstars),
            filled_column('Pmem_novar',-999,nstars),


            # INDIVUDAL MASK DATA
            filled_column('mask_v',-999.*np.ones(nmasks),nstars),
            filled_column('mask_v_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_coadd_v',-999*np.ones(nmasks),nstars),
            filled_column('mask_coadd_verr',-999*np.ones(nmasks),nstars),
            filled_column('mask_coadd_flag',-999*np.ones(nmasks),nstars),
            filled_column('mask_nexp',-999.*np.ones(nmasks),nstars),
            filled_column('mask_SN',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mjd',-999.*np.ones(nmasks),nstars),
            filled_column('mask_rms_arc',-999.*np.ones(nmasks),nstars),

            filled_column('mask_marz_z',-999.*np.ones(nmasks),nstars),
            filled_column('mask_marz_flag',-999*np.ones(nmasks),nstars),
            filled_column('mask_marz_tmpl',-999.*np.ones(nmasks),nstars),

            filled_column('mask_teff',-999.*np.ones(nmasks),nstars),
            filled_column('mask_logg',-999.*np.ones(nmasks),nstars),
            filled_column('mask_feh',-999.*np.ones(nmasks),nstars),
            filled_column('mask_vchi2',-999.*np.ones(nmasks),nstars),

            filled_column('mask_cat',-999.*np.ones(nmasks),nstars),
            filled_column('mask_naI',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mgI',-999.*np.ones(nmasks),nstars),
            filled_column('mask_cat_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_naI_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_mgI_err',-999.*np.ones(nmasks),nstars),
            filled_column('mask_cat_gl',-999.*np.ones(nmasks),nstars),

            filled_column('mask_flag_short_var',-999.*np.ones(nmasks),nstars),
            filled_column('mask_var_short_max_t',-999.*np.ones(nmasks),nstars)


           ]
            
    slits = Table(cols)
    return slits



######################################################
def deimos_google():
    key = '1V2aVg1QghpQ70Lms40zNUjcCrycBF2bjgs-mrp6ojI8'
    gid=1906496323
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    masklist = Table.read(url, format='csv')

    gid =0
    url = 'https://docs.google.com/spreadsheets/d/{0}/export?format=csv&gid={1}'.format(key, gid)
    objlist = ascii.read(url, format='csv')
    
    return objlist,masklist


######################################################
# FILL A COLUMN
def filled_column(name, fill_value, size):
    """
    Tool to allow for large strings
    """
    return Column([fill_value]*int(size), name)


######################################################
def combine_mask_velocities(stars):
    
    # COMBINE STARS WITH MEASURED VELOCITIES
    mgood       = stars['dmost_v_err'] > 0.
    good_stars  = stars[mgood]
    t_exp=0
    
    v,verr, v_chi2, teff,feh,ncomb = [-999.,-999.,-999,-999.,-999.,0]
    verr_rand,verr_sys = [-999.,-999.]
    if (np.size(good_stars) == 1):
        v     = float(good_stars['dmost_v'][0])
        teff  = float(good_stars['chi2_teff'][0])
        feh   = float(good_stars['chi2_feh'][0])

        verr_rand = 0
#        verr_rand = good_stars['dmost_v_err_rand']
        verr_sys  = float(good_stars['dmost_v_err'][0])

        v_chi2   = np.mean(good_stars['v_chi2'])
        t_exp    = float(good_stars['texp'][0])

    
    if np.size(good_stars) > 1:
        vt,et,ets,tt,ft = [],[],[],[],[]
        for obj in good_stars:
            vt   = np.append(vt,obj['dmost_v'])
            tt   = np.append(tt,obj['chi2_teff'])
            ft   = np.append(ft,obj['chi2_feh'])
            et   = np.append(et,obj['dmost_v_err'])
            ets  = np.append(ets,obj['dmost_v_err_rand'])
            t_exp= t_exp + obj['texp']

        sum1 = np.sum(1./et**2)
        sum1s = np.sum(1./ets**2)

        sum2 = np.sum(vt/et**2)
        sum3 = np.sum(tt/et**2)
        sum4 = np.sum(ft/et**2)

        v    = sum2/sum1                   
        verr_sys  = np.sqrt(1./sum1)      
        verr_rand = np.sqrt(1./sum1s)     # This isn't used, but keeping anyways

        teff = sum3/sum1
        feh  = sum4/sum1 

        v_chi2       = np.mean(good_stars['v_chi2'])


    return v, verr_rand,verr_sys, v_chi2, teff, feh, t_exp




#####################################
#####################################
def set_binary_flag(alldata):


    ns, nvar = 0,0
    for i,obj in enumerate(alldata):
        

        # FIRST SET INNER MASK VARIABLE FLAG
        mnv = any(obj['mask_flag_short_var'] == 0)
        if (np.sum(mnv) > 0):
            mxt = np.max(obj['mask_var_short_max_t'][mnv])
            alldata['var_short_max_t'][i]= mxt
            alldata['flag_short_var'][i] = 0
        

        mvs       = obj['mask_flag_short_var'] == 1
        if  (np.sum(mvs) > 0):
            mxt = np.max(obj['mask_var_short_max_t'][mvs])
            alldata['var_short_max_t'][i]= mxt
            alldata['flag_short_var'][i] = 1
        


        m = (obj['mask_v_err'] > 0)
        if np.sum(m) > 1:

            alldata['flag_var'][i]  = 0
            ns=ns+1

            v_mean = np.average(obj['mask_v'][m],weights=1./(obj['mask_v_err'][m])**2)
            chi2   = np.sum((obj['mask_v'][m] - v_mean)**2/(obj['mask_v_err'][m]**2))
            pv     = 1 - stats.chi2.cdf(chi2, np.sum(m)-1.)


            if (pv == 0) | (pv < 1e-14):
                pv = 1e-14

            lpv = np.log10(pv)

            alldata['var_pval'][i]  = lpv
            alldata['var_max_v'][i] = np.max(obj['mask_v'][m]) - np.min(obj['mask_v'][m])
            alldata['var_max_t'][i] = 24*(np.max(obj['mask_mjd'][m])-np.min(obj['mask_mjd'][m]))
            alldata['flag_var'][i]  = 0


            # SET VARIABLE THRESHOLD
            # MAXTED suggests -4, but setting looser threshold of -1
            if lpv < -1:
                alldata['flag_var'][i]  = 1
                nvar = nvar + 1

    print('VVAR: Setting {} of {} repeats as velocity variable'.format(nvar,ns))

    return alldata



#####################################
def combine_mask_ew(stars):
    
    
    # COMBINE STARS WITH MEASURED VELOCITIES
    mgood       = (stars['dmost_v_err'] > 0.) & (stars['cat_err'] > 0.)
    good_stars  = stars[mgood]
    
    cat,cat_err,naI,naI_err,mgI,mgI_err, gl,ncomb = [-999.,-999.,-999.,-999.,-999.,-999.,-999.,0]
    w1,w2,w3 = [-999.,-999.,-999.]

    # (a size==1 shortcut used to live here, but the block below already
    # handles it correctly via the weighted-sum formulas -- for a single
    # star that reduces exactly to cat=ct[0], cat_err=cterr[0], etc. The
    # shortcut was dead code: it ran first, then got unconditionally
    # overwritten by this block, which also runs for size==1.)
    if np.size(good_stars) >= 1:
        ct,cterr,na,naerr,mg,mgerr = [],[],[],[],[],[]
        for obj in good_stars:
            ct   = np.append(ct,obj['cat'])
            na   = np.append(na,obj['naI'])
            mg   = np.append(mg,obj['mgI'])
            

            # APPLY SYSTEMATIC ERROR FOR CaT HERE
            ctmp = np.sqrt((CAT_SYS_MULT * obj['cat_err'])**2 + CAT_SYS_FLOOR**2)
            cterr   = np.append(cterr,ctmp)

            # 0.05A SYSTEMATIC ALREADY APPLIED FOR THESE LINES
            naerr   = np.append(naerr,obj['naI_err'])
            mgerr   = np.append(mgerr,obj['mgI_err'])

            ncomb=ncomb+1
            
        sum1 = np.sum(1./cterr**2)
        sum2 = np.sum(ct/cterr**2)
        
        sum1n = np.sum(1./naerr**2)
        sum2n = np.sum(na/naerr**2)
        
        sum1m = np.sum(1./mgerr**2)
        sum2m = np.sum(mg/mgerr**2)
        
      
        cat = sum2/sum1
        naI = sum2n/sum1n
        mgI = sum2m/sum1m

        #  ADD SYSTEMATIC ERROR FLOOR TO COADDS -- CaT's per-observation
        #  floor (CAT_SYS_FLOOR) still shrinks with sqrt(N) as more masks
        #  are combined, so a hard floor on the COADDED value is kept too,
        #  guaranteeing it never reports below CAT_SYS_FLOOR regardless
        #  of how many masks contribute.
        sys_err = 0.05
        cat_err = np.sqrt(1./sum1)
        naI_err = np.sqrt(1./sum1n)
        mgI_err = np.sqrt(1./sum1m)
        if cat_err < CAT_SYS_FLOOR:
            cat_err = CAT_SYS_FLOOR
        if naI_err < sys_err:
            naI_err = 0.05
        if mgI_err < sys_err:
            mgI_err = 0.05


    for obj in good_stars:
        w1       = obj['w1']
        w2       = obj['w2']
        w3       = obj['w3']
        gl       = obj['cat_gl']
        if (w1 < 0) & (w1 > -500):
            print(obj['w1'],obj['w2'],obj['w3'])

    return cat,cat_err,mgI,mgI_err,naI,naI_err, w1,w2,w3, gl,ncomb 



###########################################
def combine_mask_marz(star):
    '''
    Templates QUASAR     = 12
              Elliptical = 6
    '''
  
    mset       = star['marz_flag'] > -1
    marz_obj   = star[mset]
    marz_flag, marz_z, t_exp = -999,-999.,0



    # COPY SINGLE MEASUREMENT
    if (np.size(marz_obj) == 1):
        marz_z    = marz_obj['marz_z'][0]
        marz_flag = marz_obj['marz_flag'][0]
        if (marz_flag > 2):     
            t_exp     = np.sum(marz_obj['texp'])


    # PARSE MULTIPLE MEASUREMENTS
    # Priority order 2 -> 3 -> 4 -> 6 -> 1 -> average: each block is
    # guarded on marz_flag still being unset (-999) so a higher-priority
    # match can't be silently overwritten by a later, lower-priority one.
    if (np.size(marz_obj) >1):

        if (marz_flag == -999) & np.any(marz_obj['marz_flag'] == 2):
            marz_flag    = 2


        # IF ANY EXP IS QSO, SET AS 6
        if (marz_flag == -999) & np.any(marz_obj['marz_flag'] == 3):
            marz_flag   = 3
            marz_z = np.mean(marz_obj['marz_z'][marz_obj['marz_flag'] == 3])

        # IF ANY EXP IS GOOD GALAXY, SET AS 4
        if (marz_flag == -999) & np.any(marz_obj['marz_flag'] == 4):
            marz_flag   = 4
            marz_z = np.mean(marz_obj['marz_z'][marz_obj['marz_flag'] == 4])

        # IF ANY EXP IS QSO, SET AS 6
        if (marz_flag == -999) & np.any(marz_obj['marz_flag'] == 6):
            marz_flag  = 6
            marz_z = np.mean(marz_obj['marz_z'][marz_obj['marz_flag'] == 6])


        if (marz_flag == -999) & np.any(marz_obj['marz_flag'] == 1):
            marz_flag = 1
            marz_z = np.mean(marz_obj['marz_z'][marz_obj['marz_flag'] == 1])


        # ELSE AVERAGE THE FLAGS
        if (marz_flag == -999):
            marz_flag = np.mean(marz_obj['marz_flag'])

        # SET EXP TIME IF EXTRAGALACTIC
        if (marz_flag > 2):
            t_exp= np.sum(marz_obj['texp'])

    return marz_z, marz_flag, t_exp



###########################################
def set_v_chi2(slits):

    v_chi2 = np.median(slits['emcee_lnprob'],1)
    m = v_chi2 ==0
    v_chi2[m] = slits['coadd_lnprob'][m]

    t = Column(v_chi2, name='v_chi2')
    slits.add_column(t)

    return slits

###########################################
def read_dmost_files(masklist):
    n=0

    data_dir = os.getenv('DEIMOS_REDUX')
    allslits = []

    for msk in masklist:

        maskname = msk['MaskName']        
        dmost_file = glob.glob(data_dir + '*'+maskname+'/dmost/dmost*'+maskname+'.fits')
        if (np.size(dmost_file) > 0) & (msk['pypeit_redux'] != 'N'):

            slits, mask = dmost_utils.read_dmost(dmost_file[0])
            nexp  = mask['nexp'][0]
            sltwd = mask['slitwidth'][0]
            texp  = np.sum(mask['exptime'])


            nslits = np.size(slits)
            maskname = filled_column('maskname',maskname,nslits)

            mjd      = filled_column('mean_mjd',np.mean(mask['mjd']),nslits)
            nexp     = filled_column('nexp',nexp,nslits)
            slitwidth= filled_column('slitwidth',sltwd,nslits)
            texp     = filled_column('texp',texp,nslits)

            # 
            slits = set_v_chi2(slits)


            # KEEP ONLY MASK AVERAGED QUANTITIES
            new_slits = Table([maskname, mjd,nexp,slitwidth,texp,slits['objname'],slits['RA'],slits['DEC'],slits['collate1d_filename'],\
                             slits['collate1d_SN'], slits['marz_z'],slits['marz_flag'],slits['serendip'],\
                             slits['chi2_teff'],slits['chi2_logg'],slits['chi2_feh'],slits['rms_arc'],\
                             slits['dmost_v'],slits['dmost_v_err'],slits['dmost_v_err_rand'],slits['v_chi2'],\
                             slits['v_nexp'],slits['coadd_flag'],\
                             slits['coadd_v'],slits['coadd_v_err'],\
                             slits['var_short_pval'], slits['var_short_max_v'],slits['var_short_max_t'],slits['flag_short_var'],\
                             slits['cat'],slits['cat_err'],slits['cat_gl'],\
                             slits['naI'],slits['naI_err'],\
                             slits['mgI'],slits['mgI_err']])
                             
            new_slits.add_column(slits['cat_all'][:,0],name='w1')
            new_slits.add_column(slits['cat_all'][:,1],name='w2')
            new_slits.add_column(slits['cat_all'][:,2],name='w3')

            # UPDATE MARZ FLAG IF MISSING
            m=slits['marz_flag'] < 1
            slits['marz_flag'][m] = 1

            # CREATE OR APPEND TO ALL TABLE
            if (n==0):  allslits = new_slits
            if (n > 0): allslits = table.vstack([allslits,new_slits])
            n=n+1
        else:
            print('Skipping mask {}'.format(msk['MaskName']))



    return allslits, n

###########################################
def get_unique_spectra(allslits):
    """
    Group repeat slits of the same star (within 1.0" on sky) into unique
    stars. Complete-linkage clustering on the 1.0" adjacency graph: a
    candidate pair is only merged into one group if every existing member
    on each side is within 1.0" of every member on the other side (i.e.
    the merged group is a clique in the 1.0" graph), not just
    chain-connected through an intermediate slit. This avoids incorrectly
    merging genuinely distinct, nearby stars via a chain through an
    independently-astrometered auxiliary-mask target (e.g. Draco's RRL
    follow-up masks) while still combining true repeat observations of
    the same star.

    Returns (nstars, group_id): nstars is the number of unique stars,
    group_id[k] in [0, nstars) labels which unique star slit k belongs to.
    """
    n = len(allslits)
    cdeimos = SkyCoord(ra=allslits['RA']*u.degree, dec=allslits['DEC']*u.degree)
    idx1, idx2, sep2d, _ = cdeimos.search_around_sky(cdeimos, 1.0*u.arcsec)

    # CANONICAL, DEDUPED EDGE LIST, PROCESSED CLOSEST PAIR FIRST
    pairs = {}
    for a, b, s in zip(idx1, idx2, sep2d.arcsec):
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        pairs[key] = s
    edges = set(pairs.keys())
    ordered = sorted(pairs.items(), key=lambda kv: kv[1])

    parent  = np.arange(n)
    members = {i: [i] for i in range(n)}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def linked(i, j):
        return (i, j) in edges if i < j else (j, i) in edges

    for (a, b), s in ordered:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # ONLY MERGE IF EVERY CROSS-PAIR IS WITHIN 1.0" (CLIQUE, NOT CHAIN)
        if all(linked(i, j) for i in members[ra] for j in members[rb]):
            parent[ra] = rb
            members[rb].extend(members[ra])
            del members[ra]

    labels = np.array([find(k) for k in range(n)])
    _, group_id = np.unique(labels, return_inverse=True)
    nstars = int(group_id.max()) + 1 if n > 0 else 0

    return nstars, group_id

###########################################
def combine_mask_quantities(nmasks, nstars, group_id, sc_gal, allslits):

    # CREATE DATA TABLE
    single_mask =0
    if (nmasks ==1):
        nmasks = 2
        single_mask =1
    dmost_allstar  = create_allstars(nmasks, nstars)

    # ONE PASS PER UNIQUE STAR (group_id groups repeat slits within 1.0",
    # see get_unique_spectra) -- i IS the output row index directly, so
    # every row 0..nstars-1 is filled and no post-hoc trim is needed.
    for i in range(nstars):

        m = group_id == i
        group_slits = allslits[m]
        nrpt = np.sum(m)
        if nrpt == 0:
            continue
        obj = group_slits[0]

        dmost_allstar['RA'][i]      = obj['RA']
        dmost_allstar['DEC'][i]     = obj['DEC']
        dmost_allstar['objname'][i] = obj['objname']
        dmost_allstar['slitwidth'][i] = round(obj['slitwidth'], 2)
        dmost_allstar['mean_mjd'][i] = obj['mean_mjd']


        if (obj['serendip'] > 0):
            dmost_allstar['serendip'][i] = 1
        else:
            dmost_allstar['serendip'][i] = 0

        for j,robj in enumerate(group_slits):

            # KEEP TRACK OF MASK NAMES
            if (j==0):
                dmost_allstar['masknames'][i]   = obj['maskname']
                dmost_allstar['collate1d_filename'][i]  = robj['collate1d_filename']

            if (j > 0):
                dmost_allstar['masknames'][i]   = dmost_allstar['masknames'][i]+'+'+robj['maskname']

            c1 = (j == 0)
            c2 = (j > 0) & (single_mask ==0)

            # IF FIRST OR SINGLE MASK
            if (c1 | c2):

                dmost_allstar['mask_v'][i,j]     = robj['dmost_v']
                dmost_allstar['mask_v_err'][i,j] = robj['dmost_v_err']

                dmost_allstar['mask_coadd_v'][i,j]    = robj['coadd_v']
                dmost_allstar['mask_coadd_verr'][i,j] = robj['coadd_v_err']
                dmost_allstar['mask_coadd_flag'][i,j] = robj['coadd_flag']

                dmost_allstar['mask_SN'][i,j]    = robj['collate1d_SN']
                dmost_allstar['mask_nexp'][i,j]  = robj['v_nexp']
                dmost_allstar['mask_mjd'][i,j]   = robj['mean_mjd']
                dmost_allstar['mask_rms_arc'][i,j]   = robj['rms_arc']


                dmost_allstar['mask_marz_flag'][i,j] = robj['marz_flag']
                dmost_allstar['mask_marz_z'][i,j]    = robj['marz_z']

                dmost_allstar['mask_teff'][i,j]  = robj['chi2_teff']
                dmost_allstar['mask_logg'][i,j]  = robj['chi2_logg']
                dmost_allstar['mask_feh'][i,j]   = robj['chi2_feh']
                dmost_allstar['mask_vchi2'][i,j]   = robj['v_chi2']


                dmost_allstar['mask_cat'][i,j]   = robj['cat']
                dmost_allstar['mask_naI'][i,j]   = robj['naI']
                dmost_allstar['mask_mgI'][i,j]   = robj['mgI']
                dmost_allstar['mask_cat_gl'][i,j]   = robj['cat_gl']

                dmost_allstar['mask_cat_err'][i,j]  = robj['cat_err']
                dmost_allstar['mask_naI_err'][i,j]  = robj['naI_err']
                dmost_allstar['mask_mgI_err'][i,j]  = robj['mgI_err']

                dmost_allstar['mask_flag_short_var'][i,j]  = robj['flag_short_var']
                dmost_allstar['mask_var_short_max_t'][i,j] = robj['var_short_max_t']



        # COMBINE VELOCITIES
        v, verr_rand,verr_sys, vchi2, teff, feh, t_exp = combine_mask_velocities(group_slits)
        dmost_allstar['v'][i]      = v
        dmost_allstar['verr_rand'][i]  = verr_rand
        dmost_allstar['v_err'][i]  = verr_sys
        dmost_allstar['v_chi2'][i] = vchi2
        dmost_allstar['t_exp'][i]  = t_exp


        dmost_allstar['nmask'][i]  = nrpt
        dmost_allstar['nexp'][i]   = np.sum(group_slits['nexp'])
        dmost_allstar['flag_coadd'][i]  = np.max(group_slits['coadd_flag'])

        dmost_allstar['tmpl_teff'][i]  = teff
        dmost_allstar['tmpl_feh'][i]   = feh

        # COMBINE EW
        cat,cat_err,mgI,mgI_err,naI,naI_err, w1,w2,w3,gl, ncomb = combine_mask_ew(group_slits)
        dmost_allstar['ew_cat'][i]       = cat
        dmost_allstar['ew_cat_err'][i]   = cat_err
        dmost_allstar['ew_mgI'][i]       = mgI
        dmost_allstar['ew_mgI_err'][i]   = mgI_err
        dmost_allstar['ew_naI'][i]       = naI
        dmost_allstar['ew_naI_err'][i]   = naI_err
        dmost_allstar['ew_w1'][i]        = w1
        dmost_allstar['ew_w2'][i]        = w2
        dmost_allstar['ew_w3'][i]        = w3
        dmost_allstar['ew_gl'][i]        = gl


        # CALCULATE Mean SN
        # CATCH STRANGE CASES
        msn = dmost_allstar['mask_SN'][i,:] > -1
        dmost_allstar['SN'][i] = np.sum(dmost_allstar['mask_SN'][i,msn]) / np.sqrt(np.size(dmost_allstar['mask_SN'][i,msn]))
        if dmost_allstar['SN'][i] > 700:
            dmost_allstar['SN'][i] = np.min(dmost_allstar['mask_SN'][i,msn])
            if dmost_allstar['SN'][i] > 700:
                dmost_allstar['SN'][i] = 500

       # COMBINE MARZ
        zgal, zflag, zexp  = combine_mask_marz(group_slits)
        dmost_allstar['marz_z'][i]    = zgal
        dmost_allstar['marz_flag'][i] = zflag
        if (zexp > 0):
            dmost_allstar['t_exp'][i]     = zexp


  # SET EXTRAGALACTIC VALUES
    mgal  = dmost_allstar['marz_flag']  > 2
    dmost_allstar['v'][mgal]     =  dmost_allstar['marz_z'][mgal]*3e5
    dmost_allstar['v_err'][mgal] =  0


    return dmost_allstar


######################################################
def combine_masks(object_name, max_obs_date = 20500101,file_create_date='',**kwargs):


    DEIMOS_REDUX  = os.getenv('DEIMOS_REDUX')
    outfile       = DEIMOS_REDUX + '/dmost_alldata/dmost_alldata_'+file_create_date+'_'+object_name+'.fits'  
    full_outfile  = DEIMOS_REDUX + '/dmost_alldata/full_dmost_alldata_'+file_create_date+'_'+object_name+'.fits'  

    if not ('objlist' in kwargs):
        objlist, masklist = deimos_google()
    else:
        objlist  = kwargs['objlist']
        masklist = kwargs['masklist']



    # CHECK FOR DATE LIMITS ON MASKS
    m = (masklist['DateObs'] < max_obs_date) & (masklist['pypeit_redux'] != 'S')

    masklist=masklist[m]

    # PROPERTIES OF OBJECT
    object_properties = objlist[(objlist['Name2'] == object_name)]
    sc_gal            = SkyCoord(object_properties['RA'],object_properties['Dec'], unit=(u.deg, u.deg))


    # MASK LIST FOR OBJECT
    mobj   = (masklist['Object'] == object_name) 
    nmasks = np.sum(mobj)
    print('{} Combining {} masks'.format(object_name,nmasks))


    # READ AND COMBINE ALL DMOST FILES
    alldata, nrun = read_dmost_files(masklist[mobj])
    if (nrun == 0):
        print('No masks run for this object, skipping')
        print()
        alldata=[]
        return alldata


    # HOW MANY UNIQUE SPECTRA?
    nstars, group_id = get_unique_spectra(alldata)


    # CREATE AND POPULATE FINAL DATA TABLE
    alldata  = combine_mask_quantities(nmasks, nstars, group_id, sc_gal, alldata)

    # SET BINARY FLAGS
    alldata  = set_binary_flag(alldata)


    # MATCH PHOTOMETRY, MATCH GAIA
    alldata = dmost_photometry_gaia.match_photometry(object_properties[0],alldata)
    alldata = dmost_photometry_gaia.match_gaia(object_properties[0],alldata)



    # CATCH EW NON-DETECTIONS
    thrs = 50
    mn = alldata['ew_cat_err'] > thrs
    alldata['ew_cat_err'][mn] = -999.
    mn = alldata['ew_naI_err'] > thrs
    alldata['ew_naI_err'][mn] = -999.    
    mn = alldata['ew_mgI_err'] > thrs
    alldata['ew_mgI_err'][mn] = -999.

    # MEMBERSHIP
    Pmem, Pmem_novar, Pmem_cmd, Pmem_EW,Pmem_px,Pmem_pm,Pmem_feh,\
                      Pmem_v = dmost_membership.find_members(alldata,object_properties[0])

    alldata['Pmem']          = Pmem
    alldata['Pmem_novar']    = Pmem_novar

    alldata['Pmem_cmd']      = Pmem_cmd
    alldata['Pmem_EW']       = 1*Pmem_EW
    alldata['Pmem_parallax'] = 1*Pmem_px
    alldata['Pmem_pm']       = Pmem_pm
    alldata['Pmem_feh']      = Pmem_feh
    alldata['Pmem_v']        = Pmem_v


    alldata['flag_HB']    = dmost_membership.flag_HB_stars(alldata,object_properties[0])

    if object_name == 'Eri':
        gr=alldata['gmag_o'] - alldata['rmag_o']
        m=(alldata['Pmem'] > 0.5) & (gr < 0.5) & (alldata['flag_HB']==0)
        alldata['flag_HB'][m] = 1
        print(np.sum(m))


    # ENSURE [Fe/H] IS ONLY IN CALIBRATED REGIONS FOR MEMBER STARS
    mhb   = (alldata['flag_HB'] == 1) | (alldata['tmpl_teff'] >7000)
    alldata['ew_feh'][mhb]     = -999.
    alldata['ew_feh_err'][mhb] = -999.
    mcalib = (alldata['MV_o'] > 3.) | (alldata['ew_feh_err'] < 0) | (alldata['ew_cat_err'] < 0)| (alldata['Pmem'] < 0.2)
    alldata['ew_feh'][mcalib]     = -999.
    alldata['ew_feh_err'][mcalib] = -999.


    # CLEAN BAD PHOTOMETRY VALUES
    mphot = (alldata['rmag_err'] > 2) | (alldata['rmag_o'] < -100.)| (alldata['rmag_o'] > 30.)
    alldata['rmag_o'][mphot] = -999.
    alldata['rmag_err'][mphot] = -999.
    mphot = (alldata['gmag_err'] > 2) | (alldata['gmag_o'] < -100.)| (alldata['gmag_o'] > 30.)
    alldata['gmag_o'][mphot] = -999.
    alldata['gmag_err'][mphot] = -999.

  # ENSURE GOOD VELOCITY CUT
    mverr = alldata['v_err'] > 15
    alldata['Pmem_novar'][mverr] = 0
    alldata['Pmem'][mverr] = 0

  # SET EXTRAGALACTIC OBJECTS
    mg = alldata['marz_flag'] > 2
    alldata['Pmem_novar'][mg] = 0
    alldata['Pmem'][mg] = 0


    # SET SYSTEM NAME
    alldata['system_name'] = object_name



    # REMOVE MASK-LEVEL DATA, BUT WRITE OUT FULL FILE FIRST
    alldata.write(full_outfile, overwrite=True)

    alldata.remove_columns(['mask_v','mask_v_err','mask_nexp','mask_SN','mask_mjd','mask_rms_arc',\
                            'mask_marz_z','mask_marz_flag','mask_marz_tmpl','mask_coadd_v','mask_coadd_verr','mask_coadd_flag',\
                            'mask_teff','mask_feh','mask_logg','mask_cat_gl','mask_vchi2',\
                            'mask_cat','mask_cat_err','mask_naI','mask_naI_err','mask_mgI','mask_mgI_err',\
                            'mask_flag_short_var','mask_var_short_max_t'])


    print('{} Combined {} masks with {} unique stars, {} measured velocities'.format(object_name,nmasks,nstars,np.sum(alldata['v_err'] > 0)))
    print('{} There are {} member stars and {} pure member stars'.format(object_name,np.sum(Pmem > 0.5),np.sum(Pmem_novar> 0)))
    print()
    alldata.write(outfile, overwrite=True)
            

    return alldata


#def combine_all():
#    objlist, masklist = deimos_google()
#    for obj in objlist:       
#        if obj['Phot'] != 'PanS':
#            tmp  = combine_masks(obj['Name2'])



#####################################################    
def main(*args):


    mask = sys.argv[1]
    
    DEIMOS_RAW     = os.getenv('DEIMOS_RAW')
    DEIMOS_REDUX   = os.getenv('DEIMOS_REDUX')
    
    alldata = combine_masks(object_name)
    
if __name__ == "__main__":
    main()
    
