# Generate transmission spectrum with PandExo and simulate transits with
# spot occultations.

import sys
sys.path.append('/home/giovanni/Dropbox/Shelf/python/pandexo/')
import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import pandexo.engine.justplotit as jpi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pysynphot
import pathlib
import rebin
import pickle, os
from astropy.io import fits
from pdb import set_trace
sys.path.append('/home/giovanni/Dropbox/Projects/granulation/detect_LD_variations/code/Light-curve-tools/')
import ld_coeffs
from astropy.convolution import convolve

modelsfolder = '/home/giovanni/Dropbox/Shelf/stellar_models/phxinten/HiRes/'
foldthrough = '/home/giovanni/Dropbox/Shelf/filters/'
thrfile1 = foldthrough + 'JWST_NIRCam.F150W2.dat'
thrfile2 = foldthrough + 'JWST_NIRCam.F322W2.dat'
thrfile3 = foldthrough + 'JWST_NIRCam.F444W.dat'

def generate_spectrum_jwst(pardict):

    print('TO ADD: 3D STELLAR MODELS FOR FGK STARS (NOT M)')

    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = 80  #saturation level in percent
                                               #of full well
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 1   #number of transits
    exo_dict['observation']['R'] = None    #fixed binning. I usually suggest ZERO
                                         #binning.. you can always bin later
                                         #without having to redo the calcualtion
    exo_dict['observation']['baseline'] = 1.0   #fraction of time in transit
                                                #versus out = in/out
    exo_dict['observation']['baseline_unit'] = 'frac'
    exo_dict['observation']['noise_floor'] = 20   #this can be a fixed level
                                                  #or it can be a filepath
    exo_dict['star']['type'] = 'phoenix'  #phoenix or user (if you have your own)
    exo_dict['star']['mag'] = pardict['magstar']     #magnitude of the system
    exo_dict['star']['ref_wave'] = 1.25   #For J mag = 1.25, H = 1.6, K =2.22..
                                          #etc (all in micron)
    exo_dict['star']['temp'] = pardict['tstar']     #in K
    exo_dict['star']['metal'] = 0.0        # as log Fe/H
    exo_dict['star']['logg'] = pardict['loggstar']
    exo_dict['star']['radius'] = pardict['rstar']
    exo_dict['star']['r_unit'] = 'R_sun'

    exo_dict['planet']['type'] = pardict['planettype']
    exo_dict['planet']['w_unit'] = 'um'
    exo_dict['planet']['radius'] = pardict['rplanet'] #other options include
                                                      #"um","nm" ,"Angs",
                                                      #"secs" (for phase curves)
    exo_dict['planet']['r_unit'] = 'R_jup'
    exo_dict['planet']['transit_duration'] = 2.0
    exo_dict['planet']['td_unit'] = 'h'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'


    #inst_dict = jdi.load_mode_dict('NIRSpec Prism')
    #inst_dict["configuration"]["detector"]["subarray"] = 'sub512'
    #inst_dict["configuration"]["detector"]["readmode"] = 'nrs'
    set_trace()
    print('Starting PandExo run for JWST')
    jdi.run_pandexo(exo_dict, ['NIRSpec Prism'], save_file=True, \
            output_file=pardict['pandexo_out_jwst'])
    '''
    # Load in output from run
    out = pickle.load(open(pardict['pandexo_out'],'rb'))

    #for a single run
    x, y, e = jpi.jwst_1d_spec(out, R=100, num_tran=2, model=True)
                                #$, x_range=[.8,1.28])

    # Determine errorbar on the base of pandexo e- counts and out of transit SNR
    signal = jpi.jwst_1d_flux(out)[1]
    wls = jpi.jwst_1d_flux(out)[0]
    snr = jpi.jwst_1d_snr(out)[1]
    wl = jpi.jwst_1d_snr(out)[0]
    # Why is there one more value in the wavelengths here?
    ssignal = interp1d(wls, signal)
    signal_wl = ssignal(x[0])
    ssnr = interp1d(wl, snr)
    snr_wl = ssnr(x[0])
    sigma = signal_wl/snr_wl
    relsigma = sigma/signal_wl
    spec_obs = [x[0], y[0], e[0], relsigma]
    savefile = open(pardict['pandexo_out'], 'wb')
    pickle.dump(spec_obs, savefile)
    savefile.close()
    '''
    return #relsigma

def generate_spectrum_hst(pardict):

    exo_dict = jdi.load_exo_dict()
    exo_dict['star']['jmag'] = pardict['magstar']
    exo_dict['star']['hmag'] = pardict['magstar']
    exo_dict['planet']['type'] = pardict['planettype']
    #exo_dict['planet']['exopath'] = 'WASP43b-Eclipse_Spectrum.txt' # filename for model spectrum
    exo_dict['planet']['w_unit'] = 'um'
    exo_dict['planet']['f_unit'] = 'fp/f*'
    exo_dict['planet']['depth'] = (pardict['rplanet']/pardict['rstar'])**2
    exo_dict['planet']['i'] = 88. # Orbital inclination in degrees
    exo_dict['planet']['ars'] = 7.38 # a/R*
    exo_dict['planet']['period'] = 2.0  # Orbital period in days
    exo_dict['planet']['transit_duration']= 2./24 #transit duration in days
    exo_dict['planet']['w'] = 90 #(optional) longitude of periastron. Default is 90
    exo_dict['planet']['ecc'] = 0 #(optional) eccentricity. Default is 0

    inst_dict = jdi.load_mode_dict('WFC3 G141')

    exo_dict['observation']['noccultations'] = 5 # Number of transits/eclipses
    inst_dict['configuration']['detector']['subarray'] = 'GRISM256'   # GRISM256 or GRISM512
    inst_dict['configuration']['detector']['nsamp'] = 'optimal' # WFC3 N_SAMP, 1..15
    inst_dict['configuration']['detector']['samp_seq'] = 'optimal'  # WFC3 SAMP_SEQ, SPARS5 or SPARS10
    inst_dict['strategy']['norbits'] = 4 # Number of HST orbits
    inst_dict['strategy']['nchan'] = 15 # Number of spectrophotometric channels
    inst_dict['strategy']['scanDirection'] = 'Forward'  # Spatial scan direction, Forward or Round Trip
    inst_dict['strategy']['schedulability'] = 30 # 30 for small/medium program, 100 for large program
    inst_dict['strategy']['windowSize'] = 20 # (optional) Observation start window size in minutes. Default is 20 minutes.
    inst_dict['strategy']['calculateRamp'] = False

    jdi.run_pandexo(exo_dict, inst_dict, param_space=0, param_range=0, \
                    save_file=True, output_file=pardict['pandexo_out_jwst'])

    return

def read_pandexo_results(pardict, instrument, res=4):
    '''
    Used for both HST and JWST simulations.

    res: resolution for JWST observations.
    '''

    if instrument == 'jwst':
        spec_obs = pickle.load(open(pardict['pandexo_out_jwst'], 'rb'))
        xobs, yobs, yobs_err = jpi.jwst_1d_spec(spec_obs, R=res, \
                    num_tran=1, model=False, plot=False)
        xobs = np.array(xobs)[0]
        yobs = np.array(yobs)[0]
        yobs_err = np.array(yobs_err)[0]
    elif instrument == 'hst':
        spec_obs = pickle.load(open(pardict['pandexo_out'], 'rb'))
        xobs, yobs, yobs_err, modelwave, modelspec \
                        = jpi.hst_spec(spec_obs, plot=False)

    plt.errorbar(xobs, yobs, yerr=yobs_err, fmt='o')
    plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
    plt.ylabel('Transit depth', fontsize=16)
    plt.title('Transmission spectrum output from PandExo', fontsize=16)
    plt.show()
    plt.savefig(pardict['data_folder'] + 'spec_model_' + instrument + '.pdf')
    plt.close('all')

    return xobs, yobs, yobs_err

def add_spots(pardict, instrument, resol=10, simultr=None, models=['phoenix']):
    '''
    Parameters
    ----------
    simultr: whether the transit depth uncertainties are calculated with
    PandExo (None), provided with a file (file name), or both (file name
    with '+' appended), or also explicitely given (list of w, f, err arrays).
    '''

    print('***Pplanet = 2 days')

    if simultr == None:
        xobs, yobs, yobs_err \
                    = read_pandexo_results(pardict, instrument, res=resol)
        #xobs -= 0.5
        flag = xobs < 6.5
        xobs, yobs, yobs_err = xobs[flag], yobs[flag], yobs_err[flag]
        #yobs_err /= 2.
        # Rebin from resolution 100 to res
        #xobs = rebin.rebin_wave(xobs_, 70)
        #yobs, yobs_err = rebin.rebin_spectrum(yobs_, xobs_, xobs, unc=yobs_err_)
    elif type(simultr) == 'str' and rsimultr.endswith('+'):
        xobs1, yobs1, yobs_err1 = np.loadtxt(uncert_file[:-1], unpack=True)
        xobs2, yobs2, yobs_err2 = read_pandexo_results(pardict, instrument, \
                                    res=resol)
        xobs = np.concatenate((xobs1, xobs2))
        yobs = np.concatenate((yobs1, yobs2))
        yobs_err = np.concatenate((yobs_err1, yobs_err2))
    elif type(simultr) == 'str' and not simultr.endswith('+'):
        xobs, yobs, yobs_err = np.loadtxt(uncert_file, unpack=True)
    elif len(simultr) == 3:
        # First and last one must have some rebinning problem
        xobs, yobs, yobs_err = simultr[0], simultr[1], simultr[2]

    # Save transmission spectrum
    savespec = open(pardict['data_folder'] + 'spec_model_' + instrument \
                    + '.pic', 'wb')
    pickle.dump([xobs, yobs, yobs_err], savespec)
    savespec.close()

    # Calculate LD coefficients
    LDcoeffs = [[], []]
    if pardict['instrument'] == 'NIRSpec_Prism/':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam/':
        ldendname = 'nircam'

    if (pardict['magstar'] == 4.5 or pardict['magstar'] == 10.5) \
        and not pathlib.Path(pardict['project_folder'] \
                + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
                            + 'K/' + 'LDcoeffs_' + ldendname + '.pic').exists():
        print('Computing LD coefficients...')
        for i in np.arange(len(xobs)):
            if i < len(xobs) - 2:
                wl_bin_low = xobs[i] - np.diff(xobs)[i]/2.
                wl_bin_up = xobs[i] + np.diff(xobs)[i + 1]/2.
            else:
                wl_bin_low = xobs[i] - np.diff(xobs)[i - 1]/2.
                wl_bin_up = xobs[i] + np.diff(xobs)[i - 1]/2.
            if ldendname == 'prism':
                thrfile = 'None'
            elif ldendname == 'nircam' and xobs[i] < 2.4:
                thrfile \
                = '/home/giovanni/Dropbox/Shelf/filters/JWST_NIRCam.F150W2.dat'
            elif ldendname == 'nircam' and xobs[i] >= 2.4:
                thrfile \
                = '/home/giovanni/Dropbox/Shelf/filters/JWST_NIRCam.F322W2.dat'

            ww_, LDcoeffs_ = ld_coeffs.fit_law(pardict['tstar'], \
                pardict['loggstar'], 0.0, thrfile=thrfile, grid='phoenix', \
                wlmin=wl_bin_low, wlmax=wl_bin_up, nchannels=1, plots=False)
            LDcoeffs[0].append(LDcoeffs_)

        LDcoeffs[1] = [xobs, yobs, yobs_err]
        ldout = open(pardict['project_folder'] \
            + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname +  '.pic', 'wb')
        pickle.dump(LDcoeffs, ldout)
        ldout.close()

    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    ldlist = pickle.load(ldd)[0]
    ldd.close()
    #flag = xobs < 3.
    #xobs, yobs, yobs_err = xobs[flag], yobs[flag], yobs_err[flag]
    sys.path.append('../KSint_wrapper/SRC/')
    import ksint_wrapper_fitcontrast

    # Add "white" light curve term
    # If spectrum computed with PandExo
    #floor = spec_obs['error_w_floor']

    #yobs_white = np.average(yobs, weights=1./(np.array(yobs_err)**2))
    #relsigma = np.concatenate((yobs, [yobs_white]))
    #relsigma_white = np.average(yobs, \
    #                weights=1./(yobs_err**2))#/(len(yobs)**0.5)
    #relsigma = np.concatenate((yobs_err, [relsigma_white]))
    #relsigma = yobs_err
    #yerr_white = np.mean(yobs_err)/len(yobs)**0.5
    '''
    if pardict['instrument'] == 'NIRSpec_Prism/':
        wlowblue, wupblue, uablue, ubblue \
                = np.loadtxt(pardict['ldfile_quadratic'], \
               unpack=True, skiprows=4)
        wlowred, wupred, uared, ubred = wlowblue, wupblue, uablue, ubblue
    if pardict['instrument'] == 'NIRCam/':
        wlowblue, wupblue, uablue, ubblue = \
                np.loadtxt(pardict['ldfile_quadratic_blue'], \
                unpack=True, skiprows=4)
        wlowred, wupred, uared, ubred = \
                np.loadtxt(pardict['ldfile_quadratic_red'], \
                unpack=True, skiprows=4)
    '''
    #if len(uablue) != len(yobs):
    #    print('# LD coefficients != # transmission spectrum points. Wrong LD file?')
        #set_trace()
    # Here, the last element of arange is the white light curve
    rise = []
    for i in np.arange(len(xobs)):
        # Compute transit with two spots
        fix_dict = {}
        fix_dict['prot'] = 11.0   # Hebrard+2012
        fix_dict['incl'] = 90.
        fix_dict['posang'] = 0.
        if pardict['tstar'] == 3500:
            fix_dict['lat'] = 28. # 12umbra
        elif pardict['tstar'] == 5000:
            fix_dict['lat'] = 12.
        fix_dict['latp'] = 12. # penumbra
        #fix_dict['rho'] = 1.7*1.408 #5000 K star ~ WASP-52
        #fix_dict['rho'] = 13.7*1.408 # 3500 K star ~ Kepler-1646
        # Density derived from logg
        fix_dict['rho'] = 5.14e-5/pardict['rstar']*10**pardict['loggstar']  #g/cm3
        fix_dict['P'] = pardict['pplanet']
        fix_dict['i'] = 88.
        fix_dict['e'] = 0.
        fix_dict['w'] = 180.
        fix_dict['omega'] = 0.
        fix_dict['nspots'] = 1

        # Random distribution for spots parameters
        params = np.zeros(8 + 3) # second spot is penumbra
        params[0] = yobs[i]**0.5 # kr
        params[1] = 252 # M, K
        #params[1] = 232. # F
        # LD coeffs
        '''
        if type(uared) == np.float64:
            uared = [uared]
            ubred = [ubred]
        if i < len(uablue):
            params[2], params[3] = uablue[i], ubblue[i]
        elif i - len(uablue) < len(uared):
            params[2], params[3] = uared[i - len(uablue)], \
                                                        ubred[i - len(uablue)]
        else:
            params[2], params[3] = 0., 0.
        '''
        #if i < len(xobs) - 2:
        #    wl_bin_low = xobs[i] - np.diff(xobs)[i]/2.
        #    wl_bin_up = xobs[i] + np.diff(xobs)[i + 1]/2.
        #else:
        #    wl_bin_low = xobs[i] - np.diff(xobs)[i - 1]/2.
        #    wl_bin_up = xobs[i] + np.diff(xobs)[i - 1]/2.

        #ww_, LDcoeffs = ld_coeffs.fit_law(pardict['tstar'], \
        #        pardict['loggstar'], 0.0, thrfile=None, grid='phoenix', \
        #        wlmin=wl_bin_low, wlmax=wl_bin_up, nchannels=2, plots=False)
        params[2], params[3] = ldlist[i][0], ldlist[i][1]
        #params[2], params[3] = 0.3, 0.3
        #params[2], params[3] = ua[i], ub[i]
        #params[2], params[3] = 0.2, 0.2
        # Long, size
        #params[4], params[5] = 300., 5.0
        #params[4], params[5] = 270, 3. # K ??
        if pardict['tstar'] == 3500 or pardict['tstar'] == 5000:
            params[4], params[5] = 260, 3.  # M
        #elif pardict['tstar'] == 5000:
        #    params[4], params[5] = 250, 3. # K ??
        # Contrast
        modstar = pysynphot.Icat(models[0], pardict['tstar'], 0.0, \
                            pardict['loggstar'])
        wl = modstar.wave
        starm = modstar.flux
        umbram =  pysynphot.Icat(models[0], pardict['tumbra'], 0.0, \
                            pardict['loggstar'] - 0.5).flux
        penumbram = pysynphot.Icat(models[0], pardict['tpenumbra'], 0.0, \
                            pardict['loggstar'] - 0.5).flux

        # Throughput on models
        if pardict['instrument'] == 'NIRCam/':
            wth1, fth1 = np.loadtxt(thrfile1, unpack=True, skiprows=2)
            wth2, fth2 = np.loadtxt(thrfile2, unpack=True, skiprows=2)
            wth3, fth3 = np.loadtxt(thrfile3, unpack=True, skiprows=2)
            wth = np.concatenate((wth1, wth2, wth3))
            fth = np.concatenate((fth1, fth2, fth3))
            #plt.plot(wl, starm)
            #plt.plot(wl, umbram)
            starm = integ_filter(wth, fth, wl, starm)
            umbram = integ_filter(wth, fth, wl, umbram)
            #plt.plot(wl, starm)
            #plt.plot(wl, umbram)
        #wl = fits.open(modelsfolder \
        #            +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
        #starm = fits.open(modelsfolder + 'lte0' + str(int(pardict['tstar'])) \
        #            + '-' + '{:3.2f}'.format(pardict['loggstar']) \
        #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        #umbram = fits.open(modelsfolder + 'lte0' + str(int(pardict['tumbra'])) \
        #            + '-' + '{:3.2f}'.format(pardict['loggstar'] - 0.5) \
        #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        #penumbram = fits.open(modelsfolder + 'lte0' + str(int(pardict['tpenumbra'])) \
        #            + '-' + '{:3.2f}'.format(pardict['loggstar'] - 0.5) \
        #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        contrast = umbram/starm
        contrastp = penumbram/starm
        #plt.plot(wl, contrast)
        #plt.show()
        #set_trace()
        if i == 0:
            chanleft = xobs[i] ##wl.min()*1e-4
        else:
            chanleft = (xobs[i] - 0.5*(xobs[i] - xobs[i - 1])) #* u.micrometer
        if i == len(xobs) - 1:
            chanright = xobs[i]#*1e4
        else:
            chanright = (xobs[i] + 0.5*(xobs[i + 1] - xobs[i])) #* u.micrometer
        wlcenter = np.mean([chanleft, chanright]) #* u.micrometer
        wlbin = np.logical_and(wl*1e-4 >= chanleft, wl*1e-4 <= chanright)

        params[6] = 1. - np.mean(contrast[wlbin]) # Contrast spot 1
        rise.append(params[6])
        # Some numberical problems in the models?
        #if params[6] < 0.:
        #    set_trace()
        #    params[6] = 1e-6
        # Spot 2
        params[7] = params[4] - 2
        params[8] = 12.#params[5]*2.4 # see reference in notes
        params[9] = 1. - np.mean(contrastp[wlbin])
        print('Channel', str(i), ', Kr =', np.round(params[0], 3), \
                        'Contrast umbra =', np.round(params[6], 3))#, \
                        #'Contrast penumbra =', np.round(params[9], 3))
        tt = np.arange(0., 0.2, 120.*1./86400.)  # M, K
        #tt = np.arange(0., 0.4, 60.*1./86400.) # F star
        transit = ksint_wrapper_fitcontrast.main(params, tt, fix_dict)
        # White noise
        unckr = 1./(2.*params[0])*yobs_err[i]
        transit *= np.random.normal(loc=1., scale=unckr*2**0.5, \
                                    size=len(tt))
        yerr = np.zeros(len(transit)) + unckr*2**0.5
        #transit *= np.random.normal(loc=1., scale=yobs_err[i]*len(tt)**0.5, \
        #                            size=len(tt))
        #yerr = np.zeros(len(transit)) + np.mean(yobs_err[i])*len(tt)**0.5
        #tt, transit, yerr = tt[flag], transit[flag], yerr[flag]
        # Delta x^2 = 2 x Delta x
        #relsigma_i = np.zeros(len(tt)) + relsigma[i]/(2.*(yobs[i]**0.5))
        plt.close('all')
        #plt.plot(tt, transit - transit[0])
        plt.errorbar(tt, transit, yerr=yerr, fmt='k.')#, capsize=2)
        #plt.show()

        '''
        flagv = np.logical_and(tt > 0.07491, tt < 0.084)
        flago = np.logical_and(tt > 0.07264, tt < 0.08986)
        plt.errorbar(tt[flago], transit[flago], yerr=relsigma[i], \
                    fmt='.', color='orange', label='Penumbra')
        plt.errorbar(tt[flagv], transit[flagv], yerr=relsigma[i], \
                            fmt='g.', label='Umbra')
        '''
        plt.xlabel('Time [d]', fontsize=16)
        plt.ylabel('Relative flux', fontsize=16)
        plt.title('Channel: ' + str(round(xobs[i], 3)) + ' $\mu$m', fontsize=16)
        #plt.legend(loc='best', fontsize=16, frameon=False)
        plt.savefig(pardict['data_folder'] + 'transit_spots' + str(i) \
                        + '_' + instrument + '.pdf')
        savefile = open(pardict['data_folder'] + 'transit_spots' \
                        + '_' + str(i) + '.pic', 'wb')
        pickle.dump([tt, transit, yerr, xobs[i]], savefile)
        savefile.close()

    # Remove pandexo plot files
    #os.system('rm *html')
    #plt.figure()
    #plt.plot(xobs, np.array(rise)/min(rise), 'o')
    #plt.show()

    return len(xobs) #i#tt, transit

def channelmerge(pardict, ch1, ch2):
    '''
    Concatenates two transits.
    '''
    f1 = pardict['data_folder'] + 'transit_spots' + str(ch1) + '.pic'
    f2 = pardict['data_folder'] + 'transit_spots' + str(ch2) + '.pic'
    t1, y1, err1 = pickle.load(open(f1, 'rb'))
    t2, y2, err2 = pickle.load(open(f2, 'rb'))
    t = np.concatenate((t1, t2))
    y = np.concatenate((y1, y2))
    err = np.concatenate((err1, err2))

    fileout = open(pardict['data_folder'] + 'channel' + str(ch1) + '+' \
                    + str(ch2) + '.pic', 'wb')
    pickle.dump([[t1, t2], [y1, y2], [err1, err2]], fileout)

    return

def air_to_vacuum(wl):
    '''
    From https://ui.adsabs.harvard.edu/abs/1991ApJS...77..119M/abstract.
    '''
    ratio = 6.4328e-5 + 2.94981e-2/(146. - 1e4/wl) + 2.5540e-4/(41 - 1e4/wl)
    wl_vac = ratio*wl + wl

    return wl_vac

def integ_filter(wth, fth, wmodel, fmodel):

    fth_int = interp1d(wth, fth, bounds_error=False, fill_value=0.)
    fthgrid = fth_int(wmodel)
    fthgrid[fthgrid < 1e-6] = 0.
    #fintegrated = np.trapz(fmodel*fthgrid, wmodel)
    fintegrated = fmodel*fthgrid
    return fintegrated
