# Generate transmission spectrum with PandExo and simulate transits with
# spot occultations.

import sys
sys.path.append('/home/giovanni/Downloads/Shelf/python/pandexo/')
import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import pandexo.engine.justplotit as jpi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from transit_fit import transit_model, gauss
from scipy.optimize import minimize
import numpy as np
import pysynphot
import pathlib
import rebin_spectra
import pickle
import os
from astropy.io import fits
homedir = os.path.expanduser('~')
sys.path.append(homedir + '/Projects/git_code/Light-curve-tools/')
import ld_coeffs
from astropy.convolution import convolve
from glob import glob
from pdb import set_trace

modelsfolder = homedir + '/Downloads/Shelf/stellar_models/phxinten/HiRes/'
foldthrough = homedir + '/Downloads/Shelf/filters/'
thrfile1 = foldthrough + 'JWST_NIRCam.F150W2.dat'
thrfile2 = foldthrough + 'JWST_NIRCam.F322W2.dat'
thrfile3 = foldthrough + 'JWST_NIRCam.F444W.dat'
thrfile4 = foldthrough + 'JWST_NIRSpec.CLEAR.dat'

def generate_spectrum_jwst(pardict, models):

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
    exo_dict['star']['mag'] = pardict['magstar']     #magnitude of the system
    exo_dict['star']['ref_wave'] = 2.22   #For J mag = 1.25, H = 1.6, K =2.22..
                                          #etc (all in micron)

    if not pardict['spotted_starmodel']:
        exo_dict['star']['type'] = 'phoenix'
        exo_dict['star']['temp'] = pardict['tstar']     #in K
        exo_dict['star']['metal'] = 0.0        # as log Fe/H
        exo_dict['star']['logg'] = pardict['loggstar']
    else:
        exo_dict['star']['type'] = 'user'
        generate_spotted_spectra(pardict, models=models)
        exo_dict['star']['starpath'] = pardict['data_folder'] + 'spotted_star.dat'
        exo_dict['star']['w_unit'] = 'Angs'
        exo_dict['star']['f_unit'] = 'FLAM'

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

    inst_dict = jdi.load_mode_dict('NIRSpec Prism')
    inst_dict["configuration"]["detector"]["ngroup"] = 'optimize'
    #   inst_dict["configuration"]["detector"]["subarray"] = 'sub512'
    #   inst_dict["configuration"]["detector"]["readmode"] = 'nrs'
    print('Starting PandExo run for JWST')
    jdi.run_pandexo(exo_dict, inst_dict, save_file=True, \
            output_file=pardict['pandexo_out_jwst'])

    #jdi.run_pandexo(exo_dict, inst_dict, save_file=True, \
    #        output_file=pardict['pandexo_out_jwst'])
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

def read_pandexo_results(pardict, res=4):
    '''
    Used for both HST and JWST simulations.

    res: resolution for JWST observations.
    '''

    if pardict['observatory'] == 'jwst':
        spec_obs = pickle.load(open(pardict['pandexo_out_jwst'], 'rb'))
        xobs, yobs, yobs_err = jpi.jwst_1d_spec(spec_obs, R=res, \
                    num_tran=1, model=False, plot=False)
        xobs = np.array(xobs)[0]
        yobs = np.array(yobs)[0]
        yobs_err = np.array(yobs_err)[0]
    elif pardict['observatory'] == 'hst':
        spec_obs = pickle.load(open(pardict['pandexo_out'], 'rb'))
        xobs, yobs, yobs_err, modelwave, modelspec \
                        = jpi.hst_spec(spec_obs, plot=False)

    plt.errorbar(xobs, yobs, yerr=yobs_err, fmt='o')
    plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
    plt.ylabel('Transit depth', fontsize=16)
    plt.title('Transmission spectrum output from PandExo', fontsize=16)
    #plt.show()
    plt.savefig(pardict['data_folder'] + 'spec_model_' \
                        + pardict['observatory'] + '.pdf')
    plt.close('all')

    return xobs, yobs, yobs_err

def add_spots(pardict, resol=10, simultr=None, models='phoenix', \
            noscatter=False):
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
                    = read_pandexo_results(pardict, res=resol)
        #xobs -= 0.5
        flag = xobs < 6.5
        xobs, yobs, yobs_err = xobs[flag], yobs[flag], yobs_err[flag]
        #yobs_err /= 2.
        # Rebin from resolution 100 to res
        #xobs = rebin.rebin_wave(xobs_, 70)
        #yobs, yobs_err = rebin.rebin_spectrum(yobs_, xobs_, xobs, unc=yobs_err_)
    elif type(simultr) == 'str' and rsimultr.endswith('+'):
        xobs1, yobs1, yobs_err1 = np.loadtxt(uncert_file[:-1], unpack=True)
        xobs2, yobs2, yobs_err2 = read_pandexo_results(pardict, res=resol)
        xobs = np.concatenate((xobs1, xobs2))
        yobs = np.concatenate((yobs1, yobs2))
        yobs_err = np.concatenate((yobs_err1, yobs_err2))
    elif type(simultr) == 'str' and not simultr.endswith('+'):
        xobs, yobs, yobs_err = np.loadtxt(uncert_file, unpack=True)
    elif len(simultr) == 3:
        # First and last one must have some rebinning problem
        xobs, yobs, yobs_err = simultr[0], simultr[1], simultr[2]

    # Rescale kr with new planet/star radius ratio
    if pardict['instrument'] == 'NIRCam' and int(pardict['tstar']) == 3500:
        yobs = yobs - np.mean(yobs) \
                        + (pardict['rplanet']*0.1005/pardict['rstar'])**2
    if pardict['spotted_starmodel'] and pardict['instrument'] == 'NIRCam':
        yobs = spotted_transmsp(xobs, yobs, pardictmodels=models)

    if noscatter:
        yobs = np.zeros(len(xobs)) + np.median(yobs)#0.01
    # Add one point for the band-integrated transit
    weights = 1./yobs_err**2
    yobs = np.concatenate((yobs, [np.average(yobs, weights=weights)]))
    yobs_err = np.concatenate((yobs_err, [np.sum(weights)**-0.5]))
    xobs = np.concatenate((xobs, [np.mean(xobs)]))

    # Save transmission spectrum
    savespec = open(pardict['data_folder'] + 'spec_model_' \
                + pardict['observatory'] + '.pic', 'wb')
    pickle.dump([xobs, yobs, yobs_err], savespec)
    savespec.close()

    # Calculate LD coefficients
    LDcoeffs = [[], []]
    if pardict['instrument'] == 'NIRSpec_Prism':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam':
        ldendname = 'nircam'

    if (pardict['magstar'] == 4.5 or pardict['magstar'] == 10.5) \
        and not pathlib.Path(pardict['project_folder'] \
                + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
                            + 'K/' + 'LDcoeffs_' + ldendname + '.pic').exists():
        print('Computing LD coefficients...')
        for i in np.arange(len(xobs)):
            if i < len(xobs) - 2:
                wl_bin_low = xobs[i] - np.diff(xobs)[i]/2.
                wl_bin_up = xobs[i] + np.diff(xobs)[i + 1]/2.
                if wl_bin_low > wl_bin_up:
                    wl_bin_up = xobs[i] + np.diff(xobs)[i]/2.
            elif i == len(xobs) - 2:
                wl_bin_low = xobs[i] - np.diff(xobs)[i - 1]/2.
                wl_bin_up = xobs[i] + np.diff(xobs)[i - 1]/2.
            else:
                wl_bin_low = xobs[0]
                wl_bin_up = xobs[-2]

            if ldendname == 'prism':
                thrfile = foldthrough + 'JWST_NIRSpec.CLEAR.dat'
            elif ldendname == 'nircam' and xobs[i] < 2.4:
                thrfile = foldthrough + 'JWST_NIRCam.F150W2.dat'
            elif ldendname == 'nircam' and xobs[i] >= 2.4:
                thrfile = foldthrough + 'JWST_NIRCam.F322W2.dat'
            else:
                # Combine both filters
                thrfile = [foldthrough + 'JWST_NIRCam.F150W2.dat', \
                    foldthrough + 'JWST_NIRCam.F322W2.dat']
            ww_, LDcoeffs_ = ld_coeffs.fit_law(pardict['tstar'], \
                pardict['loggstar'], 0.0, thrfile=thrfile, grid='custom', \
                wlmin=wl_bin_low, wlmax=wl_bin_up, nchannels=1, plots=False, \
                dict_models=pardict['starmodel'])
            LDcoeffs[0].append(LDcoeffs_)

        LDcoeffs[1] = [xobs, yobs, yobs_err]
        ldout = open(pardict['project_folder'] \
            + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname +  '.pic', 'wb')
        pickle.dump(LDcoeffs, ldout)
        ldout.close()

    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    ldlist = pickle.load(ldd)[0]
    ldd.close()

    sys.path.append('../KSint_wrapper/SRC/')
    import ksint_wrapper_fitcontrast

    # Here, the last element of arange is the white light curve
    rise = []
    # Save contrast spectrum
    fout_ = open(pardict['data_folder'] + 'contrast_spectrum.pic', 'wb')
    for i in np.arange(len(xobs)):
        # Compute transit with two spots
        fix_dict = {}
        fix_dict['prot'] = 11.0   # Hebrard+2012
        fix_dict['incl'] = 90.
        fix_dict['posang'] = 0.
        #if pardict['tstar'] == 3500:
        #    fix_dict['lat'] = 28. # 1at umbra
        #elif pardict['tstar'] == 5000:
        #    fix_dict['lat'] = 12.
        fix_dict['lat'] = pardict['latspot']
        fix_dict['latp'] = 12. # penumbra
        # Density derived from logg
        fix_dict['rho'] = 5.14e-5/pardict['rstar']*10**pardict['loggstar']  #g/cm3
        fix_dict['P'] = pardict['pplanet']
        fix_dict['i'] = pardict['incl']
        fix_dict['e'] = 0.
        fix_dict['w'] = 180.
        fix_dict['omega'] = 0.
        fix_dict['nspots'] = 1
        #set_trace()
        #aR = (6.67408e-11*fix_dict['rho']*1e3*(fix_dict['P']*86400.)**2 \
        #                /(3.*np.pi))**(1./3.)
        # Random distribution for spots parameters
        params = np.zeros(8 + 3) # second spot is penumbra
        params[0] = yobs[i]**0.5 # kr
        params[1] = 252 # M, K

        params[2], params[3] = ldlist[i][0], ldlist[i][1]
        if pardict['tstar'] == 3500 or pardict['tstar'] == 5000:
            # Find spot longitude corresponding to given mu
            longspot = 266. - pardict['theta']
            params[4], params[5] = longspot, pardict['aumbra']  # M
        if not pardict['spotted_starmodel']:
            if models == 'phoenix':
                modstar = pysynphot.Icat(models, pardict['tstar'], 0.0, \
                                pardict['loggstar'])
                wave = modstar.wave
                starm = modstar.flux
            elif models == 'josh':
                # Whenever you compute occultations, you should use this and
                # not phoenix (dependency on mu angle)
                starm = pardict['starmodel']['spec'][pardict['muindex']]
                wave = pardict['starmodel']['wl']
                #tspot_ = format(pardict['tumbra'], '2.2e')
                #spot = pardict['spotmodels'][tspot_]['spec'][pardict['muindex']]
        else:
            if pardict['instrument'] == 'NIRCam':
                generate_spotted_spectra(pardict, models='phoenix')
            wave, starm = np.loadtxt(pardict['data_folder'] \
                + 'spotted_star.dat', unpack=True)
        if models == 'phoenix':
            wlumbra = pysynphot.Icat(models, pardict['tumbra'], 0.0, \
                                pardict['loggstar'] - 0.5).wave
            umbram = pysynphot.Icat(models, pardict['tumbra'], 0.0, \
                                pardict['loggstar'] - 0.5).flux
            #penumbram = pysynphot.Icat(models, pardict['tpenumbra'], 0.0, \
            #                    pardict['loggstar'] - 0.5).flux
        elif models == 'josh':
            tspot_ = format(pardict['tumbra'], '2.2e')
            wlumbra = pardict['spotmodels'][tspot_]['wl']
            umbram = pardict['spotmodels'][tspot_]['spec'][pardict['muindex']]
            #penumbram = np.copy(umbram)
        wave = np.array(wave)

        # Throughput on models
        if pardict['instrument'] == 'NIRCam':
            wth1, fth1 = np.loadtxt(thrfile1, unpack=True)
            wth2, fth2 = np.loadtxt(thrfile2, unpack=True)
            wth3, fth3 = np.loadtxt(thrfile3, unpack=True)
            wth = np.concatenate((wth1, wth2, wth3))
            fth = np.concatenate((fth1, fth2, fth3))
        elif pardict['instrument'] == 'NIRSpec_Prism':
            wth, fth = np.loadtxt(thrfile4, unpack=True)
            wth*= 1e4

        starm = integ_filter(wth, fth, wave, starm)
        umbram = integ_filter(wth, fth, wlumbra, umbram)
        if i != len(xobs) - 1:
            starm = degrade_spec(starm, wth, xobs[:-1])
            umbram = degrade_spec(umbram, wth, xobs[:-1])
        else:
            starm = np.zeros(len(starm)) + np.mean(starm)
            umbram = np.zeros(len(umbram)) + np.mean(umbram)
        contrast = umbram/starm
        #contrastp = penumbram/starm
        params[6] = 1. - contrast[i] # Contrast
        rise.append(params[6])
        # Spot 2
        params[7] = params[4] - 2
        params[8] = 12.#params[5]*2.4 # see reference in notes
        params[9] = 1. #- np.mean(contrastp[wlbin])
        print('Bin #', str(i), ', Kr =', np.round(params[0], 3), \
                        'Contrast umbra =', np.round(params[6], 3))#, \
                        #'Contrast penumbra =', np.round(params[9], 3))
        #tt = np.arange(0., 0.2, 120.*1./86400.)  # M, K
        #tt = np.arange(0., 0.4, 60.*1./86400.) # F star
        # Define in-transit/out-of-transit duration
        # Use 90-s time resolution and then interpolate, otherwise there is some
        # numerical error
        if pardict['tstar'] == 3500:
            tt_ = np.arange(0.075 - 0.04, 0.125 + 0.04, 90./86400.)
            tt = np.arange(0.075 - 0.04, 0.125 + 0.04, 60./86400.)
        elif pardict['tstar'] == 5000:
            tt_ = np.arange(0.05 - 0.1, 0.150 + 0.1, 90./86400.)
            tt = np.arange(0.05 - 0.1, 0.150 + 0.1, 60./86400.)
        if i == 0:
            print('Data points in transit:', len(tt))
        transit_ = ksint_wrapper_fitcontrast.main(params, tt_, fix_dict)
        fint = interp1d(tt_, transit_, bounds_error=False, \
                    fill_value='extrapolate')
        transit = fint(tt)
        # Subtract planet-only contribution and add pytransit model
        #fix_dict['nspots'] = 0
        # From stellar density
        rhostar = 5.14e-5/pardict['rstar']*10**pardict['loggstar'] #cgs
        aR = (6.67408e-11*rhostar*1e3*(pardict['pplanet']*86400.)**2 \
                            /(3.*np.pi))**(1./3.)
        # White noise
        uncph = yobs_err[i]*len(tt)**0.5/2.
        if not noscatter:
            transit *= np.random.normal(loc=1., scale=uncph, size=len(tt))
        yerr = np.zeros(len(transit)) + uncph

        #plt.close('all')
        #plt.errorbar(tt, transit, yerr=yerr, fmt='k.')#, capsize=2)
        ##plt.plot(tt_, modspot(soln.x, tt_))
        #plt.xlabel('Time [d]', fontsize=16)
        #plt.ylabel('Relative flux', fontsize=16)
        #plt.title('Channel: ' + str(round(xobs[i], 3)) + ' $\mu$m', fontsize=16)
        if i == len(xobs) - 1:
            i = -1
        #plt.show()
        #plt.savefig(pardict['data_folder'] + 'transit_spots' + str(i) \
        #                + '_' + pardict['observatory'] + '.pdf')
        savefile = open(pardict['data_folder'] + 'transit_spots' \
                        + '_' + str(i) + '.pic', 'wb')
        pickle.dump([tt, transit, yerr, xobs[i]], savefile)
        #pickle.dump([tt_, transit3, yerr, xobs[i]], savefile)
        savefile.close()

    pickle.dump([xobs, rise], fout_)
    fout_.close()
    # Save fix_dict
    fout2 = open(pardict['data_folder'] + 'KSint_pars.pic', 'wb')
    pickle.dump(fix_dict, fout2)
    fout2.close()

    return len(xobs)

def modspot(par, t):

    Aspot, x0, sig, n, r0, r1, r2 = par
    return gauss(t, [Aspot, x0, sig, n]) * np.polyval([r0, r1, r2], t)

def lnlike(par, t, y, yerr):

    sigma = np.mean(yerr)
    lnL = -0.5*len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) \
                - 0.5*chi2(modspot(par, t), y, yerr)
    return lnL

def chi2(mod, y, yerr):
    return np.sum((y - mod)**2/(yerr**2))

def integ_filter(wth, fth, wmodel, fmodel, integrate='model'):

    if integrate == 'filter':
        fth_int = interp1d(wth, fth, bounds_error=False, fill_value=0.)
        fthgrid = fth_int(wmodel)
        fthgrid[fthgrid < 1e-6] = 0.
        fintegrated = fmodel*fthgrid
    elif integrate == 'model':
        modint = interp1d(wmodel, fmodel, bounds_error=False, fill_value=0.)
        modgrid = modint(wth)
        fth[fth < 1e-6] = 0.
        try:
            fintegrated = modgrid*fth
        except ValueError:
            set_trace()
    #fintegrated = np.trapz(fmodel*fthgrid, wmodel)

    return fintegrated

def generate_spotted_spectra(pardict, models='phoenix'):
    '''
    Generate spotted stellar spectrum.

    Parameters
    ----------
    delta: starspot filling factor.
    muindex: the index corresponding to the input mu angle (for Josh's models)
    '''

    loggspot = pardict['loggstar'] - 0.5
    delta = np.radians(pardict['aumbra'])**2

    if models == 'phoenix':
        mm = pysynphot.Icat('phoenix', pardict['tstar'], 0.0, \
                                    pardict['loggstar'])
        star = mm.flux
        wl = mm.wave
        spot = pysynphot.Icat('phoenix', pardict['tumbra'], 0.0, \
                                    loggspot).flux
    elif models == 'josh':
        star = pardict['starmodel']['spec'][pardict['muindex']]
        wl = pardict['starmodel']['wl']
        tspot_ = format(pardict['tumbra'], '2.2e')
        spot = pardict['spotmodels'][tspot_]['spec'][pardict['muindex']]

    newstar = star*(1. - delta) + delta*spot
    fout = open(pardict['data_folder'] + 'spotted_star.dat', 'w')
    np.savetxt(fout, np.transpose(np.array([wl, newstar])))

    return

def spotted_transmsp(wtransnm, ytransm, pardict, models='phoenix'):
    '''
    Apply starspot effect to a transmission spectrum. See McCullough+2014, eq. 1.

    Parameters
    ----------
    wtransnm, ytransm: wl and transit depth of transmission spectrum.
    '''
    if models == 'phoenix':
        mm = pysynphot.Icat('phoenix', pardict['tstar'], 0.0, \
                            pardict['loggstar'])
        star = mm.flux
        wave = mm.wave
        spot = pysynphot.Icat('phoenix', pardict['tumbra'], 0.0, \
                            pardict['loggstar'] - 0.5).flux
    elif models == 'josh':
        star = pardict['starmodel']['spec'][pardict['muindex']]
        wave = pardict['starmodel']['wl']
        tspot_ = format(pardict['tumbra'], '2.2e')
        wlspot = pardict['spotmodels'][tspot_]['wl']
        spot = pardict['spotmodels'][tspot_]['spec'][pardict['muindex']]
    if pardict['instrument'] == 'NIRSpec_Prism':
        wth, fth = np.loadtxt(thrfile4, unpack=True)
        wth*= 1e4
    elif pardict['instrument'] == 'NIRCam':
        wth1, fth1 = np.loadtxt(thrfile1, unpack=True)#, skiprows=2)
        wth2, fth2 = np.loadtxt(thrfile2, unpack=True)#, skiprows=2)
        wth3, fth3 = np.loadtxt(thrfile3, unpack=True)#, skiprows=2)
        wth = np.concatenate((wth1, wth2, wth3))
        fth = np.concatenate((fth1, fth2, fth3))
    # Apply filter throughput and degrade resolution
    star = integ_filter(wth, fth, wave, star)
    spot = integ_filter(wth, fth, wlspot, spot)
    star = degrade_spec(star, wave, wtransnm)
    spot = degrade_spec(spot, wlspot, wtransnm)

    delta = np.radians(pardict['aumbra'])**2
    newtransm = ytransm/(1. - delta*(1. - spot/star))

    return newtransm

def degrade_spec(spec, oldwl, newwl):
    '''
    Use provided new wavelength to degrade spectrum.
    '''

    newspec = np.zeros(len(newwl))
    dwl = np.diff(newwl)
    for j, wli in enumerate(newwl):
        #if j == 0:
        #    wl1, wl2 = wli - dwl[0], wli + dwl[0]
        if j == len(newwl) - 1:
            wl1, wl2 = wli - 0.5*dwl[-1], wli + 0.5*dwl[-1]
        else:
            wl1, wl2 = wli - 0.5*dwl[j], wli + 0.5*dwl[j]
        flag = np.logical_and(oldwl/1e4 >= wl1, oldwl/1e4 < wl2)
        #flag = np.logical_and(oldwl >= wl1, oldwl < wl2)
        newspec[j] = np.mean(spec[flag])

    return newspec

def plot_transmission_spectra():
    '''
    Plot the uncertanties obtained on the transmission spectra.
    '''
    instrument = ['NIRCam', 'NIRSpec_Prism']
    stars = ['3500K', '5000K']
    root = '/home/giovanni/Projects/jwst_spots/revision2/'
    for pnum, instr in enumerate(instrument):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        if instr == 'NIRCam':
            labels = [4.5, 6.0, 7.5, 9.0]
        else:
            labels = [10.5, 11.5, 12.5, 13.5, 14.5]
        for pcount, star in enumerate(stars):
            if star == '3500K':
                ff = glob(root + instr + '/star_' + star \
                        + '/*2600*theta0*noscatter*/simulated_data/' \
                                + 'spec_model_jwst.pic')
            else:
                ff = glob(root + instr + '/star_' + star \
                        + '/*3800*theta0*noscatter*/simulated_data/' \
                                + 'spec_model_jwst.pic')
            for fj, fspec in enumerate(ff):
                print(fspec.split('/simulated_data')[0])
                mag = fspec.split('mag')[1].split('_')[0]
                spec = pickle.load(open(fspec, 'rb'))
                wl = spec[0][:-1]
                yerr = spec[2][:-1]*1e6
                if instr == 'NIRCam':
                    flag = wl < 4.02
                    wl = wl[flag]
                    yerr = yerr[flag]
                axs[pcount].errorbar(wl, np.zeros(len(wl)), \
                 yerr=yerr, fmt='o-', capsize=2, mfc='white', mec='k')
                #if fj == 0:
                #    for ll in labels:
                #        axs[pcount].errorbar([], [], yerr=[], label=ll, capsize=2)
            axs[pcount].set_title(star.replace('K', '') + ' K star - ' \
                    + instr.replace('_', ' '), fontsize=16)
            if pcount == 0:
                axs[pcount].set_ylabel('Transit depth uncertainty [ppm]', \
                            fontsize=14)
            axs[pcount].set_xlabel(r'Wavelength [$\mu$m]', fontsize=14)
            #axs[pcount].legend(title='K mag')
        plt.show()
        plt.savefig(root + instr + '/' + instr + '_transmission_spec_unc.pdf')

    return
