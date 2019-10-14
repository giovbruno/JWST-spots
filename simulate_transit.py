import sys
sys.path.append('/home/giovanni/archive/python/pandexo/')
import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import pandexo.engine.justplotit as jpi
import numpy as np
import os, sys, pickle
from astropy.io import fits
import matplotlib.pyplot as plt
from pdb import set_trace
from text_operations import printlog
import blackbody as bb
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.analytic_functions import blackbody_lambda, blackbody_nu
from astropy.utils.exceptions import AstropyDeprecationWarning
import pysynphot
#modfolder = '/home/giovanni/archive/Stellar_models/phxinten/HiRes/'

def generate_spectrum(pardict):

    print('TO ADD: 3D STELLAR MODELS FOR FGK STARS (NOT M)')

    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = 80  #saturation level in percent of full well
    exo_dict['observation']['sat_unit'] = '%'
    exo_dict['observation']['noccultations'] = 2   #number of transits
    exo_dict['observation']['R'] = 20    #fixed binning. I usually suggest ZERO binning.. you can always bin lat34 without having to redo the calcualtion
    exo_dict['observation']['baseline'] = 1.0    #fraction of time in transit versus out = in/out
    exo_dict['observation']['baseline_unit'] = 'frac'
    exo_dict['observation']['noise_floor'] = 20   #this can be a fixed level or it can be a filepath
    exo_dict['star']['type'] = 'phoenix'  #phoenix or user (if you have your own)
    exo_dict['star']['mag'] = pardict['magstar']     #magnitude of the system
    exo_dict['star']['ref_wave'] = 1.25   #For J mag = 1.25, H = 1.6, K =2.22.. etc (all in micron)
    exo_dict['star']['temp'] = pardict['tstar']     #in K
    exo_dict['star']['metal'] = 0.0        # as log Fe/H
    exo_dict['star']['logg'] = pardict['loggstar']
    exo_dict['star']['radius'] = pardict['rstar']
    exo_dict['star']['r_unit'] = 'R_sun'

    #exo_dict['planet']['type'] = 'constant'
    exo_dict['planet']['exopath'] = pardict['data_folder'] + 'exotransmit.dat'
    exo_dict['planet']['w_unit'] = 'um'
    exo_dict['planet']['radius'] = pardict['rplanet']    #other options include "um","nm" ,"Angs", "secs" (for phase curves)
    exo_dict['planet']['r_unit'] = 'R_jup'
    exo_dict['planet']['transit_duration'] = 2.0
    exo_dict['planet']['td_unit'] = 'h'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'

    #inst_dict = jdi.load_mode_dict('NIRSpec Prism')
    #inst_dict["configuration"]["detector"]["subarray"] = 'sub512'
    #inst_dict["configuration"]["detector"]["readmode"] = 'nrs'

    print('Starting TEST run')
    #jdi.run_pandexo(exo_dict, ['NIRCam F444W'], save_file=True)
    #jdi.run_pandexo(exo_dict, ['NIRSpec G140M'], save_file=True)
    #jdi.run_pandexo(exo_dict, ['NIRSpec G235H'], save_file=True)
    jdi.run_pandexo(exo_dict, ['NIRSpec Prism'], save_file=True, \
                    output_file=pardict['pandexo_out'])
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
    set_trace()
    return relsigma

def add_spots(pardict, res):

    # To implement
    print('TO ADD: TRANSIT SYSTEMATICS, NEW LD COEFFICIENTS FROM 3D \
                STELLAR MODELS?') # Nope
    print('TO ADD: UNCERTAINTIES ON LD COEFFICIENTS, FROM REASONABLE UNC. \
                ON STELLAR PARAMETERS') # Meh
    print('TO ADD: GRANULATION') # To be considered as starspots, especially
                                 # for giant stars? Not this time

    sys.path.append('KSint_wrapper/SRC/')
    import ksint_wrapper_fitcontrast

    spec_obs = pickle.load(open(pardict['pandexo_out'], 'rb'))
    #xobs, xobs_err, yobs, yobs_err = jpi.jwst_1d_spec(spec_obs, R=res, \
    #            num_tran=10, model=False, plot=False)#, x_range=[.8,1.28])
    xobs, yobs, xobs_err, yobs_err = spec_obs
    xobs = np.array(xobs)#[0]
    xobs_err = np.array(xobs_err)
    yobs = np.array(yobs)#[0]
    yobs_err = np.array(yobs_err)#[0]

    plt.errorbar(xobs, yobs, yerr=yobs_err, xerr=xobs_err, fmt = 'o')
    plt.show()
    set_trace()
    # Add "white" light curve term
    # If spectrum computed with PandExo
    #floor = spec_obs['error_w_floor']

    yobs_white = np.average(yobs, weights=1./(np.array(yobs_err)**2))
    #relsigma = np.concatenate((yobs, [yobs_white]))
    relsigma_white = np.average(yobs, \
                    weights=1./(yobs_err**2))#/(len(yobs)**0.5)
    relsigma = np.concatenate((yobs_err, [relsigma_white]))
    yerr_white = np.mean(yobs_err)/len(yobs)**0.5
    wm, ua, eua, ub, eub = np.loadtxt(pardict['ldfile'], \
                    unpack=True, skiprows=3)
    plt.errorbar(xobs, yobs, yerr=yobs_err, fmt='o')
    #plt.show()
    plt.savefig(pardict['data_folder'] + 'spec_model.pdf')

    # Here, the last element of arange is the white light curve
    for i in np.arange(1, len(xobs) - 1):

        # Compute transit with two spots
        fix_dict = {}
        if i < 10:   # Need to implement LD coefficients here
            fix_dict['c1'] = 0.25 #+ np.random.normal(loc = 0., scale = 0.01)#ua[i]#ub[i]
            fix_dict['c2'] = 0.2 #+ np.random.normal(loc = 0.0, scale = 0.01)#ua[i]#ub[i]
        else:
            fix_dict['c1'] = 0. #+ np.random.normal(loc = 0., scale = 0.01)#ua[i]#ub[i]
            fix_dict['c2'] = 0. #+ np.random.normal(loc = 0.0, scale = 0.01)#ua[i]#ub[i]
        #print(fix_dict['c1'], fix_dict['c2'])
        fix_dict['prot'] = 11.8   # Hebrard+2012
        fix_dict['incl'] = 90.
        fix_dict['posang'] = 0.
        fix_dict['lat'] = 12. # umbra
        fix_dict['latp'] = 12. # penumbra
        fix_dict['rho'] = 1.7*1.408
        fix_dict['contrast1'] = 0.8
        fix_dict['contrast2'] = 0.5
        fix_dict['P'] = 2.0
        fix_dict['i'] = 88.
        fix_dict['e'] = 0.
        fix_dict['w'] = 180.
        fix_dict['omega'] = 0.
        fix_dict['nspots'] = 1
        # Random distribution for spots parameters
        params = np.zeros(8 + 3) # second spot is penumbra
        params[0] = yobs[i]**0.5
        #print('Rp/R* =', params[0])
        # LD coeffs
        if xobs[i] < 3.:
            params[2], params[3] = 0.25 - 0.003*i, 0.2 - 0.003*i
        if 0.2 - 0.003*i < 0.01:
            params[2], params[3] = 0.01, 0.01
        # Long, size
        params[4], params[5] = 300., 5.0 #240., 5.0
        params[8] = 12.#params[5]*2.4 # see reference in notes
        params[7] = params[4] - 2
        # Contrast
        #wlarr, contrast = bb.contrast_analytic(5000, 3800, 4000, 4800)
        #set_trace()
        #wlarr *= 1e6
        '''
        wl = np.arange(0.1, 10, 5e-3) * u.micrometer
        tstar = 5000 * u.K
        tspot = 4000 * u.K
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyDeprecationWarning)
                flux_star = blackbody_lambda(wl, tstar)
                flux_spot = blackbody_lambda(wl, tspot)
                contrast = flux_spot/flux_star

        wl = fits.open(modfolder \
                    + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data   #AA
        starm = fits.open(modfolder \
          + 'lte' + str(pardict['tstar']) + '-' + "%.2f" %pardict['loggstar'] \
          + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        spotm = fits.open(modfolder \
           + 'lte' + str(pardict['tspot']) + '-' + "%.2f" %pardict['loggstar'] \
           + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        penumbram = fits.open(modfolder \
           + 'lte' + str(pardict['tpenumbra']) + '-' + "%.2f" %pardict['loggstar'] \
           + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        '''
        wl = pysynphot.Icat('phoenix', pardict['tstar'], 0.0, \
                pardict['loggstar']).wave
        starm = pysynphot.Icat('phoenix', pardict['tstar'], 0.0, \
                pardict['loggstar']).flux
        umbram = pysynphot.Icat('phoenix', pardict['tumbra'], 0.0, \
                pardict['loggstar']).flux
        penumbram = pysynphot.Icat('phoenix', pardict['tpenumbra'], 0.0, \
                pardict['loggstar']).flux
        contrast = umbram/starm
        contrastp = penumbram/starm
        chanleft = (xobs[i] - 0.5*(xobs[i] - xobs[i - 1])) #* u.micrometer
        chanright = (xobs[i] + 0.5*(xobs[i + 1] - xobs[i])) #* u.micrometer
        wlcenter = np.mean([chanleft, chanright]) #* u.micrometer
        wlbin = np.logical_and(wl*1e-4 >= chanleft, wl*1e-4 <= chanright)
        params[6] = 1. - np.mean(contrast[wlbin])
        params[9] = 1. - np.mean(contrastp[wlbin])
        print(params[6], params[9])
        # Spot 2
        #params[5], params[6], params[7] = 0.1, 240., 4.0
        # M
        params[1] = 252.#270.

        tt = np.arange(0., 0.2, 60.*1./86400.)
        #flag = np.logical_and(tt > 1.9, tt < 2.1)
        transit = ksint_wrapper_fitcontrast.main(params, tt, fix_dict)#, fitcontrast = False)
        # White noise
        transit *= np.random.normal(loc=1., scale=relsigma[i], size=len(tt))
        yerr = np.zeros(len(transit)) + np.mean(relsigma[i])
        #tt, transit, yerr = tt[flag], transit[flag], yerr[flag]
        # Delta x^2 = 2 x Delta x
        relsigma_i = np.zeros(len(tt)) + relsigma[i]/(2.*(yobs[i]**0.5))
        print(i, chanleft, chanright, 1 - params[6], params[0])
        plt.close('all')
        plt.errorbar(tt, transit, yerr=relsigma[i], fmt='k.')#, capsize=2)
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
        plt.title('Channel: ' + str(round(xobs[i], 3)) + '$\mu$m', fontsize=16)
        plt.legend(loc='best', fontsize=16, frameon=False)
        plt.savefig(pardict['data_folder'] + 'transit_spots' + str(i) + '.pdf')
        savefile = open(pardict['data_folder'] + 'transit_spots' \
                        + str(i) + '.pic', 'wb')
        #plt.show()
        #set_trace()
        #pickle.dump([tt, transit, relsigma_i], savefile)
        pickle.dump([tt, transit, yerr, xobs[i]], savefile)
        savefile.close()

    return i#tt, transit

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
