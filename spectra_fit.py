# Compare spot fits with stellar models.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import medfilt
import get_uncertainties as getunc
import simulate_transit
import rebin
import pysynphot
import glob
import pickle
from pdb import set_trace
from astropy.io import fits

modelsfolder = '/home/giovanni/Dropbox/Shelf/stellar_models/phxinten/HiRes/'

plt.ioff()

def read_res(pardict, instrument, plotname, resfile, models, resol=10):
    '''
    Read in the chains files and initialize arrays with spot properties.

    wlwin: used to exlude parts of the spectra
    '''

    #ldfile = open(pardict['project_folder'] \
    #    + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
    #                    + 'K/' + 'LDcoeffs_prism.pic', 'rb')
    #ldd = pickle.load(ldfile)
    xobs, yobs, yobs_err \
            = simulate_transit.read_pandexo_results(pardict, instrument, \
                res=resol)
    #xobs -= 0.3
    flag = xobs < 6
    expchan = len(xobs[flag])
    #ldfile.close()

    wl, A, x0, sigma = [], [], [], []
    yerrup, yerrdown = [], []
    kr, krunc = [], []
    for i in np.arange(expchan):
        #ffopen = open(pardict['chains_folder'] + 'chains_' \
        #            + str(i) + '.pickle', 'rb')
        ffopen = open(pardict['chains_folder'] + 'sol_LM_' + str(i) \
                        + '.pic', 'rb')
        res = pickle.load(ffopen)
        #Achain = res['Chains'][:, -2]
        #Achain = Achain[Achain > 0.]
        #sigmachain = res['Chains'][:, -1]
        #sigmachain = sigmachain[sigmachain > 0.]
        #sigmachain = res['Starspot_size']
        # Recompute percentiles, excluding negative results for A
        #perc = [np.percentile(i,[4.55, 15.9, 50, 84.1, 95.45]) \
        #                    for i in [Achain, sigmachain]]
        #perc = res['Percentiles'][0]
        wl.append(res['wl'])
        #A.append(perc[-3]*1e6)
        #x0.append(perc[-2]*1e6)
        #sigma.append(perc[-1])
        #A.append(perc[0]*1e6)
        #sigma.append(perc[1])
        A.append(res['sol'].x[-2]*1e6)
        sigma.append(res['sol'].x[-1])
        yerrup.append(res['1sigma_unc'][-2]*1e6)
        yerrdown.append(res['1sigma_unc'][-2]*1e6)
        ffopen.close()
        kr.append(res['sol'].x[0])
        krunc.append(res['1sigma_unc'][0])

    # Compare with initial simulation
    plt.figure(42)
    kr = np.array(kr)
    krunc = np.array(krunc)
    #plt.errorbar(wl, kr**2, yerr=2.*kr*krunc, fmt='o', label='fit')
    #plt.errorbar(xobs, yobs, yerr=yobs_err, fmt='o', label='simulated')
    #plt.legend()
    #plt.savefig(pardict['chains_folder'] + 'compare_kr.pdf')
    #plt.close('all')
    # Group stuff together
    #A = np.vstack(A)
    #x0 = np.vstack(x0)
    #sigma = np.vstack(sigma)
    wl = np.array(wl)
    if pardict['instrument'] == 'NIRCam/':
        flag = wl < 4.
    elif pardict['instrument'] == 'NIRSpec_Prism/':
        flag = wl < 6.
    wl = wl[flag]
    A = np.array(A)[flag]
    sigma = np.array(sigma)[flag]
    yerrup = np.array(yerrup)[flag]
    yerrdown = np.array(yerrdown)[flag]

    res = np.diff(wl)[0] #int(3./np.diff(wl)[0])

    #for i in np.arange(len(A)):
    #    #if A[i, 1] > 0:
    #    #    yerr.append(max(abs(A[i, 2] - A[i, 1]), abs(A[i, 3] - A[i, 2])))
    #    #else:
    #    #    yerr.append(abs(A[i, 3] - A[i, 2]))
    #    #yerrup.append(A[i, 3] - A[i, 2])
    #    #if A[i, 1] > 0:
    #    #yerrdown.append(A[i, 2] - A[i, 1])
    #    yerrup.append(res['1sigma_unc'][-2]*1e6)
    #    yerrdown.append(res['1sigma_unc'][-2]*1e6)
        #else:
            #yerrdown.append(A[i, 2])
    #yerr = np.array(yerr)
    #plt.figure()
    #plt.errorbar(wl, sigma[:, 2], yerr=sigma[:, 2] - sigma[:, 0], fmt='ko')
    #plt.title('$\sigma$', fontsize=16)
    #plt.xlabel('Channel', fontsize=16)
    #plt.ylabel('Flux rise [ppm]', fontsize=16)
    '''
    plt.figure()
    plt.errorbar(ind, x0[:, 2], yerr=x0[:, 3] - x0[:, 2], fmt = 'ko')
    plt.title('$x_0$', fontsize=16)
    plt.figure()
    plt.errorbar(ind, sigma[:, 2], yerr=sigma[:, 3] - sigma[:, 2], fmt = 'ko')
    plt.title('$\sigma$', fontsize=16)
    '''

    dict_results = {}
    # Try with different stellar models and find best Tspot fit
    for stmod in models:
        pm = pardict['magstar']
        dict_results[pm] = {}
        dict_results[pm][stmod] = {}
        if pardict['tstar'] == 3500:
            tspot_ = np.arange(2300, 3400, 100)
        else:
            tspot_ = np.arange(3500, 4900, 100)
        tspot_ = tspot_[tspot_ != pardict['tstar']]
        chi2red = np.zeros(len(tspot_)) + np.inf
        plt.figure()
        plt.errorbar(wl, A, yerr=[yerrup, yerrdown], fmt='ko', mfc='None')
        plt.title('$A$', fontsize=16)
        print('Fitting ' + stmod + ' models...')
        # Get spectra with starspot contamination and perform LM fit
        for i, temp in enumerate(tspot_):
            if temp == pardict['tstar']:
                continue
            ww, spec = combine_spectra(pardict, [temp], 0.05, stmod, res=res)
            ww/= 1e4
            spec*= A.max()
            specint = interp1d(ww, spec)
            specA = specint(wl)
            # Use max uncertainty between yerrup and yerrdown
            yerrfit = []
            for j in np.arange(len(yerrup)):
                yerrfit.append(max([yerrup[j], yerrdown[j]]))
            yerrfit = np.array(yerrfit)
            boundsm = ([0.], [np.inf])
            soln = least_squares(scalespec, [1.], \
                        bounds=boundsm, args=(specA, A, yerrfit))
            #scale_unc = getunc.unc_jac(soln.fun, soln.jac, len(yerrfit) - 1)
            print(temp, 'K Spectrum scaling factor:', soln.x[0])#, '+/-', \
            #                    scale_unc[0])
            plt.plot(ww, soln.x[0]*spec, label=str(temp), alpha=.5)
            #plt.plot(ww, soln.x[0]*spec + soln.x[1], label=str(temp), alpha=.5)
            chi2red[i] = np.sum((A - soln.x[0]*specA)**2/yerrfit**2) \
                            /(len(yerrfit) - 1)
            dict_results[pm][stmod][temp - pardict['tstar']] = chi2red[i]
        #plt.legend(frameon=False, loc='best', fontsize=16)
        plt.xlabel('Wavelength [$\mu m$]', fontsize=16)
        plt.ylabel('Transit depth rise [ppm]', fontsize=16)
        chi2min = np.argmin(chi2red)
        print('chi2 min =', chi2red[chi2min], 'with Tspot =', \
                                    tspot_[chi2min], 'K')
        plt.title(stmod + r', $\min (\tilde{\chi}^2)=$' \
           + str(np.round(chi2red[chi2min], 2)) \
           + r', $T_\mathrm{spot}=$' + str(tspot_[chi2min]) + ' K', fontsize=16)
        plt.xlim(wl.min() - 0.2, wl.max() + 0.2)
        plt.ylim(0., 6000)
        plt.savefig(plotname + stmod + '_' + instrument + '.pdf')
        plt.close('all')

        fresults = open(resfile + stmod + '.pic', 'wb')
        pickle.dump(dict_results, fresults)
        fresults.close()

    return np.array(wl), np.array(A), np.array(x0), np.array(sigma)

def scalespec(x, spec, y, yerr):
    '''
    Distance between model spectrum and data.
    '''
    return (x[0]*spec - y)**2/yerr**2
    #return (x[0]*spec + x[1] - y)**2/yerr**2

def combine_spectra(pardict, tspot, ffact, stmodel, res=100.):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19

    Input
    -----
    tspot: list (of effective temperatures)
    '''

    # This is in Angstroms
    if stmodel == 'ck04models' or stmodel == 'phoenix':
        wave = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).wave
        star = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).flux
    #elif stmodel == 'phoenix':
    #    wave = fits.open(modelsfolder \
    #                +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    #    star = fits.open(modelsfolder + 'lte0' + str(int(pardict['tstar'])) \
    #        + '-' + '{:3.2f}'.format(pardict['loggstar']) \
    #        + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    # Reduce resolution to the one of the transmission spetrum

    #wnew = rebin.rebin_wave(wave, res)
    #rebstar, errstar = rebin.rebin_spectrum(star, wave, wnew, unc=[0.])
    #if int(res*1e4/2) % 2 == 1:
    #    kern = int(res*1e4/2.)
    #else:
    #    kern = int(res*1e4/2.) - 1
    kern = 5
    wnew = wave[::kern]
    rebstar = medfilt(star, kernel_size=kern)[::kern]
    rebstar[rebstar < 1e-6] = 0.
    # Increase errors (based on ETC?)
    #errstar += rebstar*abs(np.random.normal(loc=0., scale=5.5e-3, \
    #                        size=len(errstar)))
    #plt.errorbar(wnew/10., rebstar, yerr=errstar, label=str(tstar))

    for i in tspot:
        if stmodel == 'ck04models' or stmodel == 'phoenix':
            spot = pysynphot.Icat(stmodel, i, 0.0, pardict['loggstar'] - 0.5).flux
        #elif stmodel == 'phoenix':
        #    spot = fits.open(modelsfolder + 'lte0' + str(i) \
        #            + '-' + '{:3.2f}'.format(pardict['loggstar'] - 0.5) \
        #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        specnew = spot#(1 - ffact)*star + ffact*spot
        #rebnew, errnew = rebin.rebin_spectrum(specnew, wave, wnew)
        rebnew = medfilt(specnew, kernel_size=kern)[::kern]
        rebnew[rebnew < 1e-6] = 0.
        #plt.plot(spot)
        #plt.plot(star)
        #plt.plot(rebnew/rebstar)
        #plt.show()
        #set_trace()
        #errnew += rebnew*abs(np.random.normal(loc=0., scale=5.5e-3,\
        #                size=len(errnew)))
        # Compare plots
        #plt.errorbar(wnew/10., rebnew, yerr=errnew, label=str(i))
        # As Sing+2011
        wref = np.logical_and(6000 < wnew, wnew < 7000)
        #wref = np.logical_and(24000 < wnew, wnew < 25000)
        spotref = rebnew[wref]
        starref = rebstar[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - rebnew/rebstar)/fref
        #rise2 = (1. - specnew/star)/fref
        #set_trace()
        #plt.figure()
        #set_trace()
        #plt.plot(wnew*1e-4, rebstar)
        #plt.plot(wnew*1e-4, rebnew)
        #plt.plot(wave/1e4, rise2)
        plt.plot(wnew/1e4, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
                       + str(i) + 'K')
        #plt.xlim(0.6, 5)
        #plt.ylim(0, 4)
        #plt.show()
        #set_trace()
        #return
    #plt.legend(frameon=False, fontsize=14)
    plt.title('Stellar spectrum: ' + str(pardict['tstar']) + ' K', fontsize=16)
    #plt.plot([0.719, 0.719], [0.5, 9.5], 'k--')
    plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
    plt.ylabel('Flux rise [Relative values]', fontsize=16)
    #plt.xlim(0.6, 5)
    #plt.ylim(1500, 6500)
    #plt.show()
    flag = np.logical_or(np.isnan(wnew), np.isnan(rise))

    return wnew[~flag], rise[~flag]

def plot_precision(pardict, xaxis, deltapar):
    '''
    Plot starspot parameter precision as a function of some simulation
    parameter.

    Deltapar is used for the final plot label.
    '''

    res = {}
    res['x'] = []
    res['y'] = []

    #plt.figure(figsize=(8, 7))
    # Take the min chi2 value for each stellar model, then get stddev
    diff_spotpar = []
    std_spotpar = []
    for j, xpar in enumerate(xaxis):
        resfolder = pardict['project_folder'] \
                    + pardict['instrument'] + 'p' + str(np.round(xpar, 4)) \
                    + '_star' + str(pardict['rstar']) + '_' \
                    + str(pardict['tstar']) + '_' + str(pardict['loggstar']) \
                    + '_spot' + str(pardict['tumbra']) + '_' \
                + str(pardict['tpenumbra']) + '_mag' + str(pardict['magstar'])
        chi2results = pickle.load(open(resfolder + '/MCMC/contrast_res.pic', \
                        'rb'))
        tmin, chi2min = [], []
        for stmod in ['phoenix']:#, 'ck04models']:
            t, chi2 = [], []
            for it in chi2results[stmod].items():
                t.append(it[0])
                chi2.append(it[1])
            chi2 = np.array(chi2)
            t = np.array(t)
            ind = chi2.argmin()
            tmin.append(t[ind])
            chi2min.append(chi2[ind])
        diff_spotpar.append(abs(pardict['tumbra'] - np.mean(tmin)))
        # Largest sunspot temp - min with different starspot models
        std_spotpar.append(max(tmin) - min(tmin))
        plt.errorbar([abs(xpar)], diff_spotpar[j], yerr=std_spotpar[j], \
                            fmt='ko', ms=10, mfc='white', capsize=2)
        res['x'].append(xpar)
        res['y'].append(diff_spotpar[j])

    # Polynomial fit
    xaxis = np.array(xaxis)
    fit = np.polyfit(abs(xaxis), np.array(diff_spotpar), 2, \
                        w=1./np.array(std_spotpar))
    fun = np.polyval(fit, abs(xaxis))
    plt.plot(abs(xaxis), fun, 'r--', ms=2)
    plt.xlabel(deltapar, fontsize=16)
    #plt.xlabel('$T_\star - T_\mathrm{spot}$', fontsize=16)
    #plt.xlabel('$R_\mathrm{p} [R_\mathrm{J}]$', fontsize=16)
    plt.ylabel('$|T_\mathrm{spot, mod} - T_\mathrm{spot, meas}$|', fontsize=16)
    plt.text(0.8, 1250, '$T_\star =$' + str(pardict['tstar']) \
                    + ' K', fontsize=16)
    plt.show()
    plt.savefig(pardict['project_folder'] \
                + pardict['instrument'] + 'diff_' + deltapar + '.pdf')

    return res

def spectrum_var(res=100000):
    '''
    Apply correction factor to planet transmission spectrum from Ballerini+2012.
    '''

    wout = np.arange(0.55, 5.5, 0.001)
    fplanet = '/home/giovanni/Dropbox/Projects/jwst_spots/NIRSpec_Prism/1000K_jupiter.pickle'
    planet = pickle.load(open(fplanet, 'rb'))
    w = planet['OriginalInput']['model_wave']
    D = planet['OriginalInput']['model_spec']*1e6
    #wnewp = rebin.rebin_wave(w, res)
    #D_new, errp_ = rebin.rebin_spectrum(D, w, wnewp)

    tstar = 5000
    tspot = 4300
    logg = 4.5
    wavestar = fits.open(modelsfolder \
                +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data/1e4
    fstar = fits.open(modelsfolder + 'lte0' + str(tstar) \
            + '-' + '{:3.2f}'.format(logg) \
            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    fspot = fits.open(modelsfolder + 'lte0' + str(tspot) \
            + '-' + '{:3.2f}'.format(logg - 0.5) \
            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    Alambda = 1. - fspot/fstar

    # Bring planet spectrum to same resolution
    #wnew = rebin.rebin_wave(wavestar, res)
    #Alambda_new, err = rebin.rebin_spectrum(Alambda, wavestar, wnew)
    #set_trace()
    interp_planet = interp1d(w, D)
    D_newres = interp_planet(wout)
    interp_star = interp1d(wavestar, Alambda)
    Alambda_new = interp_star(wout)

    #plt.plot(w, D, label='Real')
    fi = 0.05
    for fo in np.arange(0.05, 0.5, 0.1):
        corrf = (1. - fi*Alambda_new)/(1. - fo*Alambda_new)
        # Apparent transit depth
        Dapp = D_newres*corrf
        #plt.plot(wout, Dapp, label='$f_o =' + str(np.round(fo, 2)) + '$')
        #plt.plot(wout, corrf)
        plt.plot(wout, 2e-5/(1. - fo*Alambda_new)*1e6, label='$f_o =' + str(np.round(fo, 2)) + '$')
    plt.xlabel('Wavelenght [$\mu$m]', fontsize=16)
    plt.ylabel('Transit depth variation [ppm]', fontsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.title('Assumed photometric precision: 20 ppm', fontsize=16)
    plt.xlim(0.55, 5.4)
    plt.show()
    '''
    fi = np.linspace(0.01, 0.5, 100)
    fo = np.linspace(0.01, 0.5, 100)
    corrf = (1. - fi*Alambda_new)/(1. - fo*Alambda_new)
    Dapp = D_newres*corrf
    '''
    return

def compare_pysynphot_phoenix():
    logg = 5.0
    for tstar in np.arange(2500, 3500, 200):
        wavestar = fits.open(modelsfolder \
                        +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data/1e4
        fstar = fits.open(modelsfolder + 'lte0' + str(tstar) \
                    + '-' + '{:3.2f}'.format(logg) \
                    + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        pysp = pysynphot.Icat('phoenix', tstar, 0.0, logg)
        wpy = pysp.wave/1e4
        fpy = pysp.flux
        plt.plot(wavestar, fstar/fstar.max(), 'b')
        plt.plot(wpy, fpy/fpy.max(), 'r')
    plt.show()
    plt.xlim(0.1, 10)
    plt.figure()
    plt.plot(np.diff(wavestar))
    plt.plot(np.diff(wpy))
    plt.show()
    return
