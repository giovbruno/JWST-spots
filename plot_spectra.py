import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
import rebin
import numpy as np
from decimal import Decimal
from astropy.modeling.blackbody import blackbody_lambda
import pysynphot
from pdb import set_trace

#folder = '/home/giovanni/archive/Stellar_models/phxinten/HiRes/'
folder = '/home/giovanni/archive/python/pysynphot/synphot1/grp/hst/cdbs/grid/k93models/kp00/'

def plot_contrast(tstar, tspotmin, tspotmax, tspotdelta, res=100):
    '''
    Plots contrast ratio at a given resolution.
    Fit with Kurucz instead of Phoenix models 2/10/19
    '''

    # This is in Angstroms
    #wave = fits.open(folder + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    wave = fits.open(folder \
                + 'kp00_' + str(tstar) + '.fits')[1].data['WAVELENGTH']
    # Reduce resolution
    wnew = rebin.rebin_wave(wave, res)

    #star = fits.open(folder + 'lte0' + str(tstar) \
    #    + '-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    star = fits.open(folder + 'kp00_' + str(tstar) \
            + '.fits')[1].data['g45']
    rebstar, errstar = rebin.rebin_spectrum(star, wave, wnew)
    #plt.errorbar(wnew/10., rebstar, yerr=errstar)

    for i in np.arange(tspotmin, tspotmax, tspotdelta):
        #spot = fits.open(folder + 'lte0' + str(i) \
        #        + '-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        spot = fits.open(folder + 'kp00_' + str(i) \
            + '.fits')[1].data['g45']
        rebspot, errspot = rebin.rebin_spectrum(spot, wave, wnew)
        # Add 10 ppm uncertainty
        #errspot += abs(np.random.normal(loc=0., scale=1e-2, size=len(errspot)))
        errspot += rebspot*abs(np.random.normal(loc=0., scale=5.5e-3,\
                            size=len(errspot)))
        #mm = Decimal(str(np.mean(errspot[np.logical_and(wnew > 1000, \
        #                        wnew < 15000)])))
        #print('Average uncertainty:', '{:.2e}'.format(mm))
        plt.errorbar(wnew/10., rebspot, yerr=errspot, label=str(i))
        #plt.errorbar(wnew/10., rebspot/rebstar, yerr=rebspot/rebstar \
        #            *((errspot/rebspot)**2 + (errstar/rebstar)**2)**0.5, \
        #                label=str(i))
    plt.legend()
    plt.show()

    return

def combine_spectra(pardict, tspot, ffact, res=100.):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19

    Input
    -----
    tspot: list (of effective temperatures)
    '''
    # This is in Angstroms
    #wave = fits.open(folder + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    #wave = fits.open(folder \
    #            + 'kp00_' + str(tstar) + '.fits')[1].data['WAVELENGTH']
    wave = pysynphot.Icat('ck04models', pardict['tstar'], 0.0, \
                pardict['loggstar']).wave
    # Reduce resolution
    wnew = rebin.rebin_wave(wave, res)
    #star = fits.open(folder + 'lte0' + str(tstar) \
    #    + '-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    #star = fits.open(folder + 'kp00_' + str(tstar) \
    #        + '.fits')[1].data['g45']
    star = pysynphot.Icat('ck04models', pardict['tstar'], 0.0, \
                pardict['logg']).flux
    rebstar, errstar = rebin.rebin_spectrum(star, wave, wnew)
    # Increase errors (based on ETC?)
    errstar += rebstar*abs(np.random.normal(loc=0., scale=5.5e-3,\
                            size=len(errstar)))
    #plt.errorbar(wnew/10., rebstar, yerr=errstar, label=str(tstar))

    for i in tspot:
        #spot = fits.open(folder + 'lte0' + str(i) \
        #    + '-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        #spot = fits.open(folder + 'kp00_' + str(i) \
        #    + '.fits')[1].data['g45']
        spot = pysynphot.Icat('ck04models', pardict['tumbra'], 0.0, \
                    pardict['loggstar'])
        specnew = spot#(1 - ffact)*star + ffact*spot
        rebnew, errnew = rebin.rebin_spectrum(specnew, wave, wnew)
        errnew += rebnew*abs(np.random.normal(loc=0., scale=5.5e-3,\
                        size=len(errnew)))
        # Compare plots
        #plt.errorbar(wnew/10., rebnew, yerr=errnew, label=str(i))
        # As Sing+2011
        wref = np.logical_and(39000 < wnew, wnew < 41000)
        spotref = rebnew[wref]
        starref = rebstar[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - rebnew/rebstar)/fref
        '''
        plt.errorbar(wnew/1e4, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
                        + str(i) + 'K')

    plt.legend(frameon=False)
    plt.title('Stellar spectrum: ' + str(tstar) + ' K', fontsize=16)
    #plt.plot([0.719, 0.719], [0.5, 9.5], 'k--')
    plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
    plt.ylabel('Dimming factor', fontsize=16)
    plt.show()
    '''
    flag = np.logical_or(np.isnan(wnew), np.isnan(rise))

    return wnew[~flag], rise[~flag]

def combine_spectra_justplot(tstar, tspot, ffact, res=100.):
    '''
    Combine starspot and stellar spectrum.

    Input
    -----
    tspot: list (of effective temperatures)

    '''
    # This is in Angstroms
    #wave = fits.open(folder + 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    wave = fits.open(folder \
                + 'kp00_' + str(tstar) + '.fits')[1].data['WAVELENGTH']
    # Reduce resolution
    wnew = rebin.rebin_wave(wave, res)
    #star = fits.open(folder + 'lte0' + str(tstar) \
    #    + '-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    star = fits.open(folder + 'kp00_' + str(tstar) \
            + '.fits')[1].data['g45']
    rebstar, errstar = rebin.rebin_spectrum(star, wave, wnew)
    # Increase errors (based on ETC?)
    errstar += rebstar*abs(np.random.normal(loc=0., scale=5.5e-3,\
                            size=len(errstar)))
    #plt.errorbar(wnew/10., rebstar, yerr=errstar, label=str(tstar))

    for i in tspot:
        #spot = fits.open(folder + 'lte0' + str(i) \
        #    + '-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        spot = fits.open(folder + 'kp00_' + str(i) \
            + '.fits')[1].data['g45']
        specnew = spot#(1 - ffact)*star + ffact*spot
        rebnew, errnew = rebin.rebin_spectrum(specnew, wave, wnew)
        errnew += rebnew*abs(np.random.normal(loc=0., scale=5.5e-3,\
                        size=len(errnew)))
        # Compare plots
        #plt.errorbar(wnew/10., rebnew, yerr=errnew, label=str(i))
        # As Sing+2011
        wref = np.logical_and(39000 < wnew, wnew < 41000)
        spotref = rebnew[wref]
        starref = rebstar[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - rebnew/rebstar)
        plt.plot(wnew/1e4, rise, 'k', alpha=0.5)
        # Plot contrast too
        contr = blackbody_lambda(wave, i)/blackbody_lambda(wave, tstar)
        cref = 1 - np.mean(contr)
        crise = (1. - contr)
        plt.plot(wave/1e4, crise, linewidth=2, \
                    label=r'$T_{\mathrm{eff}, \bullet}$ = ' + str(i) + 'K')

    plt.legend(frameon=False, fontsize=14)
    plt.title('Star: $T_\mathrm{eff}=$ ' + str(tstar) \
                    + ' K', fontsize=16)
    #plt.plot([0.719, 0.719], [0.5, 9.5], 'k--')
    plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
    plt.ylabel('Brightness contrast ($A_\lambda$)', fontsize=16)
    plt.xlim(0.1, 5.)
    plt.show()
    flag = np.logical_or(np.isnan(wnew), np.isnan(rise))

    return
