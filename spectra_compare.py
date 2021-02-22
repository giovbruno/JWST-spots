import numpy as np
import pysynphot
import matplotlib.pyplot as plt
import os
from simulate_transit import degrade_spec, integ_filter
from astropy.modeling import blackbody as bb
from pdb import set_trace

def compare_contrast_spectra(tstar, loggstar, tspot, modgrid, mu=-1):
    '''
    Compare stellar spectra to black body curves.
    mu works only for Josh's models
    '''
    if modgrid == 'phoenix':
        starmod = pysynphot.Icat('phoenix', tstar, 0.0, loggstar)
        starflux = starmod.flux
        wl = starmod.wave
    elif modgrid == 'husser':
        wl = fits.open(modelsfolder \
                    +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
        starflux = fits.open(modelsfolder + 'lte0' + str(tstar) \
                    + '-' + '{:3.2f}'.format(loggstar) \
                    + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    elif modgrid == 'josh':
        josh_grid_folder = os.path.expanduser('~') \
                                        + '/Projects/jwst_spots/josh_models/'
        tstar_ = format(tstar, '2.2e')
        loggstar_ = format(loggstar, '2.2e')
        filename = josh_grid_folder + 'starspots.teff=' \
                        + tstar_ + '.logg=' + loggstar_ + '.z=0.0.irfout.csv'
        g = np.genfromtxt(filename, delimiter=',')
        mus = g[0][1:]
        wl = np.array([g[i][0] for i in range(2, len(g))])
        starflux = np.array([g[i][mu] for i in range(2, len(g))])

    wnew = np.linspace(0.6, 5.3, 100)
    starflux = degrade_spec(starflux, wl, wnew)

    for ti in tspot:
        if modgrid == 'phoenix':
            spotflux = pysynphot.Icat('phoenix', ti, 0.0, loggstar - 0.5).flux
        elif modgrid == 'husser':
            spotflux = fits.open(modelsfolder + 'lte0' + str(ti) \
                + '-' + '{:3.2f}'.format(loggstar - 0.5) \
                + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        elif modgrid == 'josh':
            tspot_ = format(ti, '2.2e')
            loggspot_ = format(loggstar - 0.5, '2.2e')
            filename = josh_grid_folder + 'starspots.teff=' \
                        + tspot_ + '.logg=' + loggspot_ + '.z=0.0.irfout.csv'
            g = np.genfromtxt(filename, delimiter=',')
            spotflux = np.array([g[i][mu] for i in range(2, len(g))])
        spotflux = degrade_spec(spotflux, wl, wnew)
        wref = np.logical_and(4. < wnew, wnew < 5.0)
        spotref = spotflux[wref]
        starref = starflux[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - spotflux/starflux)#/fref
        bspot = bb.blackbody_lambda(wl, ti)
        bstar = bb.blackbody_lambda(wl, tstar)
        wref = np.logical_and(40000. < wl, wl < 50000)
        cref = 1. - np.mean(bspot[wref]/bstar[wref])
        contrast = (1. - bspot/bstar)#/cref
        plt.plot(wnew, rise, label=str(ti) + ' K')
        plt.plot(wl*1e-4, contrast, 'k', alpha=0.5)

    plt.xlim(0.6, 5.2)
    plt.ylim(0., 0.95)
    plt.xlabel(r'Wavelength [$\mu$m]', fontsize=14)
    plt.ylabel(r'Brightness contrast at $\mu=1$', fontsize=14)
    plt.title('$T_\star=$ ' + str(tstar) + ' K', fontsize=16)
    plt.legend(frameon=False)
    plt.show()
    plt.savefig('/home/giovanni/Projects/jwst_spots/contrast_model_' \
                + str(tstar) + '_.pdf')

    return
