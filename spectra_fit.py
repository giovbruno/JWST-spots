# Compare spot fits with stellar models.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy import stats
import pysynphot
import pickle
from pdb import set_trace
from astropy.io import fits
from astropy.modeling import blackbody as bb
from simulate_transit import degrade_spec, integ_filter

modelsfolder = '/home/giovanni/Shelf/stellar_models/phxinten/HiRes/'
foldthrough = '/home/giovanni/Shelf/filters/'
intensfolder = '/home/giovanni/Shelf/stellar_models/phxinten/SpecInt/'
thrfile1 = foldthrough + 'JWST_NIRCam.F150W2.dat'
thrfile2 = foldthrough + 'JWST_NIRCam.F322W2.dat'
thrfile3 = foldthrough + 'JWST_NIRCam.F444W.dat'
thrfile4 = foldthrough + 'JWST_NIRSpec.CLEAR.dat'

plt.ioff()

def read_res(pardict, plotname, resfile, models, fittype='grid', \
            resol=10):
    '''
    Read in the chains files and initialize arrays with spot properties.

    wlwin: used to exlude parts of the spectra
    '''

    if pardict['instrument'] == 'NIRSpec_Prism':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam':
        ldendname = 'nircam'
    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    xobs = pickle.load(ldd)[1][0]
    ldd.close()
    flag = xobs < 6
    expchan = len(xobs[flag])

    # Read results
    spec = pickle.load(open(pardict['data_folder'] + 'spec_model_' \
                + pardict['observatory'] + '.pic', 'rb'))
    ymoderr = spec[2]
    bestbin = ymoderr.argmin()
    #bestbin = 1
    wl, A, x0, sigma = [], [], [], []
    yerrup, yerrdown = [], []
    kr, krunc = [], []
    for i in np.arange(expchan):
        ffopen = open(pardict['chains_folder'] + 'chains_' \
                + str(i) + '.pickle', 'rb')
        res = pickle.load(ffopen)
        wl.append(res['wl'])
        if i == bestbin:
            perc = res['Percentiles'][0][-4]
            tspot = res['Percentiles'][0][-1]
        else:
            perc = res['Percentiles'][0][-1]
        A.append(perc[2])
        yerrup.append(perc[3] - perc[2])
        yerrdown.append(perc[2] - perc[1])
        ffopen.close()

    # Take mean value for spot position
    tspot = tspot[2]
    # Get **observed** mid-transit time
    mufit = -1
    ### Starspot spectrum
    # Quadraticall add uncertainty from first point
    #for j in np.arange(1, len(A)):
    #    yerrup[j] = np.sqrt(yerrup[0]**2 + yerrup[j]**2)
    #    yerrdown[j] = np.sqrt(yerrdown[0]**2 + yerrdown[j]**2)
    wl = np.array(wl)
    if pardict['instrument'] == 'NIRCam':
        flag = wl < 4.
    elif pardict['instrument'] == 'NIRSpec_Prism':
        flag = wl < 6.

    wl = wl[flag]
    A = np.array(A)[flag]
    yerrup = np.array(yerrup)[flag]
    yerrdown = np.array(yerrdown)[flag]
    # Here you can change the reference wavelength, according to the instrument
    global wlminPrism
    global wlmaxPrism
    global wlminNIRCam
    global wlmaxNIRCam
    wlminPrism = 1.
    wlmaxPrism = 1.8
    wlminNIRCam = 1.
    wlmaxNIRCam = 1.8
    if pardict['instrument'] == 'NIRSpec_Prism':
        wref = np.logical_and(wlminPrism < wl, wl < wlmaxPrism)
    elif pardict['instrument'] == 'NIRCam':
        wref = np.logical_and(wlminNIRCam < wl, wl < wlmaxNIRCam)
    Aref = np.mean(A[wref])
    set_trace()
    yerrup /= Aref
    yerrdown /= Aref
    A /= Aref
    plt.figure()
    plt.errorbar(wl, A, yerr=[yerrup, yerrdown], fmt='ko', mfc='None', \
                    capsize=2)
    plt.arrow(wl[bestbin], A[bestbin] + 1., 0, -0.5, linewidth=3)

    res = np.diff(wl)[0]

    dict_results = {}
    # Try with different stellar models and find best Tspot fit
    for stmod in [models]:
        pm = pardict['magstar']
        dict_results[pm] = {}
        dict_results[pm][stmod] = {}
        if pardict['tstar'] == 3500:
            tspot_ = np.arange(2300, pardict['tstar'], 100)
        else:
            tspot_ = np.arange(3600, pardict['tstar'], 100)
        # Arrays for likelihood calculation
        likelihood = np.zeros(len(tspot_)) + np.inf
        chi2r = np.copy(likelihood)

        print(plotname + stmod + '_' + pardict['observatory'])
        print('Fitting ' + stmod + ' models...')
        # Get spectra with starspot contamination and perform LM fit
        for i, temp in enumerate(tspot_):
            if temp == pardict['tstar']:
                continue

            ww, spec = combine_spectra(pardict, [temp], 0.05, stmod, \
                    wl, mufit=mufit, res=res, isplot=False)
            specint = interp1d(ww, spec, bounds_error=False, \
                            fill_value='extrapolate')
            specA = specint(wl)

            boundsm = ([-10, 10])
            soln = least_squares(scalespec, [0.], \
                        bounds=boundsm, args=(specA, A, yerrup, yerrdown))
            print(temp, 'K, spectrum shifting factor:', soln.x[0])
            plt.plot(wl, specA + soln.x[0], label=str(int(temp)) + ' K')
            chi2 = np.sum(scalespec(soln.x, specA, A, yerrup, yerrdown)) \
                        /(len(A) - 1)
            chi2r[i] = chi2
            likelihood[i] = np.exp(-0.5*chi2)
            dict_results[pm][stmod][temp - pardict['tstar']] = likelihood[i]

        plt.xlabel('Wavelength [$\mu m$]', fontsize=16)
        plt.ylabel(r'$\Delta f(\lambda)/\Delta f(\lambda_0)$', \
                    fontsize=16)
        maxL = np.argmax(likelihood)
        print('L max =', likelihood[maxL], 'with Tspot =', \
                                    tspot_[maxL], 'K')
        plt.title(pardict['instrument'] \
           + r', best fit: $T_\bullet=$' + str(int(tspot_[maxL])) + ' K', \
            fontsize=16)
        plt.xlim(wl.min() - 0.2, wl.max() + 0.2)
        plt.savefig(plotname + stmod + '_' + pardict['observatory'] + '.pdf')
        plt.close('all')

    plt.figure()
    x, y = [], []
    for jj in dict_results[pardict['magstar']][models].keys():
        x.append(jj)
        y.append(dict_results[pardict['magstar']][models][jj])
    x = np.array(x)
    y = np.array(y)
    C = np.sum(y)
    prob = y/C

    valmax = prob.max()
    xmax = prob.argmax()

    # Create PDF with the prob sampling
    pdf = stats.rv_discrete(a=x.min(), b=x.max(), values=(x, prob))
    Tmean = pdf.mean()
    Tconf = [pdf.std(), pdf.std()]
    Tconf = pdf.interval(0.642) # % 64.2% confence interval
    dist = pdf.mean() - (pardict['tumbra'] - pardict['tstar'])
    #Tunc = dist
    #if (Tconf == pdf.median()).any():
    #    set_trace()
    #if dist > 0:
    #    Tsigma = abs(dist)/(Tconf[1] - pdf.median())
    #else:
    #    Tsigma = abs(dist)/(pdf.median() - Tconf[0])
    Tsigma = pdf.std()
    if Tsigma < np.diff(tspot_)[0]:
        Tsigma = np.diff(tspot_)[0]
    # Fit a Gaussian centered here
    #bbounds = ([0, x.min(), 50], [2*valmax, x.max(), 1000.])
    #fitgauss = lambda p, t, u: (u - gauss(p, t))**2
    #soln = least_squares(fitgauss, [valmax, x[prob.argmax()], 150.], \
    #            args=(x, prob), bounds=bbounds)
    #Tunc = soln.x[2]
    dict_results[pardict['magstar']]['Tunc'] = [dist, Tsigma]
    #ress = dict_results[10.5]['phoenix'].values()
    #flag = ress < 2e-22
    plt.plot(x, prob, 'k-', label=r'$\Delta T = $' + str(int(dist)) \
            + 'K\n' + str(np.round(dist/Tsigma, 1)) + r'$\sigma$')
    #xTh = np.linspace(x.min(), x.max(), 1000)
    #plt.plot(xTh, gauss(soln.x, xTh), 'cyan')
    plt.plot([-pardict['tstar'] + pardict['tumbra'], \
            -pardict['tstar'] + pardict['tumbra']], [prob.min(), \
                    prob.max()], 'r--')
    plt.xlabel(r'$\Delta T_\mathrm{spot}$ [K]', fontsize=16)
    plt.ylabel('Probability likelihood', fontsize=16)
    plt.legend()
    plt.savefig(plotname + stmod + '_' + pardict['observatory'] + '_like.pdf')
    fresults = open(resfile + stmod + '_grid.pic', 'wb')

    pickle.dump(dict_results, fresults)
    fresults.close()

    return np.array(wl), np.array(A), np.array(x0), np.array(sigma)

def gauss(par, x):
    '''
    Par is defined as [A, x0, sigma]
    '''
    A, x0, sigma = par
    return A*np.exp(-0.5*(x - x0)**2/sigma**2)

def scalespec(x, spec, y, yerrup, yerrdown):
    '''
    Distance between model spectrum and data (with unequal uncertainties).
    '''
    res = np.zeros(len(y))
    flag = spec + x[0] >= y
    res[flag] = (spec + x[0] - y)[flag]**2/yerrup[flag]**2
    res[~flag] = (spec + x[0] - y)[~flag]**2/yerrdown[~flag]**2

    return res

def combine_spectra(pardict, tspot, ffact, stmodel, wnew, \
                                        res=100., mufit=-1, isplot=False):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19
    Use Phoenix models in pysynphot (submitted version)
    Use Josh's models (revision 1)

    Input
    -----
    tspot: list (of effective temperatures)
    wl: array to degrade the model spectrum. All wavelengts are in Angrstoms
    mufit: the mu angle of the transited spot, found by the transit fit
    '''

    if not pardict['spotted_starmodel']:
        if stmodel == 'phoenix':
            wave = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                    pardict['loggstar']).wave
            star = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                    pardict['loggstar']).flux
        elif stmodel == 'josh':
            star = pardict['starmodel']['spec'][pardict['muindex']]
            wave = pardict['starmodel']['wl']
    else:
        wave, star = np.loadtxt(pardict['data_folder'] \
                    + 'spotted_star.dat', unpack=True)

    if pardict['instrument'] == 'NIRSpec_Prism':
        wth, fth = np.loadtxt(thrfile4, unpack=True)
        wth*= 1e4
    elif pardict['instrument'] == 'NIRCam':
        wth1, fth1 = np.loadtxt(thrfile1, unpack=True)
        wth2, fth2 = np.loadtxt(thrfile2, unpack=True)
        wth3, fth3 = np.loadtxt(thrfile3, unpack=True)
        wth = np.concatenate((wth1, wth2, wth3))
        fth = np.concatenate((fth1, fth2, fth3))
    star = integ_filter(wth, fth, wave, star)
    #fflag = wth < 60000.
    #wth = wth[fflag]
    #fth = fth[fflag]
    #star = star[fflag]
    rebstar = degrade_spec(star, wth, wnew)

    for i in tspot:
        if stmodel == 'ck04models' or stmodel == 'phoenix':
            wls = pysynphot.Icat(stmodel, i, 0.0, \
                                            pardict['loggstar'] - 0.5).wave
            spot = pysynphot.Icat(stmodel, i, 0.0, \
                                            pardict['loggstar'] - 0.5).flux
        elif stmodel == 'josh':
            tspot_ = format(i, '2.2e')
            wls = pardict['spotmodels'][tspot_]['wl']
            spot = pardict['spotmodels'][tspot_]['spec'][mufit]
        #fflagu = wls < 60000.
        #wls = wls[fflagu]
        #spot = spot[fflagu]
        spot = integ_filter(wth, fth, wls, spot)
        rebnew = degrade_spec(spot, wth, wnew)

        # As Sing+2011
        if pardict['instrument'] == 'NIRSpec_Prism':
            wref = np.logical_and(wlminPrism < wnew, wnew < wlmaxPrism)
        elif pardict['instrument'] == 'NIRCam':
            wref = np.logical_and(wlminNIRCam < wnew, wnew < wlmaxNIRCam)
        spotref = rebnew[wref]
        starref = rebstar[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - rebnew/rebstar)/fref

        if isplot:
            plt.plot(wnew, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
                        + str(i) + 'K')
    if isplot:
        plt.title('Stellar spectrum: ' + str(pardict['tstar']) + ' K', \
                    fontsize=16)
        plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
        plt.ylabel('Flux rise [Relative values]', fontsize=16)

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

    # Take the min chi2 value for each stellar model, then get stddev
    diff_spotpar = []
    std_spotpar = []
    for j, xpar in enumerate(xaxis):
        resfolder = pardict['project_folder'] \
                    + pardict['instrument'] + '/p' + str(np.round(xpar, 4)) \
                    + '_star' + str(pardict['rstar']) + '_' \
                    + str(pardict['tstar']) + '_' + str(pardict['loggstar']) \
                    + '_spot' + str(pardict['tumbra']) + '_' \
                + str(pardict['tpenumbra']) + '_mag' + str(pardict['magstar'])
        chi2results = pickle.load(open(resfolder + '/MCMC/contrast_res.pic', \
                        'rb'))
        tmin, chi2min = [], []
        for stmod in ['josh']:#, 'ck04models']
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
                + pardict['instrument'] + '/diff_' + deltapar + '.pdf')

    return res

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

def compare_contrast_spectra(tstar, loggstar, tspot):
    '''
    Compare stellar spectra to black body curves.
    '''
    starmod = pysynphot.Icat('phoenix', tstar, 0.0, loggstar)
    starflux = starmod.flux
    wl = starmod.wave#/1e4
    #wl = fits.open(modelsfolder \
    #                +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    #starflux = fits.open(modelsfolder + 'lte0' + str(tstar) \
    #            + '-' + '{:3.2f}'.format(loggstar) \
    #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    wnew = np.linspace(0.6, 5.3, 100)
    #wnew = wl*1e-4
    starflux = degrade_spec(starflux, wl, wnew)
    #wl = wl[::100]
    #starflux = starflux[::100]
    for ti in tspot:
        spotflux = pysynphot.Icat('phoenix', ti, 0.0, loggstar - 0.5).flux
        #spotflux = spotmod.flux
        #spotflux = fits.open(modelsfolder + 'lte0' + str(ti) \
        #    + '-' + '{:3.2f}'.format(loggstar - 0.5) \
        #    + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        #spotflux = spotflux[::100]
        spotflux = degrade_spec(spotflux, wl, wnew)
        #wref = np.logical_and(7.5 < wl/1e4, wl/1e4 < 9.0)
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
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'Wavelength [$\mu$m]', fontsize=14)
    plt.ylabel(r'Brightness contrast at $\mu=1$ ($A_\lambda$)', fontsize=14)
    plt.title('$T_\star=$ ' + str(tstar) + ' K', fontsize=16)
    plt.legend(frameon=False)
    plt.show()
    plt.savefig('/home/giovanni/Projects/jwst_spots/contrast_model_' \
                + str(tstar) + '.pdf')

    return

def compare_intens_integrated(res, tspot, tstar):
    '''
    Compare specific intensity contrast spectra and integrated flux spectra
    at a given resolution.
    '''

    eespot = fits.open(intensfolder \
            + 'lte0' + str(tspot) \
            + '-4.00-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits')
    eestar = fits.open(intensfolder \
            + 'lte0' + str(tstar) \
            + '-4.50-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits')
    wl = np.array([eestar[0].header['CRVAL1'] + (eestar[0].header['CDELT1']*i) \
                for i in np.arange(eestar[0].header['NAXIS1'])])
    ffspot = pysynphot.Icat('phoenix', tspot, 0.0, 4.0)
    ffstar = pysynphot.Icat('phoenix', tstar, 0.0, 4.5)
    wave = ffspot.wave
    flag = np.logical_and(wave > 500, wave < 25999)
    ffspot = ffspot.flux[flag]
    ffstar = ffstar.flux[flag]
    wave = wave[flag]

    wnew = wlres(0.6, 2.6, res)
    contrflux = ffspot/ffstar
    contrflux_wavenew = degrade_spec(contrflux, wave, wnew)
    mu = [-1, -10, -11, -13, -15, -20]

    for j in mu[::-1]:
        intens_spot = eespot[0].data[j, :]
        intens_star = eestar[0].data[j, :]
        contr = intens_spot/intens_star
        contr_mod = interp1d(wl, contr)
        contr_wave = contr_mod(wave)
        contr_wavenew = degrade_spec(contr_wave, wave, wnew)
        flag = np.logical_or(np.isnan(contr_wavenew), np.isnan(contrflux_wavenew))

        plt.plot(wnew[~flag], \
                (contr_wavenew[~flag]/contrflux_wavenew[~flag] - 1.)*100., \
                label=np.round(np.degrees(np.arccos(eespot[1].data[j])), \
                                    1), alpha=0.4)
        print('mu =', eespot[1].data[j], '=>', \
            np.round(np.degrees(np.arccos(eespot[1].data[j])), 1), 'deg')
    plt.plot([0., 2.6], [5, 5], 'k--')
    plt.plot([0., 2.6], [-5, -5], 'k--')
    plt.legend(title=r'$\theta$ [deg]')
    #plt.ylim(-50, 100)
    plt.xlim(0.6, 2.6)
    plt.ylabel(r'$\Delta$ contrast [%]', fontsize=14)
    plt.xlabel(r'Wavelength [$\mu$m]', fontsize=14)
    plt.show()
    #plt.savefig('/home/giovanni/Projects/jwst_spots/specint_vs_flux.pdf')

    return

def wlres(wli, wlf, res):
    '''
    Build a wl array with a given resolution.
    '''
    wl = []
    wl.append(wli)
    while wl[-1] < wlf:
        wl.append(wl[-1] + wl[-1]/res)

    return np.array(wl)
