# Compare spot fits with stellar models.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.optimize import least_squares
from scipy import stats
import pysynphot
import pickle
from pdb import set_trace
from astropy.io import fits
from astropy.modeling import blackbody as bb
from simulate_transit import degrade_spec, integ_filter
from run_all import ingest_stellarspectra
import os
import sys
sys.path.append('/home/giovanni/Shelf/python/emcee/src/emcee/')
import emcee
import cornerplot

homef = os.path.expanduser('~')
modelsfolder = homef + '/Shelf/stellar_models/phxinten/HiRes/'
foldthrough = homef + '/Shelf/filters/'
intensfolder = homef + '/Shelf/stellar_models/phxinten/SpecInt/'
thrfile1 = foldthrough + 'JWST_NIRCam.F150W2.dat'
thrfile2 = foldthrough + 'JWST_NIRCam.F322W2.dat'
thrfile3 = foldthrough + 'JWST_NIRCam.F444W.dat'
thrfile4 = foldthrough + 'JWST_NIRSpec.CLEAR.dat'

plt.ioff()

def read_res(pardict, plotname, resfile, models, fittype='grid', \
            resol=10, interpolate=True, LM=True):
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
    #bestbin = ymoderr.argmin()
    bestbin = -1
    wl, A, x0, sigma = [], [], [], []
    yerrup, yerrdown = [], []
    #kr, krunc = [], []
    global kr
    for i in np.concatenate((np.arange(expchan - 1), [-1])):
        #if i < expchan - 1:
        ffopen = open(pardict['chains_folder'] + 'chains_' \
                + str(i) + '.pickle', 'rb')
        res = pickle.load(ffopen)
        wl.append(res['wl'])
        if i == bestbin:
            perc = res['Percentiles'][0][-6]
            tspot = res['Percentiles'][0][-2]
        else:
            perc = res['Percentiles'][0][-1]
            kr = res['Percentiles'][0][0][2]
        A.append(perc[2])
        yerrup.append(perc[3] - perc[2])
        yerrdown.append(perc[2] - perc[1])
        ffopen.close()
    #Aref = A[-1]
    #yerrupmed = yerrup[-1]
    #yerrdownmed = yerrdown[-1]
    # Take mean value for spot position
    #tspot = tspot[2]
    # Get **observed** mid-transit time
    print('kr =', kr)
    mufit = pardict['muindex']
    wl = np.array(wl)
    if pardict['instrument'] == 'NIRCam':
        flag = wl < 4.
    elif pardict['instrument'] == 'NIRSpec_Prism':
        flag = wl < 6.

    # Get spot SNR
    spotSNR = A[-1]/np.mean([yerrup[-1], yerrdown[-1]])
    wl = wl[:-1]#[flag]
    A = np.array(A[:-1])#[flag]
    yerrup = np.array(yerrup[:-1])#[flag]
    yerrdown = np.array(yerrdown[:-1])#[flag]
    # Here you can change the reference wavelength, according to the instrument
    global wlminPrism
    global wlmaxPrism
    global wlminNIRCam
    global wlmaxNIRCam
    wlminPrism = 0.6
    wlmaxPrism = 5.3
    #wlminPrism = wl[np.mean([yerrdown, yerrup], axis=0).argmin()] - 0.1#0.6
    #wlmaxPrism = wl[np.mean([yerrdown, yerrup], axis=0).argmin()] + 0.1#0.8
    wlminNIRCam = wl[np.mean([yerrdown, yerrup], axis=0).argmin()] - 0.1#3.8
    wlmaxNIRCam = wl[np.mean([yerrdown, yerrup], axis=0).argmin()] + 0.1#4.0
    if pardict['instrument'] == 'NIRSpec_Prism':
        wref = np.logical_and(wlminPrism < wl, wl < wlmaxPrism)
    elif pardict['instrument'] == 'NIRCam':
        wref = np.logical_and(wlminNIRCam < wl, wl < wlmaxNIRCam)
    #global Aref
    Aref = np.mean(A[wref])
    #yerrup /= Aref
    #yerrdown /= Aref
    yerrmed = np.std(A[wref])/((np.sum(wref))**0.5)
    #yerrup = np.sqrt(A**2/Aref**4*yerrmed**2 + yerrup**2/Aref**2)
    #yerrdown = np.sqrt(A**2/Aref**4*yerrmed**2 + yerrdown**2/Aref**2)
    #A = A/Aref #+ Aref
    res = np.diff(wl)[0]

    # Remove points with too low error bars
    mflag = np.logical_or(yerrup < 0.2*np.median(yerrup), \
                                            yerrdown < 0.2*np.median(yerrdown))
    yerrup[mflag] = np.median(yerrup)
    yerrdown[mflag] = np.median(yerrdown)

    plt.figure()
    plt.errorbar(wl, A, yerr=[yerrup, yerrdown], fmt='ko', mfc='None', \
                    capsize=2)

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
        likelihood = np.zeros(len(tspot_)) - np.inf
        chi2r = np.copy(likelihood)

        if models == 'josh' and interpolate:
            mat = []
            # interpolate models, make finer grid
            #if not LM:
            for ti in tspot_:
                tt = format(ti, '2.2e')
                wl_ = pardict['spotmodels'][tt]['wl']
                mat.append(pardict['spotmodels'][tt]['spec'][pardict['muindex']])
            # interpolate with that and wl grid, and produce new spectrum
            zz = RectBivariateSpline(wl_, tspot_, np.array(mat).T)
            #else:
            #    # Interpolate in mu too
            #    mu_ = pardict['starmodel']['mus']
            #    for ti in tspot_:
            #        for mui, _ in enumerate(mu_):
            #            tt = format(ti, '2.2e')
            #            wl_ = pardict['spotmodels'][tt]['wl']
            #            mat.append(pardict['spotmodels'][tt]['spec'][mui])
            #    mat = np.array(mat).reshape(len(tspot_), len(mu_), len(wl_))
            #    zz = rgi((tspot_, mu_, wl_), mat)
            if pardict['tstar'] == 3500:
                tspot_ = np.arange(2300, pardict['tstar'] - 75, 25)
            else:
                tspot_ = np.arange(3600, pardict['tstar'] - 75, 25)
            chi2r = np.zeros(len(tspot_)) + np.inf
            likelihood = np.zeros(len(tspot_)) - np.inf

        print(plotname + stmod + '_' + pardict['observatory'])
        print('Fitting ' + stmod + ' models...')
        # Get spectra with starspot contamination and perform LM fit
        # First: use true temp to rescale error bars
        #ww, spec = combine_spectra(pardict, [pardict['tumbra']], 0.05, stmod, \
        #            wl, res=res, isplot=False)
        #specint = interp1d(ww, spec, bounds_error=False, \
        #                                            fill_value='extrapolate')
        #specA = specint(wl)
        #boundsm = ([-10, 10])
        #soln = least_squares(scalespec, [0.], \
        #            bounds=boundsm, args=(specA, A, yerrup, yerrdown))
        #chi2r_ = np.sum(scalespec(soln.x, specA, A, yerrup, yerrdown)) \
        #            /(len(A) - 2)
        #yerrup *= chi2r_**0.5
        #yerrdown *= chi2r_**0.5
        if not LM:
            for i, temp in enumerate(tspot_):
                if temp == pardict['tstar']:
                    continue
                if interpolate:
                    model = zz(wl_, temp)
                else:
                    model = []
                ww, spec = combine_spectra(pardict, [temp], 0.05, stmod, \
                        wl, res=res, isplot=False, model=model)
                specint = interp1d(ww, spec, bounds_error=False, \
                                fill_value='extrapolate')
                specA = specint(wl)
                if pardict['tstar'] == 5000:
                    boundsm = ([0., 0.01])
                elif pardict['tstar'] == 3500:
                    boundsm = ([0., 0.01])
                #boundsm = ([-10., 0.], [10., 10.])
                soln = least_squares(multspec, [1e-3], \
                            bounds=boundsm, args=(specA, A, yerrup, yerrdown))
                #boundsm = ([0., 10])
                #soln = least_squares(multspec, [1.], \
                #            bounds=boundsm, args=(specA, A, yerrup, yerrdown))
                #soln.x[0] = kr**2
                #if i % 4 == 0:
                #    plt.plot(wl, specA, label=str(int(temp)) + ' K')
                if i % 8 == 0:
                    plt.plot(wl, specA * soln.x[0], label=str(int(temp)) + ' K')
                # Save true value
                if temp == pardict['tumbra']:
                    trueval = np.copy(i)
                chi2 = np.sum(multspec(soln.x, specA, A, yerrup, yerrdown))
                #chi2 = np.sum(multspec(soln.x, specA, A, yerrup, yerrdown))
                    #        /(len(A) - 1)
                print(temp, 'K, spectrum scaling factor:', soln.x[0], \
                                'chi2r:', chi2/(len(A) - 1))
                #print(temp, 'K, chi2r:', chi2/(len(A) - 2))
                #plt.plot(wl, specA, label=str(int(temp)) + ' K')
                #chi2 = np.sum(scalespec([0.], specA, A, yerrup, yerrdown)) \
                #            /(len(A) - 1)
                chi2r[i] = chi2
                likelihood[i] = np.exp(-0.5*chi2)
                #likelihood[i] = np.prod(np.exp(-0.5*scalespec(soln.x, \
                #                specA, A, yerrup, yerrdown)))
                dict_results[pm][stmod][temp - pardict['tstar']] = likelihood[i]
        else:
            # In
            #mat = []
            # interpolate models, make finer grid
            #for ti in tspot_:
            #    tt = format(ti, '2.2e')
            #    wl_ = pardict['spotmodels'][tt]['wl']
            #    mat.append(pardict['spotmodels'][tt]['spec'][pardict['muindex']])
            ## interpolate with that and wl grid, and produce new spectrum
            #zz = RectBivariateSpline(wl_, tspot_, np.array(mat).T)

            boundsm = ([3600, 0.0, 0.0], [5000., 0.5, 1.0])
            soln = least_squares(spec_res, [4000., 1e-4, 1e-4], bounds=boundsm, \
                    args=(A, yerrup, yerrdown, wl, zz, pardict))
            print('Best sol:', soln.x)
            # Let's run an MCMC
            res = mcmc(soln, A, yerrup, yerrdown, wl, zz, pardict)
            plt.plot(wl, compute_deltaf_f(soln.x[0], soln.x[1], soln.x[2], \
                        wlobs, zz, pardict)
        plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
        #plt.ylabel(r'$\Delta f(\lambda)/\Delta f(\lambda_0)$', \
        plt.ylabel(r'$\Delta f(\lambda)$', \
                    fontsize=16)
        maxL = np.argmax(likelihood)
        print('L max =', likelihood[maxL], 'with Tspot =', \
                                    tspot_[maxL], 'K')
        plt.title('True value: ' + str(int(pardict['tumbra'])) + ' K' \
           + r', best fit: $T_\bullet=$' + str(int(tspot_[maxL])) + ' K', \
            fontsize=16)
        plt.xlim(wl.min() - 0.2, wl.max() + 0.2)

    x, y = [], []
    for jj in dict_results[pardict['magstar']][models].keys():
        x.append(jj)
        y.append(dict_results[pardict['magstar']][models][jj])
    x = np.array(x)
    y = np.array(y)#*np.diff(x)[0]
    C = np.sum(y)
    prob = y/C

    valmax = prob.max()
    xmax = prob.argmax()
    # Find the best value with chi2 + 1 for one deg of freedom (Lampton+1976)
    xmin = chi2r.argmin()
    # Divide all chi2 values by min chi2, so that min chi2 = len(A) - 2
    #chi2r /= chi2r.min()
    indices = range(len(x))
    deltachi2 = 2.28
    if xmin > 0 and xmin < len(x) - 1:
        xsigmaup = abs(chi2r[indices > xmin] - (chi2r.min() \
                        + deltachi2)).argmin() + xmin
        xsigmalow = abs(chi2r[indices < xmin] - (chi2r.min() \
                        + deltachi2)).argmin()
        xsigma = np.mean([abs(tspot_[trueval] - tspot_[xsigmaup]), \
                        abs(tspot_[trueval] - tspot_[xsigmalow])])
    elif xmin == 0:
        xsigmalow = 0
        xsigmaup = abs(chi2r[indices > xmin] - (chi2r.min() \
                        + deltachi2)).argmin() + xmin
        xsigma = abs(tspot_[trueval] - tspot_[xsigmaup])
    else:
        xsigmaup = len(x) - 1
        xsigmalow = abs(chi2r[indices < xmin] - (chi2r.min() \
                        + deltachi2)).argmin()
        xsigma = abs(tspot_[trueval] - tspot_[xsigmalow])
    xsigma = max([100., xsigma])
    #if trueval >= xmin:
    #    Tsigma = tspot_[xsigmaup] - tspot_[trueval]
    #else:
    #    Tsigma = tspot_[trueval] - tspot_[xsigmalow]
    #Tsigma = max([Tsigma, np.diff(tspot_)[0]])
    #xsigmaval = abs(x[xmin] - x[xsigma])
    # Plot the best spectrum
    #ww, spec = combine_spectra(pardict, [tspot_[xmin]], 0.05, stmod, \
    #        wl, res=res, isplot=False)
    #specint = interp1d(ww, spec, bounds_error=False, fill_value='extrapolate')
    #specA = specint(wl)
    #plt.plot(wl, specA + soln.x[0], label=str(int(tspot_[xmin])) + ' K')
    #plt.title(str(pardict['tstar']) + ' ' + str(pardict['tumbra']) + ' ' \
    #    + str(pardict['aumbra']) + ' ' + str(pardict['theta']) + ' ' \
    #        + str(pardict['magstar']))
    plt.legend(frameon=False)
    plt.savefig(plotname + stmod + '_' + pardict['observatory'] + '.pdf')
    plt.close('all')

    # Create PDF with the prob sampling
    pdf = stats.rv_discrete(a=x.min(), b=x.max(), values=(x, prob))
    Tmean = pdf.mean()
    #Tbest = tspot_[xmax]

    plt.figure()
    #print('*** Tbest ***', Tbest)
    #Tconf = [pdf.std(), pdf.std()]
    Tconf = pdf.interval(0.68) - Tmean
    #Tconf = pdf.interval(0.642) # % 64.2% confence interval
    dist = Tmean - (pardict['tumbra'] - pardict['tstar'])
    '''
    # Estimate ***confidence interval*** from chi2 distribution
    df = 2
    pdf = stats.chi2(df)
    #dist = abs(chi2r[trueval] - chi2r[xmin])
    # This is to compute the equivalent Gaussian "sigma"
    distchi2 = pdf.cdf(dist)
    nn = stats.norm()
    Tsigma = nn.interval(distchi2)[1]
    chi2_Trange = nn.interval(0.997)
    Trangedown = tspot_[abs(chi2r/chi2r.min() - chi2_Trange[0]).argmin()]
    Trangeup = tspot_[abs(chi2r/chi2r.min() - chi2_Trange[1]).argmin()]
    Trange = abs(Trangeup - Trangedown)
    '''

    #if trueval >= xmin:
    #    Tsigma *= -1
    #distchi2 = pdf.logsf(dist)
    #dist = pardict['tumbra'] - tspot_[xmin]
    #Tunc = dist
    #if (Tconf == pdf.median()).any():
    #    set_trace()
    #if dist > 0:
    #    Tsigma = abs(dist)/(Tconf[1] - pdf.median())
    #else:
    #    Tsigma = abs(dist)/(pdf.median() - Tconf[0])
    #Tsigma = pdf.std()
    #if Tsigma < np.diff(tspot_)[0]:
    #    Tsigma = np.diff(tspot_)[0]
    # Fit a Gaussian centered here
    #bbounds = ([0, x.min(), 50], [2*valmax, x.max(), 1000.])
    #fitgauss = lambda p, t, u: (u - gauss(p, t))**2
    #soln = least_squares(fitgauss, [valmax, x[prob.argmax()], 150.], \
    #            args=(x, prob), bounds=bbounds)
    #Tunc = soln.x[2]
    #dict_results[pardict['magstar']]['Tunc'] = [dist, Tsigma]
    #dict_results[pardict['magstar']]['Tunc'] \
    #        = [tspot_[xmin] - tspot_[trueval], Tsigma, distchi2, xsigma, \
    #            min(chi2r)/(len(A) - 2), chi2r, spotSNR, Trange]
    dict_results[pardict['magstar']]['Tunc'] \
            = [dist, Tconf, x, prob, 0, 0, spotSNR, chi2r]
    if dist < 0:
        zz = dist/Tconf[0]
    else:
        zz = dist/Tconf[1]
    #flag = ress < 2e-22
    sig3 = pdf.interval(0.997)
    plt.plot([sig3[0], sig3[0]], [0., prob.max() + 0.01], 'g')
    plt.plot([sig3[1], sig3[1]], [0., prob.max() + 0.01], 'g')
    plt.plot(x, prob, 'k.-', label=r'$\Delta T = $' + str(int(dist)) \
            + 'K\n' + str(np.round(zz, 1)) + r'$\sigma$')
    #plt.plot(x, chi2r/chi2r.min(), 'k.-', label=r'$\Delta T = $' \
    #    + str(tspot_[xmin] - tspot_[trueval]) + ' K\n$C=$' \
    #    + str(np.round(distchi2, 3)))
    #plt.plot(x, np.zeros(len(x)) + 11.62/chi2r.min() + 1., 'b--')

    #xTh = np.linspace(x.min(), x.max(), 1000)
    #plt.plot(xTh, gauss(soln.x, xTh), 'cyan')
    plt.plot([-pardict['tstar'] + pardict['tumbra'], \
            -pardict['tstar'] + pardict['tumbra']], [0., prob.max() + 0.01], 'r')
    plt.xlabel(r'$\Delta T_\mathrm{spot}$ [K]', fontsize=14)
    plt.ylabel('Probability likelihood', fontsize=14)
    #plt.ylabel('$\chi^2/\chi^2_\mathrm{min}$', fontsize=14)
    #plt.ylim(0.9, 5.)
    plt.title(str(pardict['tstar']) + ' ' + str(pardict['tumbra']) + ' ' \
        + str(pardict['aumbra']) + ' ' + str(pardict['theta']) + ' ' \
            + str(pardict['magstar']))
    plt.legend()
    plt.savefig(plotname + stmod + '_' + pardict['observatory'] + '_like.pdf')
    plt.close('all')
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

def scalespec2(x, spec, y, yerrup, yerrdown):
    '''
    Distance between model spectrum and data (with unequal uncertainties).
    '''
    res = np.zeros(len(y))
    flag = spec*x[1] + x[0] >= y
    res[flag] = (spec*x[1] + x[0] - y)[flag]**2/yerrup[flag]**2
    res[~flag] = (spec*x[1] + x[0] - y)[~flag]**2/yerrdown[~flag]**2

    return res

def multspec(x, spec, y, yerrup, yerrdown):
    '''
    Distance between model spectrum and data (with unequal uncertainties).
    '''
    res = np.zeros(len(y))
    flag = spec * x[0] >= y
    res[flag] = (spec * x[0] - y)[flag]**2/yerrup[flag]**2
    res[~flag] = (spec * x[0] - y)[~flag]**2/yerrdown[~flag]**2

    return res

def combine_spectra(pardict, tspot, ffact, stmodel, wnew, \
                                        res=100., isplot=False, model=[]):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19
    Use Phoenix models in pysynphot (submitted version)
    Use Josh's models (revision 1), and a model can be given via "model".

    Input
    -----
    tspot: list (of effective temperatures)
    wl: array to degrade the model spectrum. All wavelengts are in Angrstoms
    models: input model (from interpolation)
    '''

    if not pardict['spotted_starmodel']:
        if stmodel == 'phoenix':
            wave = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                    pardict['loggstar']).wave
            star = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                    pardict['loggstar']).flux
        elif stmodel == 'josh':
            # Both intensity and flux are needed
            i_star = pardict['starmodel']['spec'][pardict['muindex']]
            wave = pardict['starmodel']['wl']
            f_star = np.sum(pardict['starmodel']['spec'], axis=0)
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
    #i_star = integ_filter(wth, fth, wave, i_star)
    #f_star = integ_filter(wth, fth, wave, f_star)
    #fflag = wth < 60000.
    #wth = wth[fflag]
    #fth = fth[fflag]
    #star = star[fflag]
    #rebstar = degrade_spec(i_star, wth, wnew)
    for i in tspot:
        if stmodel == 'ck04models' or stmodel == 'phoenix':
            wls = pysynphot.Icat(stmodel, i, 0.0, \
                                            pardict['loggstar'] - 0.5).wave
            spot = pysynphot.Icat(stmodel, i, 0.0, \
                                            pardict['loggstar'] - 0.5).flux
        elif stmodel == 'josh':
            tspot_ = format(i, '2.2e')
            wls = np.copy(wave)#pardict['spotmodels'][tspot_]['wl']
            if np.shape(model) == (0.,):
                i_spot = pardict['spotmodels'][tspot_]['spec'][pardict['muindex']]
            else:
                i_spot = np.hstack(model)
            f_spot = np.sum(pardict['spotmodels'][tspot_]['spec'], axis=0)
        #fflagu = wls < 60000.
        #wls = wls[fflagu]
        #spot = spot[fflagu]
        #i_spot = integ_filter(wth, fth, wls, i_spot)
        #f_spot = integ_filter(wth, fth, wls, f_spot)
        #rebnew = degrade_spec(spot, wth, wnew)

        idiff = integ_filter(wth, fth, wls, i_star - i_spot)
        deltaf_f = (i_star - i_spot)/(f_star*(1. - ffact) + ffact*f_spot)
        f_star_spot = integ_filter(wth, fth, wls, deltaf_f)
        deltaf_f = degrade_spec(idiff/f_star_spot, wth, wnew)

        # As Sing+2011
        if pardict['instrument'] == 'NIRSpec_Prism':
            wref = np.logical_and(wlminPrism < wnew, wnew < wlmaxPrism)
        elif pardict['instrument'] == 'NIRCam':
            wref = np.logical_and(wlminNIRCam < wnew, wnew < wlmaxNIRCam)
        #spotref = rebnew[wref]
        #starref = rebstar[wref]
        #fref = 1. - np.mean(spotref/starref)
        #rise = (1. - rebnew/rebstar)#/fref #+ Aref
        #rise = degrade_spec(rise, wth, wnew)
        rise = deltaf_f[wref]
        if isplot:
            plt.plot(wnew, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
                        + str(i) + 'K')
    if isplot:
        plt.title('Stellar spectrum: ' + str(pardict['tstar']) + ' K', \
                    fontsize=16)
        plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
        plt.ylabel('Flux rise [Relative values]', fontsize=16)

    flag = np.logical_or(np.isnan(wnew), np.isnan(rise))
    set_trace()
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

def compare_contrast_spectra(tstar, loggstar, tspot, modgrid, mu=-1, \
                wlrefmin=1.0, wlrefmax=1.2):
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
        plt.figure(2)
        plt.plot(wl*1e-4, starflux, label=str(tstar) + ' K')
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
            if ti != 2900.:
                filename = josh_grid_folder + 'starspots.teff=' \
                        + tspot_ + '.logg=' + loggspot_ + '.z=0.0.irfout.csv'
                g = np.genfromtxt(filename, delimiter=',')
                wl2 = np.copy(wl)
            else:
                filename = josh_grid_folder + 'starspots.teff=' \
                   + tspot_ + '.logg=' + loggspot_ + '.z=0.0.irfout3.FIXED.csv'
                g = np.genfromtxt(filename, delimiter=',')
                wl2 = np.array([g[i][0] for i in range(2, len(g))])
            spotflux = np.array([g[i][mu] for i in range(2, len(g))])
            plt.figure(2)
            plt.plot(wl2*1e-4, spotflux, label=str(ti) + ' K')
            plt.ylabel(r'Flux [some units]', fontsize=14)
        if ti != 2900.:
            spotflux = degrade_spec(spotflux, wl, wnew)
        else:
            spotflux = degrade_spec(spotflux, wl2, wnew)
        wref = np.logical_and(wlrefmin < wnew, wnew < wlrefmax)
        spotref = spotflux[wref]
        starref = starflux[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - spotflux/starflux)/fref
                            #+ np.mean(spotflux[wref]/starflux[wref])
        bspot = bb.blackbody_lambda(wl, ti)
        bstar = bb.blackbody_lambda(wl, tstar)
        wref = np.logical_and(46000. < wl, wl < 50000)
        cref = 1. - np.mean(bspot[wref]/bstar[wref])
        contrast = (1. - bspot/bstar)/cref \
                                + np.mean(spotref/starref)
        #                        + np.mean(bspot[wref]/bstar[wref]).value
        plt.figure(1)
        plt.plot(wnew, rise, label=str(ti) + ' K')
        #if ti == tspot[-1]:
        #    plt.plot(wl*1e-4, contrast, 'k', alpha=0.5, \
        #                                label='Corresponding black body curves')
        #else:
        #    plt.plot(wl*1e-4, contrast, 'k', alpha=0.5)
        plt.ylabel(r'Brightness contrast at $\mu=1$', fontsize=14)
    for j in [1, 2]:
        plt.figure(j)
        plt.xlim(0.6, 5.2)
        #plt.ylim(0., 0.95)
        plt.xlabel(r'Wavelength [$\mu$m]', fontsize=14)
        plt.title('$T_\star=$ ' + str(tstar) + ' K', fontsize=16)
        leg = plt.legend(frameon=False)
        #vp = leg._legend_box._children[-1]._children[0]
        #for c in vp._children:
        #    c._children.reverse()
        #    vp.align="right"
    plt.show()
    plt.figure(1)
    plt.savefig('/home/giovanni/Projects/jwst_spots/contrast_model_' \
                 + str(tstar) + '_mu_' + str(wlrefmin) + '-' + str(wlrefmax) \
                + '.pdf')

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

def compare_specific_intensity(tstar, tcontrast, loggstar, muindex):

    dict_starmodel, dict_spotmodels \
                        = ingest_stellarspectra(tstar, tcontrast, loggstar)

    star = dict_starmodel['spec'][muindex]
    wl = dict_starmodel['wl']
    for tspot_ in tstar + tcontrast:
        tspot = format(tspot_, '2.2e')
        spot = dict_spotmodels[tspot]['spec'][muindex]
        plt.plot(wl, 1. - spot/star)

    plt.show()
    set_trace()
    return

def compute_deltaf_f(tspot, ffact, beta, wlobs, zz, pardict):
    '''
    Compute normalized flux variation during starspot occultation.

    Input
    -----
    ffact: starspot filling factor
    beta: fraction of planetary surface occulting the starspot
    '''

    # Both intensity and flux are needed
    muspot = pardict['starmodel']['mus'][pardict['muindex']]
    i_star = pardict['starmodel']['spec'][pardict['muindex']]
    wave = pardict['starmodel']['wl']
    f_star = 2.*np.sum(np.transpose(pardict['starmodel']['spec']) \
        *pardict['starmodel']['mus'], axis=1) \
        - 2.*np.transpose(pardict['starmodel']['spec'][pardict['muindex']]) \
                *muspot \
        + pardict['starmodel']['spec'][pardict['muindex']]*muspot*(2.-ffact)

    # Spot specific intensity from interpolation
    #pts = np.array([[np.zeros(len(wave)) + tspot], \
    #                    [np.zeros(len(wave)) + pardict['muindex']], [wave]])
    i_spot = np.hstack(zz(wave, tspot))
    f_spot = i_spot*muspot*ffact

    #i_spot = zz(pts.T)
    #tspot_ = format(tspot, '2.2e')
    wls = np.copy(wave)#pardict['spotmodels'][tspot_]['wl']
    # Compute spot flux from interpolation
    #f_spot = np.sum(pardict['spotmodels'][tspot_]['spec'], axis=0)
    #f_spot = np.zeros(len(wave))
    #for mui in mus:
    #    pts = np.array([[np.zeros(len(wave)) + tspot], \
    #                    [np.zeros(len(wave)) + mui], [wave]])
    #    f_spot = zz(pts.T)

    #f_spot = np.sum(zz)
    if pardict['instrument'] == 'NIRSpec_Prism':
        wth, fth = np.loadtxt(thrfile4, unpack=True)
        wth*= 1e4
    elif pardict['instrument'] == 'NIRCam':
        wth1, fth1 = np.loadtxt(thrfile1, unpack=True)
        wth2, fth2 = np.loadtxt(thrfile2, unpack=True)
        wth3, fth3 = np.loadtxt(thrfile3, unpack=True)
        wth = np.concatenate((wth1, wth2, wth3))
        fth = np.concatenate((fth1, fth2, fth3))

    idiff = integ_filter(wth, fth, wls, i_star - i_spot)
    #f_star_spot = f_star*(1. - ffact) + ffact*f_spot
    # Multiply by common factor
    if pardict['tstar'] == 5000:
        Rstar = 0.8*6.957e8
    elif pardict['tstar'] == 3500:
        Rstar = 0.45*6.957e8
    factor = 2.*np.pi*Rstar**2/len(pardict['starmodel']['mus'])
    f_star_spot = integ_filter(wth, fth, wls, (f_star + f_spot)*factor)
    # Only keep values within filters curves
    wlfl = np.logical_and(wth >= min(wave), wth <= max(wave))
    idiff = idiff[wlfl]
    f_star_spot = f_star_spot[wlfl]
    wth = wth[wlfl]
    #wth =
    #modint = interp1d(wl, f_star_spot)
    #deltaf_f = modint(w)
    deltaf_f = degrade_spec(idiff/f_star_spot, wth, wlobs)
    deltaf_f *= beta*np.pi*(kr*Rstar)**2

    return deltaf_f

def spec_res(par, spec, yerrup, yerrdown, wlobs, zz, pardict):
    '''
    Compute squared residuals from model spectrum and data.
    '''

    tspot, ffact, beta = par
    mod = compute_deltaf_f(tspot, ffact, beta, wlobs, zz, pardict)
    res = np.zeros(len(mod))
    flag = spec >= mod
    res[flag] = (spec - mod)[flag]**2/yerrup[flag]**2
    res[~flag] = (spec - mod)[~flag]**2/yerrdown[~flag]**2

    return res

def lnprior(par):

    tspot, ffact, beta = par
    if np.logical_or.reduce((tspot < 3600., tspot > 5000., ffact <= 1e-6, \
                ffact >= 0.6, beta <= 0, beta > 1)):
                return -np.inf
    else:
        lnp_ffact = lnp_jeffreys(ffact, 0.6, 1e-6)
        return lnp_ffact

def lnp_jeffreys(val, valmax, valmin):
    '''
    Jeffreys prior
    '''
    return np.log(1./(val*np.log(valmax/valmin)))

def lnprob(par, spec, yerrup, yerrdown, wlobs, zz, pardict):
    '''
    Compute posterior probability for spectrum fit.
    '''

    lp = lnprior(par)
    if not np.isfinite(lp):
        return -np.inf
    else:
        chi2 = np.sum(spec_res(par, spec, yerrup, yerrdown, wlobs, zz, pardict))
        return chi2 + lp

def mcmc(soln, A, yerrup, yerrdown, wl, zz, pardict):
    '''
    Get posterior distribution for the starspot configuration.
    '''

    # MCMC starting about the optimized solution
    initial = np.array(soln.x)
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=([A, yerrup, yerrdown, wl, zz, pardict]), threads=8)

    # Variation around LM solution
    p0 = initial + 0.01*(np.random.randn(nwalkers, ndim))*initial
                #+ 1e-6*(np.random.randn(nwalkers, ndim))

    # Check condition number (must be < 1e8 to maximise walker linear
    # independence). Problems might be caued by LM results = 0
    cond = np.linalg.cond(p0)
    while cond >= 1e8:
        p0 += 1e-4*(np.random.randn(nwalkers, ndim))
        cond = np.linalg.cond(p0)

    print("Running burn-in...")
    nsteps = 128
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width+1)*float(i)/nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(width - n)))
    sys.stdout.write("\n")
    p0, lp, _ = result
    sampler.reset()

    print("Running production...")
    nsteps = 128
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps, thin=10)):
        n = int((width+1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(width - n)))
    sys.stdout.write("\n")
    pfin, lpfin, _ = result

    # Merge in single chains
    samples = sampler.flatchain
    lnL = sampler.flatlnprobability

    best_sol = samples[lnL.argmax()]
    try:
        acor_time = integrated_time(samples, c=10)
        acor_multiples = np.shape(samples)[0]/acor_time
        print('Length chains:', np.shape(samples)[0])
        print('Autocorrelation multiples:', acor_multiples)
        print('Integrated autocorrelation time')
        for jpar in np.arange(np.shape(samples)[1]):
            IAT = emcee.autocorr.integrated_time(samples[:, j])
            print('IAT multiples for parameter', jpar, ':', \
                                np.shape(samples)[0]/IAT)
    except:
        print('The chain is too short')
        pass

    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

    percentiles = [np.percentile(samples[:,i],[4.55, 15.9, 50, 84.1, 95.45]) \
                        for i in np.arange(np.shape(samples)[1])]

    titles = [r'$T_\bullet$', '$\delta$', r'$\beta$']
    truths = [4000, 0.01, 0.01]
    cornerplot.cornerplot(samples, titles, truths, \
            pardict['chains_folder'] + '/cornerspec.pdf', ranges=None)
    set_trace()
    # Save chains
    #fout = open(diz['chains_folder'] + '/chains_' + str(ind) \
    #                                                + '.pickle', 'wb')
    #chains_save = {}
    #chains_save['wl'] = wl
    #chains_save['LM'] = soln.x
    #chains_save['Burn_in'] = [p0, lp]
    #chains_save['Chains'] = samples
    ##chains_save['Starspot_size'] = size
    #chains_save['Mean_acceptance_fraction'] \
    #                                = np.mean(sampler.acceptance_fraction)
    #chains_save['Autocorrelation_multiples'] = acor_time
    #chains_save["Percentiles"] = [percentiles]
    #chains_save["ln_L"] = lnL
    #pickle.dump(chains_save, fout)
    #fout.close()

    #fout = open(diz['chains_folder'] + '/chains_best_' \
    #            + str(ind) + '.p', 'wb')
    #pickle.dump(best_sol, fout)
    #fout.close()

    return
