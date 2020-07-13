# Compare spot fits with stellar models.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize
from scipy.signal import medfilt
from scipy.integrate import trapz
from scipy.sparse import csr_matrix
from scipy import stats
import get_uncertainties as getunc
import simulate_transit
import rebin
import pysynphot
import glob
import pickle
from pdb import set_trace
from astropy.io import fits
import cornerplot
import emcee
from pysynphot import observation
from pysynphot import spectrum

modelsfolder = '/home/giovanni/Dropbox/Shelf/stellar_models/phxinten/HiRes/'

plt.ioff()

def read_res(pardict, instrument, plotname, resfile, models, fittype='grid', \
            resol=10):
    '''
    Read in the chains files and initialize arrays with spot properties.

    wlwin: used to exlude parts of the spectra
    '''

    #ldfile = open(pardict['project_folder'] \
    #    + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
    #                    + 'K/' + 'LDcoeffs_prism.pic', 'rb')
    #ldd = pickle.load(ldfile)
    if pardict['instrument'] == 'NIRSpec_Prism/':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam/':
        ldendname = 'nircam'
    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    xobs = pickle.load(ldd)[1][0]
    ldd.close()
    #xobs -= 0.5
    flag = xobs < 6
    #flag = np.logical_or.reduce((xobs < 1.6, np.logical_and(2.0 < xobs, \
    #            xobs < 3.0), xobs > 4.0))
    expchan = len(xobs[flag])

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
        A.append(res['sol'].x[-2])
        sigma.append(res['sol'].x[-1])
        yerrup.append(res['1sigma_unc'][-2])
        yerrdown.append(res['1sigma_unc'][-2])
        ffopen.close()
        kr.append(res['sol'].x[0])
        krunc.append(res['1sigma_unc'][0])

    # Compare with initial simulation
    #plt.figure(42)
    #kr = np.array(kr)
    #krunc = np.array(krunc)
    #plt.errorbar(wl, kr**2, yerr=2.*kr*krunc, fmt='o', label='fit')
    ##plt.errorbar(xobs, yobs, yerr=yobs_err, fmt='o', label='simulated')
    #plt.legend()
    #plt.show()
    #set_trace()
    #plt.savefig(pardict['chains_folder'] + 'compare_kr.pdf')
    #plt.close('all')
    # Group stuff together
    #A = np.vstack(A)
    #x0 = np.vstack(x0)
    #sigma = np.vstack(sigma)
    wl = np.array(wl)
    if pardict['instrument'] == 'NIRCam/':
        flag = wl < 6.
        #flag = np.logical_or.reduce((wl < 1.6, np.logical_and(2.0 < wl, \
        #            wl < 3.0), wl > 4.0))
    elif pardict['instrument'] == 'NIRSpec_Prism/':
        #flag = np.logical_or.reduce((wl < 1.6, np.logical_and(2.0 < wl, \
        #           wl < 3.0), wl > 4.2))
        #flag = np.logical_or(wl < 3.2, wl > 4.2)
        flag = wl < 6.
    flag[1] = False
    wl = wl[flag]
    A = np.array(A)[flag]
    sigma = np.array(sigma)[flag]
    yerrup = np.array(yerrup)[flag]
    yerrdown = np.array(yerrdown)[flag]
    if pardict['instrument'] == 'NIRSpec_Prism/':
        wref = np.logical_and(0.6 < wl, wl < 1.0)
    elif pardict['instrument'] == 'NIRCam/':
        wref = np.logical_and(1.0 < wl, wl < 1.8)
    Aref = np.mean(A[wref])
    sigma /= Aref
    yerrup /= Aref
    yerrdown /= Aref
    A /= Aref
    #plt.scatter(wl, A)
    #plt.show()
    #set_trace()
    ##yerrup /= A.min()
    #yerrdown /= A.min()
    #A /= A.min()
    # Use max uncertainty between yerrup and yerrdown
    yerrfit = []
    for j in np.arange(len(yerrup)):
        yerrfit.append(max([yerrup[j], yerrdown[j]]))
    yerrfit = np.array(yerrfit)
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
            tspot_ = np.arange(2300, 3500, 50)
        else:
            tspot_ = np.arange(3300, 4900, 50)
        tspot_ = tspot_[tspot_ != pardict['tstar']]
        likelihood = np.zeros(len(tspot_)) + np.inf
        plt.figure()
        plt.errorbar(wl, A, yerr=[yerrup, yerrdown], fmt='ko', mfc='None')
        plt.title('$A$', fontsize=16)
        print('Fitting ' + stmod + ' models...')
        # Get spectra with starspot contamination and perform LM fit
        if fittype == 'grid':
            for i, temp in enumerate(tspot_):
                if temp == pardict['tstar']:
                    continue
                ww, spec = combine_spectra(pardict, [temp], 0.05, stmod, \
                        wl, res=res, isplot=False)
                #ww/= 1e4
                #spec*= A.max()
                specint = interp1d(ww, spec)
                specA = specint(wl)
                #boundsm = ([1e-3, -10000], [np.inf, 10000])
                boundsm = ([-10, 10])
                soln = least_squares(scalespec, [0.], \
                            bounds=boundsm, args=(specA, A, yerrfit))
                #scale_unc = getunc.unc_jac(soln.fun, soln.jac, len(yerrfit) - 1)
                print(temp, 'K Spectrum scaling factor:', soln.x[0])#, '+/-', \
                #                    scale_unc[0])
                #plt.plot(ww, soln.x[0]*spec, label=str(temp), alpha=.5)
                #if pardict['instrument'] == 'NIRCam/':
                #    flag = np.logical_or.reduce((ww < 1.6, np.logical_and(2.0 < ww, \
                #                    ww < 3.0), ww > 4.0))
                #elif pardict['instrument'] == 'NIRSpec_Prism/':
                #    flagf = np.logical_or(ww < 3.0, ww > 4.2)
                #soln.x[0] = 1
                plt.plot(ww, specA + soln.x[0], label=str(temp), alpha=.5)
                chi2 = np.sum((A - specA - soln.x[0])**2 \
                            /yerrfit**2)/(len(yerrfit) - 1)
                likelihood[i] = np.exp(-0.5*chi2)
                dict_results[pm][stmod][temp - pardict['tstar']] = likelihood[i]
        elif fittype == 'mcmc':
            boundsm = ([3500., 0.], [4900., np.inf])
            opt = {}
            opt['isplot'] = Falsesoln.x
            soln = least_squares(stmodfit, [4000., 1.], \
                        bounds=boundsm, args=(wl, A, yerrfit, pardict, \
                        res, stmod), kwargs=opt)
            #ftol = 1e-9
            #soln = minimize(stmodfit, [4000., 1.], jac=False, method='L-BFGS-B', \
            #            args=(wl, A, yerrfit, pardict, res, stmod, isplot=False, \
            #            bounds=boundsm, ftol=ftol)
            #uncsol = get_uncertainties.unc_minimization_lbfgsb(soln, ftol=ftol)
            #print('Spectrum fit:', soln.x)
            initial = np.array(soln.x)
            ndim, nwalkers = len(initial), 10
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
                    args=([wl, A, sigma, pardict, stmod, res]), threads=8, live_dangerously=False)

            print("Running burn-in...")
            p0 = initial + 100 * np.random.randn(nwalkers, ndim)
            p0[:, 1] /= 100.
            #p0 = np.zeros((10, 2))
            #for i in np.arange(10):
            #    p0[i, :] = [3700 + i*1, 1. + 0.1*i]
            #set_trace()
            sampler.run_mcmc(p0, 500)
            samples = sampler.flatchain
            lnL = sampler.flatlnprobability
            titles = [r'$T_\mathrm{spot}$', 'c']
            truths = [pardict['tumbra'], None]
            cornerplot.cornerplot(samples, titles, truths, \
                    pardict['chains_folder'] + '/corner.pdf')
            #solmodel = combine_spectra(pardict, [soln.x[0]], 0.05, stmod, \
            #            res=res, isplot=True)#*soln.x[1]
            #plt.plot(ww, solmodel)
            #plt.plot(ww, soln.x[0]*spec + soln.x[1], label=str(temp), alpha=.5)
            best_sol = samples[lnL.argmax()]
            chi2red[i] = np.sum((A - solmodel)**2/yerrfit**2) \
                            /(len(yerrfit) - 1)
            percentiles = [np.percentile(samples[:,i],[4.55, 15.9, 50, \
                        84.1, 95.45]) for i in np.arange(np.shape(samples)[1])]
            # Save chains
            fout = open(chains_folder_ch + 'chains.pic', 'wb')
            chains_save = {}
            chains_save['max ML']                   = soln.x
            chains_save['chains']                   = samples
            chains_save["Percentiles"]              = [percentiles]
            chains_save["ln_L"]                     = lnL
            pickle.dump(chains_save, fout)
            fout.close()

            dict_results[pm][stmod][temp - pardict['tstar']] = chi2red[i]

        #plt.legend(frameon=False, loc='best', fontsize=16)
        plt.xlabel('Wavelength [$\mu m$]', fontsize=16)
        plt.ylabel('Transit depth rise [ppm]', fontsize=16)
        #chi2min = np.argmin(chi2)
        maxL = np.argmax(likelihood)
        #print('chi2 min =', chi2[chi2min], 'with Tspot =', \
        #                            tspot_[chi2min], 'K')
        print('L max =', likelihood[maxL], 'with Tspot =', \
                                    tspot_[maxL], 'K')
        plt.title(stmod \
           #+ r', $\min (\tilde{\chi}^2)=$' \
           #+ str(np.round(chi2red[chi2min], 2)) \
           + r', $T_\mathrm{spot}=$' + str(tspot_[maxL]) + ' K', fontsize=16)
        plt.xlim(wl.min() - 0.2, wl.max() + 0.2)
        #plt.ylim(0., 6000)
        plt.savefig(plotname + stmod + '_' + instrument + '.pdf')
        plt.close('all')

    plt.figure()
    x, y = [], []
    for jj in dict_results[pardict['magstar']][models[0]].keys():
        x.append(jj)
        y.append(dict_results[pardict['magstar']][models[0]][jj])
    x = np.array(x)
    y = np.array(y)
    A = np.sum(y)
    #A = trapz(y, x)

    prob = y/A
    valmax = prob.max()
    xmax = prob.argmax()

    # Create PDF with the prob sampling
    pdf = stats.rv_discrete(a=x.min(), b=x.max(), values=(x, prob))
    Tmean = pdf.mean()
    Tconf = [pdf.std(), pdf.std()]
    #Tconf= pdf.interval(0.642) # % 64.2% confidence interval
    dist = pardict['tumbra'] - pardict['tstar'] - pdf.median()
    if dist > 0:
        Tunc = abs(dist)/(Tconf[1] - pdf.mean())
    else:
        Tunc = abs(dist)/(pdf.mean() - Tconf[0])

    # Fit a Gaussian centered here
    #bbounds = ([0, x.min(), 50], [2*valmax, x.max(), 1000.])
    #fitgauss = lambda p, t, u: (u - gauss(p, t))**2
    #soln = least_squares(fitgauss, [valmax, x[prob.argmax()], 150.], \
    #            args=(x, prob), bounds=bbounds)
    #Tunc = soln.x[2]
    dict_results[pardict['magstar']]['Tunc'] = [abs(Tunc), pdf.std()]
    #ress = dict_results[10.5]['phoenix'].values()
    #flag = ress < 2e-22
    plt.plot(x, prob, 'ko-')# bins=15)
    #xTh = np.linspace(x.min(), x.max(), 1000)
    #plt.plot(xTh, gauss(soln.x, xTh), 'cyan')
    plt.plot([-pardict['tstar'] + pardict['tumbra'], \
            -pardict['tstar'] + pardict['tumbra']], [prob.min(), \
                    prob.max()], 'r--')
    plt.xlabel(r'$\Delta T_\mathrm{spot}$ [K]', fontsize=16)
    plt.ylabel('Probability likelihood', fontsize=16)
    #plt.ylim(0, 10000)
    plt.show()

    plt.savefig(plotname + stmod + '_' + instrument + '_like.pdf')
    if fittype == 'grid':
        fresults = open(resfile + stmod + '_grid.pic', 'wb')
    elif fittype == 'LM_pysynphot':
        fresults = open(resfile + stmod + '_LMfit.pic', 'wb')
    pickle.dump(dict_results, fresults)
    fresults.close()

    return np.array(wl), np.array(A), np.array(x0), np.array(sigma)

def gauss(par, x):
    '''
    Par is defined as [A, x0, sigma]
    '''
    A, x0, sigma = par
    return A*np.exp(-0.5*(x - x0)**2/sigma**2)


def lnprob(pars, x, y, yerr, pardict, stmod, res):

    if np.logical_or(pars[0] < 2000, pars[0] >= pardict['tstar']):
        return -np.inf
    else:
        return lnlike(pars, x, y, yerr, pardict, stmod, res)

def stmodfit(x, wl, y, yerr, pardict, res, stmod, isplot=False):
    '''
    Include the starspot Teff in the fit.
    '''
    temp = x[0]
    ww, spec = combine_spectra(pardict, [temp], 0.05, stmod, wl, res=res, \
                isplot=isplot)
    #ww/= 1e4
    #spec*= 1e3
    specint = interp1d(ww, spec)
    specA = specint(wl)
    return (x[1]*specA - y)**2/yerr**2

def scalespec(x, spec, y, yerr):
    '''
    Distance between model spectrum and data.
    '''
    #return (x[0]*spec - y)**2/yerr**2
    return (spec + x[0] - y)**2/yerr**2

def combine_spectra(pardict, tspot, ffact, stmodel, wnew, \
                                                    res=100., isplot=False):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19

    Input
    -----
    tspot: list (of effective temperatures)
    wl: array to degrade the model spectrum
    '''

    # This is in Angstroms
    if stmodel == 'ck04models' or stmodel == 'phoenix':
        wave = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).wave
        star = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).flux
    #wnew = wave.copy()*1e-4
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
    #kern = 5
    #kern = int(len(wave)/len(kr))
    #if kern % 2 == 0:
    #    kern += 1
    #wnew = wave[::kern]
    #rebstar = medfilt(star, kernel_size=kern)[::kern]
    #rebstar[rebstar < 1e-6] = 0.
    rebstar = degrade_spec(star, wave, wnew)

    # Increase errors (based on ETC?)
    #errstar += rebstar*abs(np.random.normal(loc=0., scale=5.5e-3, \
    #                        size=len(errstar)))
    #plt.errorbar(wnew/10., rebstar, yerr=errstar, label=str(tstar))

    for i in tspot:
        if stmodel == 'ck04models' or stmodel == 'phoenix':
            spot = pysynphot.Icat(stmodel, i, 0.0, \
                                            pardict['loggstar'] - 0.5).flux
        #elif stmodel == 'phoenix':
        #    spot = fits.open(modelsfolder + 'lte0' + str(i) \
        #            + '-' + '{:3.2f}'.format(pardict['loggstar'] - 0.5) \
        #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
        specnew = spot#(1 - ffact)*star + ffact*spot
        #rebnew, errnew = rebin.rebin_spectrum(specnew, wave, wnew)
        #rebnew = medfilt(specnew, kernel_size=kern)[::kern]
        #rebnew[rebnew < 1e-6] = 0.
        rebnew = degrade_spec(specnew, wave, wnew)
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
        if pardict['instrument'] == 'NIRSpec_Prism/':
            wref = np.logical_and(0.6 < wnew, wnew < 1.0)
        elif pardict['instrument'] == 'NIRCam/':
            wref = np.logical_and(1.0 < wnew, wnew < 1.8)
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

        if isplot:
            #plt.plot(wnew/1e4, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
            #            + str(i) + 'K')
            plt.plot(wnew, rise, label=r'$T_{\mathrm{eff}, \bullet}$ = '\
                        + str(i) + 'K')
        #plt.ylim(0, 5.)
        #plt.show()
        #set_trace()
        #return
    if isplot:
        #plt.legend(frameon=False, fontsize=14)
        plt.title('Stellar spectrum: ' + str(pardict['tstar']) + ' K', \
                    fontsize=16)
        #plt.plot([0.719, 0.719], [0.5, 9.5], 'k--')
        plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
        plt.ylabel('Flux rise [Relative values]', fontsize=16)
        #plt.xlim(0.6, 5)
        #plt.ylim(1500, 6500)
        #plt.show()
    flag = np.logical_or(np.isnan(wnew), np.isnan(rise))

    return wnew[~flag], rise[~flag]

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

def compare_contrast_spectra(tstar, loggstar, tspot):

    starmod = pysynphot.Icat('phoenix', tstar, 0.0, loggstar)
    starflux = starmod.flux
    wl = starmod.wave#/1e4
    #wl = fits.open(modelsfolder \
    #                +'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data/1e4
    #starflux = fits.open(modelsfolder + 'lte0' + str(tstar) \
    #            + '-' + '{:3.2f}'.format(loggstar) \
    #            + '-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[0].data
    wnew = np.linspace(0.6, 5.3, 35)
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
        wref = np.logical_and(0.6 < wnew, wnew < 1.0)
        spotref = spotflux[wref]
        starref = starflux[wref]
        fref = 1. - np.mean(spotref/starref)
        rise = (1. - spotflux/starflux)/fref
        plt.plot(wnew, rise, label=str(ti))
    plt.xlim(0.6, 6)
    plt.ylim(0, 2)
    plt.legend()
    plt.show()

    return

def lnlike(pars, x, y, yerr, pardict, stmod, res):

    model = combine_spectra(pardict, [pars[0]], 0.05, stmod, x, \
                res=res, isplot=False)
    sigma = np.mean(yerr)
    lnL = -0.5*len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) \
                - 0.5*chi2(model, y, yerr)
    #print(lnL)
    #plt.plot(t, model)
    #plt.errorbar(t, y, yerr=yerr, fmt='o')
    #plt.show()
    #set_trace()
    return lnL

def chi2(model, y, yerr):
    return np.sum((model - y)**2/yerr**2)

def prior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i] = cube[i]*1500. + 3500.

def loglike(cube, ndim, nparams):
    model = combine_spectra(pardict, [cube[0]], 0.05, stmod, wl, \
                res=res, isplot=False)
    loglikelihood=-0.5*np.sum((y_meas-y_binned)**2/err**2)
