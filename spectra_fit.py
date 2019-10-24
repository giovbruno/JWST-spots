# Compare spot fits with stellar models.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import get_uncertainties as getunc
import rebin
import pysynphot
import glob
import pickle
from pdb import set_trace

def read_res(pardict, ind, instrument, plotname, resfile):
    '''
    Read in the chains files and initialize arrays with spot properties.
    '''
    wl, A, x0, sigma = [], [], [], []
    for ff in glob.glob(pardict['chains_folder'] + 'chains*pickle'):
        res = pickle.load(open(ff, 'rb'))
        perc = res['Percentiles'][0]
        wl.append(res['wl'])
        A.append(perc[-3]*1e6)
        x0.append(perc[-2]*1e6)
        sigma.append(perc[-1])
    # Group stuff together
    A = np.vstack(A)
    x0 = np.vstack(x0)
    sigma = np.vstack(sigma)
    yerrup, yerrdown = [], []
    for i in np.arange(len(A[:, 2])):
        #if A[i, 1] > 0:
        #    yerr.append(max(abs(A[i, 2] - A[i, 1]), abs(A[i, 3] - A[i, 2])))
        #else:
        #    yerr.append(abs(A[i, 3] - A[i, 2]))
        yerrup.append(A[i, 3] - A[i, 2])
        yerrdown.append(A[i, 2] - A[i, 1])
    #yerr = np.array(yerr)
    plt.figure()
    plt.errorbar(wl, sigma[:, 2], yerr=sigma[:, 2] - sigma[:, 0], fmt='ko')
    plt.title('$\sigma$', fontsize=16)
    plt.xlabel('Channel', fontsize=16)
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
    for stmod in ['k93models', 'phoenix']:#, 'ck04models']:
        dict_results[stmod] = {}
        tspot_ = np.arange(3500, 6000, 100)
        chi2red = np.zeros(len(tspot_)) + np.inf
        plt.figure()
        plt.errorbar(wl, A[:, 2], yerr=[yerrup, yerrdown], fmt='ko')
        plt.title('$A$', fontsize=16)
        print('Fitting ' + stmod + ' models...')
        # Get spectra with starspot contamination and perform LM fit
        for i, temp in enumerate(tspot_):
            if temp == pardict['tstar']:
                continue
            ww, spec = combine_spectra(pardict, [temp], 0.05, stmod)
            ww/= 1e4
            spec*= A[:, 2].max()
            specint = interp1d(ww, spec)
            specA = specint(wl)
            # Use max uncertainty between yerrup and yerrdown
            yerrfit = []
            for j in np.arange(len(yerrup)):
                yerrfit.append(max([yerrup[j], yerrdown[j]]))
            yerrfit = np.array(yerrfit)
            soln = least_squares(scalespec, 1., args=(specA, A[:, 2], yerrfit))
            scale_unc = getunc.unc_jac(soln.fun, soln.jac, len(yerrfit) - 1)
            print(temp, 'K Spectrum scaling factor:', soln.x[0], '+/-', \
                                scale_unc[0])
            plt.plot(ww, soln.x*spec, label=str(temp))
            chi2red[i] = np.sum((A[:, 2] - soln.x*specA)**2/yerrfit**2) \
                            /(len(yerrfit) - 1)
            dict_results[stmod][temp] = chi2red[i]
        plt.legend(frameon=False, loc='best', fontsize=16)
        plt.xlabel('Wavelength [$\mu m$]', fontsize=16)
        plt.ylabel('Transit depth rise [ppm]', fontsize=16)
        plt.xlim(0.5, 3.5)
        plt.ylim(-200, 4000)
        chi2min = np.argmin(chi2red)
        print('chi2 min =', chi2red[chi2min], 'with Tspot =', \
                                    tspot_[chi2min], 'K')
        plt.title(stmod + r', $\min (\tilde{\chi}^2)=$' \
                + str(np.round(chi2red[chi2min], 2)) \
                + r', $T_\mathrm{spot}=$' + str(tspot_[chi2min]), fontsize=16)
        plt.show()
        plt.savefig(plotname + stmod + '_' + instrument + '.pdf')
        plt.close('all')

    fresults = open(resfile, 'wb')
    pickle.dump(dict_results, fresults)
    fresults.close()

    return np.array(wl), np.array(A), np.array(x0), np.array(sigma)

def scalespec(x, spec, y, yerr):
    '''
    Distance between model spectrum and data.
    '''
    return (x*spec - y)**2/yerr**2

def combine_spectra(pardict, tspot, ffact, stmodel, res=100.):
    '''
    Combines starspot and stellar spectrum.
    Fit with Kurucz instead of Phoenix models 2/10/19

    Input
    -----
    tspot: list (of effective temperatures)
    '''

    # This is in Angstroms
    wave = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).wave
    # Reduce resolution
    wnew = rebin.rebin_wave(wave, res)
    star = pysynphot.Icat(stmodel, pardict['tstar'], 0.0, \
                pardict['loggstar']).flux
    rebstar, errstar = rebin.rebin_spectrum(star, wave, wnew)
    # Increase errors (based on ETC?)
    errstar += rebstar*abs(np.random.normal(loc=0., scale=5.5e-3, \
                            size=len(errstar)))
    #plt.errorbar(wnew/10., rebstar, yerr=errstar, label=str(tstar))

    for i in tspot:
        spot = pysynphot.Icat(stmodel, i, 0.0, pardict['loggstar']).flux
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

def plot_precision(pardict, xaxis, deltapar):
    '''
    Plot starspot parameter precision as a function of some simulation
    parameter.

    Deltapar is used for the final plot label.
    '''

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
        for stmod in ['k93models', 'phoenix']:#, 'ck04models']:
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
    # Polynomial fit
    xaxis = np.array(xaxis)
    fit = np.polyfit(abs(xaxis), np.array(diff_spotpar), 2, \
                        w=1./np.array(std_spotpar))
    fun = np.polyval(fit, abs(xaxis))
    plt.plot(abs(xaxis), fun, 'r--', ms=2)
    #plt.xlabel(deltapar, fontsize=16)
    #plt.xlabel('$T_\star - T_\mathrm{spot}$', fontsize=16)
    plt.xlabel('$R_\mathrm{p} [R_\mathrm{J}]$', fontsize=16)
    plt.ylabel('$|T_\mathrm{spot, mod} - T_\mathrm{spot, meas}$|', fontsize=16)
    plt.text(0.8, 1250, '$T_\star =$' + str(pardict['tstar']) \
                    + ' K', fontsize=16)
    plt.show()
    plt.savefig(pardict['project_folder'] \
                + pardict['instrument'] + 'diff_' + deltapar + '.pdf')

    return
