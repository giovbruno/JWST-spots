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
import os
import sys
sys.path.append('/home/giovanni/Shelf/python/emcee/src/emcee/')
import emcee
import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import cornerplot
from transit_fit import transit_model
import multiprocessing
import lmfit

homef = os.path.expanduser('~')
modelsfolder = homef + '/Shelf/stellar_models/phxinten/HiRes/'
foldthrough = homef + '/Shelf/filters/'
intensfolder = homef + '/Shelf/stellar_models/phxinten/SpecInt/'
thrfile1 = foldthrough + 'JWST_NIRCam.F150W2.dat'
thrfile2 = foldthrough + 'JWST_NIRCam.F322W2.dat'
thrfile3 = foldthrough + 'JWST_NIRCam.F444W.dat'
thrfile4 = foldthrough + 'JWST_NIRSpec.CLEAR.dat'

plt.ioff()

def read_res(pardict, plotname, resfile, models, resol=10, interpolate=1, \
            LM=True, model='KSint', plot_input_spectrum=True, \
            mcmc=False, nest=False):
    '''
    Read in the chains files and initialize arrays with spot properties.

    wlwin: used to exlude parts of the spectra
    interpolate: only in Tspot (1) or Tspot and mu (2)
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
    bestbin = -1
    wl, A, x0, sigma = [], [], [], []
    yerrup, yerrdown = [], []
    global kr
    global unckr
    kr = []
    unckr = []
    global r0
    global r1
    global r2
    r0, r1, r2 = [], [], []
    global polyhere
    polyhere = []
    for i in np.concatenate((np.arange(expchan - 1), [-1])):
        try:
            ffopen = open(pardict['chains_folder'] + 'chains_' \
                    + str(model) + '_' + str(i) + '.pickle', 'rb')
        except FileNotFoundError: # old version
            ffopen = open(pardict['chains_folder'] + 'chains_' \
                    + str(i) + '.pickle', 'rb')
        res = pickle.load(ffopen)
        if model != 'KSint':
            if i == bestbin:
                perc = res['Percentiles'][0][-6]
                tspot = res['Percentiles'][0][-3][2]
                # Get spot width from distance between Gaussian 3-sigma distance
                # (one side)
                wspot = res['Percentiles'][0][-4][2]*3
                # Get transit parameters to compute transit duration
                # kr, q1, q2, incl, t0
                get_spot_size(wspot, pardict, res)
                print('Spot angular size:', np.round(np.degrees(delta), 2), \
                            'deg')
            else:
                wl.append(res['wl'])
                perc = res['Percentiles'][0][-1]
                tspot = res['Percentiles'][0][-1][2]
                A.append(perc[2])
                yerrup.append(perc[3] - perc[2])
                yerrdown.append(perc[2] - perc[1])
            r0.append(res['Percentiles'][0][3][2])
            r1.append(res['Percentiles'][0][4][2])
            r2.append(res['Percentiles'][0][5][2])
            kr.append(res['Percentiles'][0][0][2])
            unckr.append(np.mean([res['Percentiles'][0][0][3] \
                        - res['Percentiles'][0][0][2], \
                res['Percentiles'][0][0][2] \
                - res['Percentiles'][0][0][1]]))
            # Get polynomial here
            polyhere.append(np.polyval([r0[i], r1[i], r2[i]], tspot))
            ffopen.close()
        else:
            if i == bestbin:
                continue
            else:
                wl.append(res['wl'])
                perc = res['Percentiles'][0][-1]
                A.append(perc[2])
                yerrup.append(perc[3] - perc[2])
                yerrdown.append(perc[2] - perc[1])
                ffopen.close()
            polyhere.append(1.)

    # Save the solution to reproduce it later
    #fout1 = open(pardict['chains_folder'] + 'contrast_spec_fit.pic', 'wb')
    #pickle.dump([wl, A, yerrup, yerrdown], fout1)
    #fout1.close()
    #fout2 = open(pardict['chains_folder'] + 'kr_fit.pic', 'wb')
    #pickle.dump([wl, kr, unckr], fout2)
    #fout2.close()

    # Import kr from somewhere else
    #krfile = open('/home/giovanni/Projects/jwst_spots/revision1/NIRSpec_Prism/star_5000K/p1.0_star1.0_5000_4.5_spot3800_i90_a5_theta0_mag14.5_noscatter/MCMC/kr_fit.pic', 'rb')
    #krf = pickle.load(krfile)
    #kr = krf[1]
    #unckr = krf[2]
    #krfile.close()
    #Afile = open('/home/giovanni/Projects/jwst_spots/revision1/NIRSpec_Prism/star_5000K/p1.0_star1.0_5000_4.5_spot3800_i90_a5_theta0_mag14.5_noscatter/MCMC/contrast_spec_fit.pic', 'rb')
    #   Af = pickle.load(Afile)
    #A = Af[1]
    #yerrup = Af[2]
    #yerrdown = A[3]
    #Afile.close()
    #return
    #Aref = A[-1]
    #yerrupmed = yerrup[-1]
    #yerrdownmed = yerrdown[-1]
    # Take mean value for spot position
    #tspot = tspot[2]
    # Get **observed** mid-transit time
    #print('kr =', kr)

    mufit = pardict['muindex']
    wl = np.array(wl)
    A = np.array(A)
    yerrup = np.array(yerrup)
    yerrdown = np.array(yerrdown)
    if pardict['instrument'] == 'NIRCam':
        flag = wl < 4.
    elif pardict['instrument'] == 'NIRSpec_Prism':
        flag = wl < 6.

    # Get spot SNR
    spotSNR = A[-1]/np.mean([yerrup[-1], yerrdown[-1]])
    #wl = wl#[flag]
    #A = np.array(A[:-1])#[flag]
    #yerrup = np.array(yerrup[:-1])#[flag]
    #yerrdown = np.array(yerrdown[:-1])#[flag]
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
    res = np.diff(wl)[0]

    # Remove points with too low error bars
    #mflag = np.logical_or(yerrup < 0.2*np.median(yerrup), \
    #                                        yerrdown < 0.2*np.median(yerrdown))
    #yerrup[mflag] = np.median(yerrup)
    #yerrdown[mflag] = np.median(yerrdown)

    # Use input contrast spectrum
    plt.figure()
    plt.errorbar(wl, A, yerr=[yerrup, yerrdown], \
                fmt='ko', mfc='None', capsize=2)

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

        if models == 'josh' and interpolate == 1:
            mat = []
            # interpolate models, make finer grid - only with a few mu values
            #if not LM:
            for ti in tspot_:
                tt = format(ti, '2.2e')
                wl_ = pardict['spotmodels'][tt]['wl']
                mat.append(pardict['spotmodels'][tt]['spec'][pardict['muindex']])
            # interpolate with that and wl grid, and produce new spectrum
            zz = RectBivariateSpline(wl_, tspot_, np.array(mat).T)
        elif models == 'josh' and interpolate == 2:
            # Interpolate in mu too
            mu_ = pardict['starmodel']['mus']
            for ti in tspot_:
                for mui, _ in enumerate(mu_):
                    tt = format(ti, '2.2e')
                    wl_ = pardict['spotmodels'][tt]['wl']
                    mat.append(pardict['spotmodels'][tt]['spec'][mui])
            mat = np.array(mat).reshape(len(tspot_), len(mu_), len(wl_))
            zz = rgi((tspot_, mu_, wl_), mat)
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
                if interpolate == 1:
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
            theta = np.arccos(pardict['starmodel']['mus'][pardict['muindex']])
            if model != 'KSint':
                # Get starspot mu ** inner circle **
                if theta != 0:
                    theta_mum1 = 0.5*(theta \
                 + np.arccos(pardict['starmodel']['mus'][pardict['muindex'] + 1]))
                 # Get starspot mu ** outer circle **
                    theta_mup1 = 0.5*(theta \
                 + np.arccos(pardict['starmodel']['mus'][pardict['muindex'] - 1]))
                 # Difference in angles from stellar centre for two rings
                    diffang = np.sin(theta_mup1)**2 - np.sin(theta_mum1)**2
                    maxspotsize = np.pi*np.sin(theta_mup1 - theta)**2
                else: # For the innermost circle
                     diffang = 1.
                     maxspotsize = np.pi*np.sin(theta)**2

                minspotsize = np.pi*delta**2 #/ diffang
                params = lmfit.Parameters()
                if pardict['tstar'] == 5000.:
                    params.add('Tspot', value=4100., min=3601., max=4999.)
                elif pardict['tstar'] == 3500:
                    params.add('Tspot', value=3100., min=2301., max=3499.)
                # This is once for the fit
                ispecstar = np.transpose(pardict['starmodel']['spec'])
                mmu = pardict['starmodel']['mus']
                f_star = 2.*np.pi*np.trapz(ispecstar*mmu, x=mmu)
                params.add('beta', value=1., min=0., max=10.)
                params.add('delta', value=0.01, min=1e-6, max=0.05 + 1e-6)

                soln = lmfit.minimize(spec_res, params, method='leastsq', \
                        args=(A, yerrup, yerrdown, wl, zz, pardict, f_star))
                lmfit.printfuncs.report_fit(soln)

                bsol = compute_deltaf_f(soln.params, wl, zz, pardict, \
                                fstar=f_star)
                plt.plot(wl, bsol, label='Best fit')
                if mcmc:
                    # Let's run an MCMC
                    res = mcmc(soln, A, yerrup, yerrdown, wl, zz, pardict, \
                                f_star)
                if nest:
                    res = nested(soln, A, yerrup, yerrdown, wl, zz, pardict, \
                                f_star)
            else:
                params = lmfit.Parameters()
                if pardict['tstar'] == 3500:
                    params.add('Tspot', value=3000., min=2301., max=3499.)
                elif pardict['tstar'] == 5000:
                    params.add('Tspot', value=4200., min=3601., max=4999.)
                soln = lmfit.minimize(contr_res, params, method='leastsq', \
                     args=(A, yerrup, yerrdown, wl, zz, pardict))
                lmfit.printfuncs.report_fit(soln)
                bsol = contrast_fit(soln.params, A, yerrup, \
                            yerrdown, wl, zz, pardict)
                soln.params['Tspot'].value = pardict['tumbra']
                bsol2 = contrast_fit(soln.params, A, yerrup, yerrdown, wl, \
                            zz, pardict)
                plt.plot(wl, bsol, label='Best fit')
                plt.plot(wl, bsol2, label='True value')
                if plot_input_spectrum:
                    # Plot input contrast spectrum
                    cspectrumf = open(pardict['data_folder'] \
                                    + 'contrast_spectrum.pic', 'rb')
                    cspectrum = pickle.load(cspectrumf)
                    plt.plot(cspectrum[0][:-1], cspectrum[1][:-1], \
                            label='Input')
                    plt.legend()
                    cspectrumf.close()
        plt.xlabel('Wavelength [$\mu$m]', fontsize=16)
        plt.ylabel(r'$\Delta f(\lambda)$', fontsize=16)
        plt.legend(frameon=False)
        plt.xlim(wl.min() - 0.2, wl.max() + 0.2)
        maxL = np.argmax(likelihood)
        if not LM:
            print('L max =', likelihood[maxL], 'with Tspot =', \
                                    tspot_[maxL], 'K')
            plt.title('True value: ' + str(int(pardict['tumbra'])) + ' K' \
               + r', best fit: $T_\bullet=$' + str(int(tspot_[maxL])) + ' K', \
                fontsize=16)
        else:
            print('Best sol:', soln.params['Tspot'])
            plt.title('True value: ' + str(int(pardict['tumbra'])) + ' K' \
               + r', best fit: $T_\bullet=$' + str(int(soln.params['Tspot'])) \
                + ' K', fontsize=16)
        if LM:
            plt.savefig(plotname + stmod + '_' + pardict['observatory'] \
                        + '_LMfit.pdf')
            plt.show()
            set_trace()
            fout = open(plotname.replace('plot', 'LMfit') + stmod + '_' \
                    + pardict['observatory'] + '.pickle', 'wb')
            pickle.dump([soln, np.round(np.degrees(delta), 2)], fout)
            fout.close()
            return

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

def nested(soln, A, yerrup, yerrdown, wl, zz, pardict, f_star):
    '''
    Explore posterior distribution with (static) nested sampling.
    '''

    print('\nStarting nested sampling\n')

    # "Static" nested sampling.
    ndim = np.shape(soln.params)[0]
    pool = multiprocessing.Pool(7)
    pool.size=7
    sampler = dynesty.NestedSampler(lnprob, prior_transform, ndim, \
                nlive=100, pool=pool, \
                logl_args=(A, yerrup, yerrdown, wl, zz, pardict, f_star), \
                ptform_args=[pardict])
    sampler.run_nested()

    sresults = sampler.results

    # Plot a summary of the run.
    rfig, raxes = dyplot.runplot(sresults)
    # Plot traces and 1-D marginalized posteriors.
    tfig, taxes = dyplot.traceplot(rsesults)
    # Plot the 2-D marginalized posteriors.
    cfig, caxes = dyplot.cornerplot(sresults)

    return

def prior_transform(u, pardict):
    '''
    Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameters of interest.
    '''
    if pardict['tstar'] == 5000.:
        u[0] = 1400. * u[0] + 3600.
    elif pardict['tstar'] == 3500.:
        u[0] = 1200. * u[0] + 2300.
    u[1] *= 10.
    u[2] = u[2]*0.05 + 1e-6

    return u

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

    from run_all import ingest_stellarspectra

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

def contrast_fit(pars, y, yerrup, yerrdown, wlobs, zz, pardict, plots=False):
    '''
    Fit contrast spectrum with stellar intensity models.
    '''

    i_star = pardict['starmodel']['spec'][pardict['muindex']]
    wave = pardict['starmodel']['wl']
    i_spot = np.hstack(zz(wave, pars['Tspot']))

    if pardict['instrument'] == 'NIRSpec_Prism':
        wth, fth = np.loadtxt(thrfile4, unpack=True)
        wth*= 1e4
    elif pardict['instrument'] == 'NIRCam':
        wth1, fth1 = np.loadtxt(thrfile1, unpack=True)
        wth2, fth2 = np.loadtxt(thrfile2, unpack=True)
        wth3, fth3 = np.loadtxt(thrfile3, unpack=True)
        wth = np.concatenate((wth1, wth2, wth3))
        fth = np.concatenate((fth1, fth2, fth3))

    i_star = integ_filter(wth, fth, wave, i_star)
    i_spot = integ_filter(wth, fth, wave, i_spot)
    wlfl = np.logical_and(wth >= min(wave), wth <= max(wave))
    i_star = i_star[wlfl]
    i_spot = i_spot[wlfl]
    wth = wth[wlfl]
    i_star = degrade_spec(i_star, wth, wlobs)
    i_spot = degrade_spec(i_spot, wth, wlobs)

    contrast = 1. - i_spot/i_star

    if plots:
        plt.close('all')
        plt.errorbar(wlobs, y, yerr=[yerrup, yerrdown], fmt='o')
        plt.plot(wlobs, contrast)
        plt.show()
        set_trace()

    return contrast

def contr_res(pars, y, yerrup, yerrdown, wlobs, zz, pardict):

    residuals = np.zeros(len(y))
    contrast = contrast_fit(pars, y, yerrup, yerrdown, wlobs, zz, pardict)
    flag = y >= contrast
    residuals[flag] = (y - contrast)[flag]**2/yerrup[flag]**2
    residuals[~flag] = (y - contrast)[~flag]**2/yerrdown[~flag]**2

    return residuals

def compute_deltaf_f(par, wlobs, zz, pardict, fstar=0., plots=False):
    '''
    Compute normalized flux variation during starspot occultation.
    '''

    #ffact = 7.99880071e-01
    # Both intensity and flux are needed
    muspot = pardict['starmodel']['mus'][pardict['muindex']]
    i_star = pardict['starmodel']['spec'][pardict['muindex']]
    wave = pardict['starmodel']['wl']

    ispecstar = np.transpose(pardict['starmodel']['spec'])
    mmu = pardict['starmodel']['mus']

    # Flux from star + spot
    i_spot = np.hstack(zz(wave, par['Tspot']))
    istar_muspot = pardict['starmodel']['spec'][pardict['muindex']]
    fstar = fstar - (2.*np.pi*istar_muspot*muspot*np.diff(mmu)[0])* \
                (1. - par['delta']) \
                + (2.*np.pi*i_spot*muspot*np.diff(mmu)[0])*par['delta']

    if pardict['instrument'] == 'NIRSpec_Prism':
        wth, fth = np.loadtxt(thrfile4, unpack=True)
        wth*= 1e4
    elif pardict['instrument'] == 'NIRCam':
        wth1, fth1 = np.loadtxt(thrfile1, unpack=True)
        wth2, fth2 = np.loadtxt(thrfile2, unpack=True)
        wth3, fth3 = np.loadtxt(thrfile3, unpack=True)
        wth = np.concatenate((wth1, wth2, wth3))
        fth = np.concatenate((fth1, fth2, fth3))

    idiff = integ_filter(wth, fth, wave, i_star - i_spot)
    fstar = integ_filter(wth, fth, wave, fstar)
    #f_star_spot = integ_filter(wth, fth, wave, f_star_spot)
    # Only keep values within filters curves
    wlfl = np.logical_and(wth >= min(wave), wth <= max(wave))

    wth = wth[wlfl]
    idiff = idiff[wlfl]
    fstar = fstar[wlfl]
    #f_star_spot = f_star_spot[wlfl]
    #modint = interp1d(wl, f_star_spot)
    #deltaf_f = modint(w)
    deltaf_f = degrade_spec(idiff/fstar, wth, wlobs)
    #deltaf_f /= np.array(polyhere[:-1])
    #f_star_spot f= degrade_spec(f_star_spot, wth, wlobs)
    #fstar = degrade_spec(fstar, wth, wlobs)
    #idiff = degrade_spec(idiff, wth, wlobs)
    #beta = np.pi*(delta*semimaj)**2/(pardict['rstar']*Rsun)**2*kr**2
    #ballerini = 1. - 0.5*(1. - f_star_spot/fstar)*ffact
    #ffact = 1. - (np.cos(kr))**2
    #ffact = 1. - (np.cos(np.array(kr)/(semimaj/pardict['rstar']/Rsun)))**2
    beta = par['beta']*np.pi*muspot#*np.diff(mmu)[0]
    #*np.array(kr)**2#*f_star_spot/fstar#)
    #beta = ffact*np.pi*kr[-1]**2#
    #beta = degrade_spec(beta, wth, wlobs)
    #beta = 2.*np.pi*np.array(kr)*np.sin(np.array(kr))
    if plots:
        print(tspot, ffact)
        plt.close('all')
        plt.plot(wlobs, deltaf_f*beta)
        plt.show()
        set_trace()
    #deltaf_f = deltaf_f*len(pardict['starmodel']['mus'])*beta*kr**2
    #planetangle = kr*pardict['rstar']*Rsun/semimaj
    return deltaf_f*beta#[:len(deltaf_f)]#[:-1]#/polyhere[:-1]

def spec_res(par, spec, yerrup, yerrdown, wlobs, zz, pardict, fstar):
    '''
    Compute squared residuals from model spectrum and data.

    fstar is the stellar spectrum computed from intensity.
    '''

    mod = compute_deltaf_f(par, wlobs, zz, pardict, fstar=fstar)
    res = np.zeros(len(mod))
    flag = spec >= mod
    res[flag] = (spec - mod)[flag]**2/yerrup[flag]**2
    res[~flag] = (spec - mod)[~flag]**2/yerrdown[~flag]**2

    return res

def lnprior(par, pardict):

    tspot, beta, delta = par
    if pardict['tstar'] == 3500:
        if np.logical_or.reduce((tspot < 2300., tspot > 3500., \
                delta <= 1e-8, delta > 1.0)):
                return -np.inf
        else:
            #lnp_ffact = lnp_jeffreys(ffact, 1.0, 1e-6)
            return 0.#lnp_ffact

    elif pardict['tstar'] == 5000:
        if np.logical_or.reduce((tspot < 3600., tspot > 5000., \
                delta <= 1e-8, delta > 1.0)):
                return -np.inf
        else:
            return 0.

def lnp_jeffreys(val, valmax, valmin):
    '''
    Jeffreys prior
    '''
    return np.log(1./(val*np.log(valmax/valmin)))

def lnprob(par, spec, yerrup, yerrdown, wlobs, zz, pardict, fstar):
    '''
    Compute posterior probability for spectrum fit.
    '''

    lp = lnprior(par, pardict)
    if not np.isfinite(lp):
        return -np.inf
    else:
        pars = lmfit.Parameters()
        pars.add('Tspot', value=par[0])
        pars.add('beta', value=par[1])
        pars.add('delta', value=par[2])
        chi2 = np.sum(spec_res(pars, spec, yerrup, yerrdown, wlobs, zz, \
                    pardict, fstar))
        lnL = -0.5*len(spec)*np.log(np.mean([yerrup, yerrdown])) \
                - 0.5*len(spec)*np.log(2.*np.pi) - 0.5*chi2
        return lnL + lp

def mcmc(soln, A, yerrup, yerrdown, wl, zz, pardict, fstar):
    '''
    Get posterior distribution for the starspot configuration.
    '''

    # MCMC starting about the optimized solution
    pars = []
    for i in soln.params.keys():
        pars.append(soln.params[i].value)
    initial = np.array(pars)
    ndim, nwalkers = len(initial), 64
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=([A, yerrup, yerrdown, wl, zz, pardict, fstar]), threads=8)

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
    nsteps = 64
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width+1)*float(i)/nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(width - n)))
    sys.stdout.write("\n")
    p0, lp, _ = result
    sampler.reset()

    print("Running production...")
    nsteps = 64
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
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

    titles = [r'$T_\bullet$', '$\delta$', r'$\beta$']#, r'$\alpha$']
    truths = [pardict['tumbra'], None, None]
    cornerplot.cornerplot(samples, titles, truths, \
            pardict['chains_folder'] + '/cornerspec.pdf', ranges=None)

    # Save chains
    fout = open(pardict['chains_folder'] + '/chains_spec.pickle', 'wb')
    chains_save = {}
    #chains_save['wl'] = wl
    #chains_save['LM'] = soln.x
    #chains_save['Burn_in'] = [p0, lp]
    chains_save['Chains'] = samples
    ##chains_save['Starspot_size'] = size
    chains_save['Mean_acceptance_fraction'] \
                                    = np.mean(sampler.acceptance_fraction)
    #chains_save['Autocorrelation_multiples'] = acor_time
    #chains_save["Percentiles"] = [percentiles]
    chains_save["ln_L"] = lnL
    pickle.dump(chains_save, fout)
    fout.close()

    #fout = open(diz['chains_folder'] + '/chains_best_' \
    #            + str(ind) + '.p', 'wb')
    #pickle.dump(best_sol, fout)
    #fout.close()

    return

def get_spot_size(wspot, pardict, dict_chains):
    '''
    Get starspot size from transit fit. Relate time of occultation crossing
    to T14 for transit. This will get you a lower limit.

    Input: wspot (3-sigma width of Gaussian profile fitted to the occultation
    feature).
    '''

    tpar = []
    for j in [0, 1, 2, -2, -1]:
        tpar.append(dict_chains['Percentiles'][0][j][2])
    if pardict['tstar'] == 3500:
        tt = np.arange(0.075 - 0.04, 0.125 + 0.04, 60./86400.)
    elif pardict['tstar'] == 5000:
        tt = np.arange(0.05 - 0.1, 0.150 + 0.1, 60./86400.)
    # Get observed T14
    rhostar = 5.14e-5/pardict['rstar']*10**pardict['loggstar'] #cgs
    per_planet = pardict['pplanet']
    aR = (6.67408e-11*rhostar*1e3*(per_planet*86400.)**2/(3.*np.pi))**(1./3.)
    tmod = transit_model([tpar[0], tpar[-1], tpar[-2], 0., 0.], tt, \
                0.0, pp=pardict['pplanet'], semimaj=aR)
    ttr = tmod != 1.
    tdur = 0.5*(tt[ttr].max() - tt[ttr].min())
    # Compute space covered during transit, from tangential velocity
    global Rsun
    Rsun = 695500e3
    global semimaj
    semimaj = aR*pardict['rstar']*Rsun
    vt = 2.*np.pi*semimaj/(per_planet*86400.)
    x_star = vt*pardict['rstar']*Rsun
    global delta
    delta = wspot*x_star/(tdur*86400.)/semimaj

    return delta
