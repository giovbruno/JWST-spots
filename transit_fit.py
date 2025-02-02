import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy import stats
import os, sys, pickle
from datetime import datetime
import emcee
from emcee.autocorr import integrated_time
import batman
from pytransit import QuadraticModel
import cornerplot
sys.path.append('../KSint_wrapper/SRC/')
import ksint_wrapper_fitcontrast
import get_uncertainties
import lmfit
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty import NestedSampler
from pdb import set_trace
plt.ioff()

def transit_spectro(pardict, resol=10, model='KSint', tight_ld_prior=False, \
        resume=False):
    '''
    Launches on all spectral bands
    '''

    if pardict['instrument'] == 'NIRSpec_Prism':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam':
        ldendname = 'nircam'
    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    ldd_ = pickle.load(ldd)
    xobs = ldd_[1][0]
    ldlist = ldd_[0]
    ldd.close()
    flag = xobs < 6
    expchan = len(xobs[flag])

    # First, fit transit with the smallest error bars, to then fix the spot
    # parameters
    spec = pickle.load(open(pardict['data_folder'] + 'spec_model_' \
                + pardict['observatory'] + '.pic', 'rb'))
    ymoderr = spec[2]
    #bestbin = ymoderr.argmin()
    bestbin = -1
    transit_emcee(pardict, bestbin, bestbin, ldlist, model=model, resume=resume)

    for i in np.arange(expchan - 1):
        if i != bestbin:
        #if i == 10:
            transit_emcee(pardict, int(i), bestbin, ldlist, model=model, \
                    tight_ld_prior=tight_ld_prior, resume=resume)

    return

def transit_emcee(diz, ind, bestbin, ldlist, model='KSint', \
            resume=False, nested=True, tight_ld_prior=True, bestlike=False):
    '''
    Parameters
    ----------
    bestlike: use best-likelihood values from the white LC to fit parameters
              in the spectroscopic channels - or median values
    '''
    if resume and os.path.exists(diz['chains_folder'] + 'transit_' + str(ind) \
            + '_nested.pic'):
            #created = os.stat(diz['chains_folder'] + 'transit_' + \
            #        str(ind) + '_nested.pic').st_ctime
            #dd = datetime.fromtimestamp(created)
            #if dd > datetime(2021, 7, 13, 12, 0, 0):
            return

    print('Channel', str(ind))
    plt.close('all')
    os.system('mkdir ' + diz['chains_folder'])
    lcfile = open(diz['data_folder'] + 'transit_spots_' + str(ind) \
                    + '.pic', 'rb')
    lc = pickle.load(lcfile)
    lcfile.close()
    t, y, yerr, wl = lc

    global per_planet
    per_planet = diz['pplanet']

    # From stellar density
    rhostar = 5.14e-5/diz['rstar']*10**diz['loggstar'] #cgs
    global aR
    aR = (6.67408e-11*rhostar*1e3*(per_planet*86400.)**2/(3.*np.pi))**(1./3.)
    print('aR = ', aR)
    global modeltype
    if not nested:
        if model == 'batman' or model == 'pytransit':
            # LM fit boundaries - fit for t0 and x0 only on the first channel
            bounds_model = []
            bounds_model.append((0.01, 0.2))         # rp/r*
            bounds_model.append((0., 0.5))          # q1 from Kipping+2013
            bounds_model.append((0., 0.5))
            #if diz['tstar'] == 3500:
            #    bounds_model.append((0.01, 0.2))          # q1 from Kipping+2013
            #    bounds_model.append((0.15, 0.3))          # q2
            #elif diz['tstar'] == 5000:
            #    bounds_model.append((0.15, 0.3))          # q1 from Kipping+2013
            #    bounds_model.append((0.15, 0.3))
            bounds_model.append((-1., 1.))           # r0
            bounds_model.append((-1., 1.))           # r1
            bounds_model.append((0., 10.))           # r2
            bounds_model.append((1e-4, 1))           # A
            bounds_model.append((2., 6.))           # n # Flat gaussian
            bounds_model.append((np.diff(t).min(), 0.02))         # sigma
            if int(diz['theta']) == 0:
                bounds_model.append((0.09, 0.11))        # x0 = 0.1051
            elif int(diz['theta']) == 40:
                bounds_model.append((0.105, 0.125))
            bounds_model.append((80., 90.))          # orbit inclination
            bounds_model.append((0.08, 0.12))        # t0
            #bounds = [[], []]
            #for i in np.arange(len(bounds_model)):
            #    bounds[0].append(bounds_model[i][0])
            #    bounds[1].append(bounds_model[i][1])

            # Initial values
            kr, r0, r1, r2 = 0.09, -1e-3, -1e-3, 1.
            if diz['tstar'] == 3500:
                q1, q2 = 0.1, 0.15
            elif diz['tstar'] == 5000:
                q1, q2 = 0.2, 0.22
            if diz['theta'] == 0.:
                tspot_ = 0.1#0.095
            else:
                if diz['tstar'] == 3500:
                    #if diz['instrument'] == 'NIRCam':
                    tspot_ = 0.115
                    if diz['latspot'] == 21.:
                        tspot_ = 0.105
                    #else:
                    #    tspot_ = 0.115#0.95 # This seems to work better for NIRSpec
                else:
                    tspot_ = 0.135
            A, wspot_ = 2e-3, 0.01
            incl_, t0_ = 89., 0.1
            n = 2.05

            # This will set the fit or fix for tspot and spot size
            # Spot position and size are fitted only on the bluemost wavelength bin
            if ind == bestbin:
                modeltype = 'fitt0'
                initial_params = kr, q1, q2, r0, r1, r2, A, n, \
                                    wspot_, tspot_, incl_, t0_
            else:
                modeltype = 'fixt0'
                # Extract spot time from first wavelength bin
                ffopen = open(diz['chains_folder'] + 'chains_' + str(model) \
                    + '_' + str(bestbin) + '.pic', 'rb')
                res = pickle.load(ffopen)
                perc = res['Percentiles'][0]
                # These will be fixed in the fit
                fix_dict = {}
                fix_dict['nspot'] = perc[-5][2]
                fix_dict['wspot'] = perc[-4][2]
                fix_dict['tspot'] = perc[-3][2]
                fix_dict['incl'] = perc[-2][2]
                fix_dict['t0'] = perc[-1][2]
                initial_params = kr, q1, q2, r0, r1, r2, A
                bounds_model = bounds_model[:-5]
                #bounds = [[], []]
                #for i in np.arange(len(bounds_model[:-5])):
                #    bounds[0].append(bounds_model[i][0])
                #    bounds[1].append(bounds_model[i][1])

            # LM fit
            options= {}
            #ftol = 1e-10
            #options['ftol'] = ftol
            nll = lambda *args: -lnlike(*args)
            soln = minimize(nll, initial_params, jac=False, method='L-BFGS-B', \
                 args=(t, y, yerr, model, fix_dict), bounds=bounds_model, \
                 options=options)
            #soln = least_squares(residuals, initial_params, bounds=bounds, \
            #        args=(t, y, yerr, model, fix_dict), method='trf')

        elif model == 'KSint':
            bounds_model = []
            bounds_model.append((0.01, 0.2))         # rp/r*
            bounds_model.append((245., 255.))        # M
            bounds_model.append((1e-6, 1. - 1e-6))   # q1
            bounds_model.append((1e-6, 1. - 1e-6))   # q2
            bounds_model.append((260., 270.))        # long spot
            bounds_model.append((1., 10.))           # a umbra
            bounds_model.append((1e-6, 1.))          # contrast
            bounds_model.append((80., 100.))         # inclination
            # Initial values
            kr, M, q1, q2, long, aumbra, contr = 0.09, 250., 0.3, 0.3, 265., 3., 0.5
            if diz['theta'] == 0.:
                tspot_ = 0.1#0.095 #0.1
            else:
                if diz['tstar'] == 3500:
                    if diz['instrument'] == 'NIRCam':
                        tspot_ = 0.10#0.115
                    else:
                        tspot_ = 0.115#0.95 # This seems to work better for NIRSpec
                else:
                    tspot_ = 0.1 # 0.13
            incl = 89.
            fout2 = open(diz['data_folder'] + 'KSint_pars.pic', 'rb')
            fix_dict = pickle.load(fout2)
            fout2.close()

            # This will set the fit or fix for tspot and spot size
            # Spot position and size are fitted only on the bluemost wavelength bin
            if ind == bestbin:
                modeltype = 'fitt0'
                initial_params = kr, M, q1, q2, long, aumbra, contr, incl
            else:
                modeltype = 'fixt0'
                # Extract spot time from first wavelength bin
                ffopen = open(diz['chains_folder'] + 'chains_' + str(model) \
                    + '_' + str(bestbin) + '.pic', 'rb')
                res = pickle.load(ffopen)
                perc = res['Percentiles'][0]
                # These will be fixed in the fit
                fix_dict['M'] = perc[1][2]
                fix_dict['longspot'] = perc[-4][2]
                fix_dict['aspot'] = perc[-3][2]
                fix_dict['incl'] = perc[-1][2]
                initial_params = kr, q1, q2, contr
                bounds_model = [element for i, element in enumerate(bounds_model) \
                            if i in [0, 2, 3, 6]]

            # LM fit
            options= {}
            ftol = 1e-10
            options['ftol'] = ftol
            nll = lambda *args: -lnlike(*args)
            soln = minimize(nll, initial_params, jac=False, method='L-BFGS-B', \
                 args=(t, y, yerr, model, fix_dict), bounds=bounds_model, \
                 options=options)

        print('Likelihood maximimazion results:')
        print(soln.x)
        # This contains the spot signal with the transit model removed
        plot_best(soln, t, y, yerr, \
                diz['chains_folder'] + 'best_LM_' + str(model) + '_' + str(ind) \
                + '.pdf', model=model, fix_dict=fix_dict)

        # Now, MCMC starting about the optimized solution
        initial = np.array(soln.x)
        if ind == bestbin:
            ndim, nwalkers = len(initial), 128
            iters = 2000
            if diz['latspot'] == 21:
                iters=4000
        else:
            ndim, nwalkers = len(initial), 64
            iters = 1000
            if diz['latspot'] == 21:
                iters=2000
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=([t, y, yerr, model, fix_dict, False]))
        # Variation around LM solution
        pos = initial*(1. + 0.01*(np.random.randn(nwalkers, ndim)))
                #+ 1e-6*(np.random.randn(nwalkers, ndim))
        # Check condition number (must be < 1e8 to maximise walker linear
        # independence). Problems might be caued by LM results = 0
        cond = np.linalg.cond(pos)
        while cond >= 1e8:
            pos += 1e-4*(np.random.randn(nwalkers, ndim))
            cond = np.linalg.cond(pos)

        print("Running MCMC...")

        sampler.run_mcmc(pos, iters, progress=False)

        # Merge in a single chain
        samples = sampler.get_chain(discard=300, thin=1, flat=True)
        # Inclination is symmetric wrt 90°
        if ind == bestbin:
            high_incl = samples[:, -2] > 90.
            samples[high_incl, -2] -= 2.*(samples[high_incl, -2] - 90.)
        lnL = sampler.get_log_prob(discard=300, thin=1, flat=True)
        best_sol = samples[lnL.argmax()]
        print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))
        # Inspect convergence using the integrated autocorrelation time
        # (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
        autocorr_time = [integrated_time(i, quiet=True)[0] for i in samples.T]
        autocorr_multiples = np.shape(samples)[0]/np.array(autocorr_time)

        percentiles = [np.percentile(samples[:,i], [4.55, 15.9, 50, 84.1, 95.45]) \
                            for i in np.arange(np.shape(samples)[1])]

        plot_samples(samples, best_sol, t, y, yerr, wl, \
                diz['chains_folder'] + 'samples_MCMC_' + str(model) + '_' \
                + str(ind) + '.pdf', model=model, fix_dict=fix_dict)

        if model != 'KSint':
            if ind == bestbin:
                titles = [r'$R_\mathrm{p}/R_\star$', r'$u_1$', r'$u_2$', \
                        r'$r_0$', r'$r_1$', r'$r_2$', \
                        r'$\alpha_\mathrm{spot}$', r'$n$', r'$w_\mathrm{spot}$', \
                        '$t_\mathrm{spot}$', r'$i$', '$t_\mathrm{tr}$']
            else:
                titles = [r'$R_\mathrm{p}/R_\star$', r'$u_1$', r'$u_2$', \
                            r'$r_0$', r'$r_1$', r'$r_2$', \
                            r'$\alpha_\mathrm{spot}$']
        else:
            if ind == bestbin:
                titles = [r'$R_\mathrm{p}/R_\star$', r'$M$', r'$u_1$', r'$u_2$', \
                        r'$\phi_\mathrm{spot}$', r'$A_\mathrm{spot}$', \
                        '$C$', '$i$']
            else:
                titles = [r'$R_\mathrm{p}/R_\star$', r'$u_1$', r'$u_2$', '$C$']

        truths = None
        if ind == 0 or ind == bestbin:
            cornerplot.cornerplot(samples, titles, truths, \
             diz['chains_folder'] + '/corner_' + model + '_' + str(ind) + '.pdf', \
             ranges=None)
        plt.close('all')
        # Starspot size
        #size = starspot_size(diz, samples, str(ind))

        # Save chains
        fout = open(diz['chains_folder'] + '/chains_' + str(model) + '_' + str(ind) \
                                                        + '.pic', 'wb')
        chains_save = {}
        chains_save['wl'] = wl
        chains_save['LM'] = soln.x
        chains_save['Chains'] = samples
        #chains_save['Starspot_size'] = size
        chains_save['Mean_acceptance_fraction'] \
                                        = np.mean(sampler.acceptance_fraction)
        chains_save['Autocorr_multiples'] = autocorr_multiples
        chains_save["Percentiles"] = [percentiles]
        chains_save["ln_L"] = lnL
        pickle.dump(chains_save, fout)
        fout.close()

        #fout = open(diz['chains_folder'] + '/chains_best_' + str(model) \
        #            + '_' + str(ind) + '.p', 'wb')
        #pickle.dump(best_sol, fout)
        #fout.close()
    elif nested:
        fix_dict = {}
        if ind == bestbin:
            titles = [r'$R_\mathrm{p}/R_\star$', r'$u_1$', r'$u_2$', \
                    r'$r_0$', r'$r_1$', r'$r_2$', \
                    r'$\Delta f$', r'$n$', r'$w$', \
                    r'$t_\bullet$', r'$i$', '$t_\mathrm{tr}$']
            modeltype = 'fitt0'
            nlive = 200
        else:
            titles = [r'$R_\mathrm{p}/R_\star$', r'$u_1$', r'$u_2$', \
                        r'$r_0$', r'$r_1$', r'$r_2$', \
                        r'$\Delta f$$']
            modeltype = 'fixt0'
            nlive = 150
            ffopen = open(diz['chains_folder'] \
                        + 'transit_-1_nested.pic', 'rb')
            sresults = pickle.load(ffopen)
            samples = sresults['samples']
            weights = np.exp(sresults.logwt - sresults.logz[-1])
            perc = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                        weights=weights) for samps in samples.T]
            # These will be fixed during the fit
            #if perc[-5][2] - perc[-5][1] > perc[-5][1]:
            #    fix_dict['nspot'] = 2.
            #else:
            #    fix_dict['nspot'] = perc[-5][1]
            samples_equal = dyfunc.resample_equal(samples, weights)
            Lmaxarg = sresults.logz.argmax()
            if bestlike:
                fix_dict['nspot'] = samples_equal[:, -5][Lmaxarg]
                fix_dict['wspot'] = samples_equal[:, -4][Lmaxarg]
                fix_dict['tspot'] = samples_equal[:, -3][Lmaxarg]
                fix_dict['incl'] = samples_equal[:, -2][Lmaxarg]
                fix_dict['t0'] = samples_equal[:, -1][Lmaxarg]
            else:
                fix_dict['nspot'] = perc[-5][1]
                fix_dict['wspot'] = perc[-4][1]
                fix_dict['tspot'] = perc[-3][1]
                fix_dict['incl'] = perc[-2][1]
                fix_dict['t0'] = perc[-1][1]

        ndim = len(titles)
        sampler = NestedSampler(lnprob, prior_transform, ndim, \
            nlive=nlive, ptform_args=[diz, t, ldlist, tight_ld_prior, ind], \
            logl_args=(t, y, yerr, model, fix_dict, True))
        sampler.run_nested(print_progress=True)
        sresults = sampler.results

        samples = sresults['samples']
        weights = np.exp(sresults.logwt - sresults.logz[-1])
        samples_equal = dyfunc.resample_equal(samples, weights)
        best_sol = samples_equal[sresults['logl'].argmax()]
        plot_samples(samples_equal, best_sol, t, y, yerr, wl, \
                diz['chains_folder'] + 'samples_nested_' + str(model) + '_' \
                + str(ind) + '.pdf', model=model, fix_dict=fix_dict)
        # Save chains
        fout = open(diz['chains_folder'] + 'transit_' + str(ind) \
                + '_nested.pic', 'wb')
        pickle.dump(sresults, fout)
        fout.close()

        # Diagnostic plots
        if ind == -1 or ind == 10:
            #try:
            #   rfig, raxes = dyplot.runplot(sresults)
            #except IndexError:
            #   rfig, raxes = dyplot.runplot(sresults, span=[(0., 10.), \
            #                    0.001, 0.2, (5., 6.)])
            #plt.savefig(diz['chains_folder'] + '/transit_' + str(ind) \
            #            + '_runplot_nested_spec.pdf')
            # Plot traces and 1-D marginalized posteriors.
            tfig, taxes = dyplot.traceplot(sresults, labels=titles)
            plt.savefig(diz['chains_folder'] + 'transit_' + str(ind) \
                    + '_traceplot_nested.pdf')
            # Plot the 2-D marginalized posteriors.
            label_kwargs = {}
            label_kwargs['fontsize'] = 14
            cfig, caxes = dyplot.cornerplot(sresults, color='b', show_titles=True, \
                    labels=titles, quantiles=[0.159, 0.50, 0.841], title_fmt='.4f')
            plt.savefig(diz['chains_folder'] + 'transit_' + str(ind) \
                    + '_cornerplot_nested.pdf')
            plt.close('all')

    return

def prior_transform(u, diz, t, ldlist, tight_ld_prior, wlind):
    '''
    Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameters of interest.

    tight_ld_prior: Set Gaussian priors on LD coefficients
    https://dynesty.readthedocs.io/en/latest/quickstart.html?highlight=corner#prior-transforms
    '''

    x = np.array(u)

    if len(u) == 12:
        #x[0] = u[0]*0.19 + 0.01 # rp/r*
        x[0] = stats.loguniform.ppf(u[0], 0.01, 0.2)
        #xx = np.linspace(0.01, 0.2, 1000)
        #yy = lnp_jeffreys(xx, 0.2, 0.01)
        #zz = stats.rv_discrete(a=xx.min(), b=xx.max(), values=(xx, yy))
        #x[0] = zz.ppf(u[0])
        if not tight_ld_prior:
            x[1] = u[1] # u1
            x[2] = u[2] # u2
        else:  # Gaussian priors for LD coefficients
            sig = [0.05]
            x1 = stats.norm.ppf([u[1]])
            x[1] = max([np.dot(sig, x1) + ldlist[wlind][0], 0.])
            x2 = stats.norm.ppf([u[2]])
            x[2] = max([np.dot(sig, x1) + ldlist[wlind][1], 0.])
        x[3] = -1. + 2.*u[3] # r0
        x[4] = -1. + 2.*u[4] # r1
        x[5] = 10.*u[5] # r2
        x[6] = 0.1*u[6] + 1.e-5 # A
        x[7] = u[7]*6. + 2. # n
        x[8] = u[8]*0.02 + np.diff(t).min() # w
        if int(diz['theta']) == 0:
            x[9] = u[9]*0.02 + 0.09 # x0
        elif int(diz['theta']) == 40:
            x[9] = u[9]*0.02 + 0.11
        elif int(diz['theta']) == 21:
            x[9] = u[9]*0.015 + 0.105
            #x[9] = u[9]*0.02 + 0.10
        #xx = np.linspace(np.radians(80.), np.pi/2., 1000)
        #yy = lnp_sine(xx, np.radians(90.), np.radians(80.))
        #yy /= np.sum(yy)
        #zz = stats.rv_continuous(a=xx.min(), b=xx.max(), values=(xx, yy))
        x[10] = u[10]*10. + 80. # i
        #x[10] = zz.ppf(u[10])
        x[11] = u[11]*0.04 + 0.08 # t0
    elif len(u) == 7:
        x[0] = stats.loguniform.ppf(u[0], 0.01, 0.2)    # rp/r*
        if not tight_ld_prior:
            x[1] = u[1] # u1
            x[2] = u[2] # u2
        else:  # Gaussian priors for LD coefficients
            sig = [0.05]
            x1 = stats.norm.ppf([u[1]])
            x[1] = np.dot(sig, x1) + ldlist[wlind][0]
            x2 = stats.norm.ppf([u[2]])
            x[2] = np.dot(sig, x1) + ldlist[wlind][1]
        x[3] = -1. + 2.*u[3]
        x[4] = -1. + 2.*u[4]
        x[5] = 10.*u[5]
        x[6] = 0.1*u[6] + 1.e-6

    return x

def chi2(model, y, yerr):
    return np.sum((model - y)**2/yerr**2)

def plot_best(sol, t, y, yerr, plotname, model='KSint', fix_dict={}, \
                alpha=1.):

    if type(sol) != np.ndarray: # This is the LM minimization
        params = sol.x
    else:
        params = sol
    bestmodel = transit_spot_syst(params, t, model, fix_dict)

    ln_evidence = -0.5*np.log(np.mean(yerr)) - 0.5*np.log(2.*np.pi) \
                    - 0.5*chi2(bestmodel, y, yerr)
    #BIC = np.log(len(y))*len(params) - 2.*ln_evidence
    rms = np.std((y - bestmodel)/y.max()*1e6)

    xTh = np.linspace(t.min(), t.max(), len(t)*10)
    yTh = transit_spot_syst(params, xTh, model, fix_dict)
    yTh_t = transit_spot_syst(params, t, model, fix_dict)
    rms = np.std((y - yTh_t)/y.max()*1e6)
    chisq = chi2(yTh_t, y, yerr)/(len(y) - len(params))
    print('Chi2 = ' + str(chisq))
    print('RMS = ' + str(rms) + ' ppm')

    plt.close('all')
    fig1 = plt.figure(1, figsize=(9, 7))
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    frame1 = fig1.add_axes((.1,.3,.8,.6), sharex=frame2)
    plt.setp(frame1.get_xticklabels(), visible=False)

    frame1.plot(xTh, yTh, 'orange')
    frame1.errorbar(t, y, yerr=yerr, fmt='k.')
    frame1.set_xlim(t.min(), t.max())
    frame1.set_ylabel('Relative flux', fontsize=16)

    frame2.errorbar(t, (y - bestmodel)*1e6, \
            yerr=yerr*1e6, fmt='k.')
    frame2.plot([t.min(), t.max()], [0., 0.], 'k--')
    frame2.set_xlim(t.min() - 0.002, t.max() + 0.002)
    frame2.set_ylabel('Residuals [ppm]', fontsize=16)
    frame2.set_xlabel('Time [days]', fontsize=16)

    plt.title('Joint transit-starspot fit', fontsize=18)
    plt.savefig(plotname)
    plt.close('all')

    return

def plot_samples(samples, best_sol, t, y, yerr, wl, plotname, \
                    model='KSint', fix_dict={}, size=300):
    '''
    Plot 1000 random samples
    '''
    plt.close('all')
    fig1 = plt.figure(1, figsize=(9, 7))
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    frame1 = fig1.add_axes((.1,.3,.8,.6), sharex=frame2)
    plt.setp(frame1.get_xticklabels(), visible=False)

    frame1.errorbar(t, y, yerr=yerr, fmt='k.')
    frame1.set_xlim(t.min() - 0.002, t.max(), 0.002)
    frame1.set_ylabel('Relative flux', fontsize=14)

    frame2.plot([t.min(), t.max()], [0., 0.], 'k--')
    frame2.set_xlim(t.min() - 0.002, t.max() + 0.002)
    frame2.set_ylabel('Residuals [ppm]', fontsize=14)
    frame2.set_xlabel('Time [days]', fontsize=14)

    plt.title(r'$\lambda = {} \, \mu$m'.format(np.round(wl, 2)), fontsize=16)

    inds = np.random.randint(np.shape(samples)[0], size=size)
    xTh = np.linspace(t.min(), t.max(), len(t)*10)
    for ind in inds:
        sample = samples[ind]
        if np.isfinite(lnprior(sample, model)):
            yTh = transit_spot_syst(sample, xTh, model, fix_dict)
            frame1.plot(xTh, yTh, 'orange', alpha=0.1)

    # Plot residuals for best solutiopn
    best_fit = transit_spot_syst(best_sol, t, model, fix_dict)
    frame2.errorbar(t, (y - best_fit)*1e6, \
            yerr=yerr*1e6, fmt='k.')

    plt.ylim(y.min() - 1e-3, y.max() + 1e-3)
    plt.savefig(plotname)
    #plt.show()
    #set_trace()
    #plt.close('all')

    return

def transit_spot_syst(par, t, model, fix_dict):

    if model != 'KSint':
        kr = par[0]
        q1 = par[1]
        q2 = par[2]
        r0 = par[3]
        r1 = par[4]
        r2 = par[5]
        Aspot = par[6]
        #n = par[7]
        if modeltype == 'fixt0':
            n = fix_dict['nspot']
            sig = fix_dict['wspot']
            x0 = fix_dict['tspot']
            inclin = fix_dict['incl']
            tc = fix_dict['t0']
        else:
            n = par[-5]
            sig = par[-4]
            x0 = par[-3]
            inclin = par[-2]
            tc = par[-1]
        model = (transit_model([kr, tc, inclin, q1, q2], t) \
            + gauss(t, [Aspot, x0, sig, n]))*np.polyval([r0, r1, r2], t)
    else:
        kr = par[0]
        if modeltype == 'fitt0':
            Mangle = par[1]
            longspot = par[4]
            aspot = par[5]
            contrast = par[6]
            temp1 = par[4] - 2
            temp2 = 12.
            temp3 = 1.
            u1 = par[2]#2.*par[2]**0.5*par[3]
            u2 = par[3]#par[2]**0.5*(1. - 2.*par[3])
            fix_dict['incl'] = par[7]
        else:
            u1 = par[1]#2.*par[2]**0.5*par[3]
            u2 = par[2]#par[2]**0.5*(1. - 2.*par[3])
            contrast = par[3]
            longspot = fix_dict['longspot']
            aspot = fix_dict['aspot']
            Mangle = fix_dict['M']
            temp1 = 20.
            temp2 = 12.
            temp3 = 1.
            fix_dict['incl'] = incl
        pars = [kr, Mangle, u1, u2, longspot, aspot, contrast, temp1, \
                temp2, temp3]
        model = ksint_wrapper_fitcontrast.main(pars, t, fix_dict)

    return model

def gauss(x, par):
    '''
    Generalised Gaussian. Par is defined as [A, x0, sigma, n]
    (flat for n > 2, peaked for n < 2).
    '''
    A, x0, sigma, n = par
    y = A*np.exp(-(abs(x - x0)/sigma)**n)

    # Bring the function to zero beyond 3 sigma
    #y[abs(x - x0) > 1.7*sigma] = 0.

    return y

def transit_model(par, t, u1=0, u2=0, pp=0., semimaj=0.):
    '''
    # From stellar density to aR*
    #aR = (6.67408e-11*1e-3*par[2]*(per_planet*86400)**2/(3*np.pi))**(1./3.)

    # q1 and q2 (as free parameters) are Kipping's parameters
    '''

    # Back to u1, u2
    #u1 = 2.*par[3]**0.5*par[4]
    #u2 = par[3]**0.5*(1. - 2.*par[4])

    params = batman.TransitParams()
    params.t0 = par[1]
    if pp == 0.:
        params.per = per_planet
    else:
        params.per = pp
    params.rp = par[0]
    if semimaj == 0.:
        params.a = aR
    else:
        params.a = semimaj
    params.inc = par[2] # in degrees
    params.ecc = 0.
    params.w = 0.
    params.u = [par[3], par[4]]
    #params.u = [u1, u2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    '''
    tm = QuadraticModel()
    tm.set_data(t)

    k = par[0]
    ldc = np.array([u1, u2])
    t0 = par[1]
    if pp == 0.:
        p = per_planet
    else:
        p = pp
    if semimaj == 0.:
        a = aR
    else:
        a = semimaj
    i = np.radians(par[2])
    flux = tm.evaluate(k, ldc, t0, p, a, i)
    '''
    return flux

def lnprob(params, t, y, yerr, model, fix_dict, nested):

    if not nested:
        lp = lnprior(params, model)
    else:
        lp = 0.
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lnlike(params, t, y, yerr, model, fix_dict) + lp

def lnlike(pars, t, y, yerr, model, fix_dict):

    model = transit_spot_syst(pars, t, model, fix_dict)

    sigma = np.mean(yerr)
    lnL = -0.5*len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) \
                - 0.5*chi2(model, y, yerr)

    return lnL

def residuals(pars, t, y, yerr, model, fix_dict):

    model = transit_spot_syst(pars, t, model, fix_dict)
    return (y - model)**2/yerr**2

def lnp_gauss(par, mm, sigma):
    '''
    Compute ln of Gaussian distribution
    '''
    return -1.*(np.log(sigma) + 0.5*np.log(2.*np.pi) \
                + (par - mm)**2/(2.*sigma)**2)

def lnp_jeffreys(val, valmax, valmin):
    '''
    Jeffreys prior
    '''
    return np.log(1./(val*np.log(valmax/valmin)))

def lnp_sine(val, valmax, valmin):
    '''
    Sine prior (value in radians) (CHECK)
    '''
    #return np.log(np.cos(val)/(np.sin(valmax) - np.sin(valmin)))
    return 0.5*np.log(np.sin(val)/(np.cos(valmin) - np.cos(valmax)))

def lnprior(p, model):

    if model != 'KSint':
        if len(p) == 12:
            kr, q1, q2, r0, r1, r2, A, n, sig, x0, inclin, ttr = p
            if not np.logical_and.reduce((sig >= 1e-3, 0.06 < x0 < 0.15, \
                        0. <= q1 <= 1.,  0. <= q2 <= 1., \
                        80. <= inclin <= 90., 2. <= n < 10.)):
                return -np.inf
        elif len(p) == 7:
            kr, q1, q2, r0, r1, r2, A = p
            if not np.logical_and(0. <= q1 <= 1., 0. <= q2 <= 1):
                return -np.inf
        elif len(p) == 6:
            kr, r0, r1, r2, A, n = p
            q1, q2 = 0., 0.

        if np.logical_and(kr >= 0., A >= 0.):
            #lnp_kr = np.log(1./(kr*np.log(0.2/0.01))) # Jeffreys prior
            lnp_kr = lnp_jeffreys(kr, 0.2, 0.01)
            if len(p) == 12:
                #if inclin > 90.:
                #    inclin -= 2.*(inclin - 90.)
                lnp_incl = lnp_sine(np.radians(inclin), np.radians(90.), \
                            np.radians(80.))
                return lnp_kr + lnp_incl
            else:
                return lnp_kr
        else:
            return -np.inf
    else:
        if modeltype == 'fitt0':
            kr, M, q1, q2, long, aumbra, contr, incl = p
            if not np.logical_and.reduce((0.01 < kr < 0.2, 0. < q1 < 1., \
                            0. < q2 < 1., 1e-6 < aumbra <= 10., \
                            0. < contr <= 1., 80. <= incl <= 100.)):
                    return -np.inf
            else:
                lnp_kr = lnp_jeffreys(kr, 0.2, 0.01)
                return lnp_kr
        else:
            kr, q1, q2, contr = p
            if not np.logical_and.reduce((0.01 < kr < 0.2, 0. < q1 < 1., \
                            0. < q2 < 1., 0. < contr <= 1.)):
                    return -np.inf
            else:
                lnp_kr = lnp_jeffreys(kr, 0.2, 0.01)
                return lnp_kr

def starspot_size(diz, samples, channel):
    '''
    Calculate starspot angular size from Gaussian sigma fit.
    '''
    k = samples[:, 0]
    b = aR*np.cos(np.radians(inc))
    # Transit duration
    td = per_planet/np.pi*np.arcsin(aR**-1*((1 + k)**2 - b**2) \
                    /np.sin(np.radians(inc)))

    size = samples[:, -1]*90./td   # It's 180°/2 because it's the radius

    plt.figure()
    plt.hist(size, bins=30)
    plt.title('Starspot size [deg]', fontsize=16)
    #plt.show()
    plt.savefig(diz['chains_folder'] + 'spot_size' + str(channel) + '.pdf')

    return size

#def gelmanrubin(chains):
#    '''
#    Calculates convergence criterion according to Gelman-Rubin (1992).
#    This uses a random chain, as chains are not independent
#    (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr).
#    '''
#    m = np.random.randint(0., np.shape(chains)[1])
