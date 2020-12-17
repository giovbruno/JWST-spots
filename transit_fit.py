import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os, sys, pickle
sys.path.append('/home/giovanni/Shelf/python/emcee/src/emcee/')
import emcee
from autocorr import integrated_time
import batman
import cornerplot
import get_uncertainties
from pdb import set_trace
plt.ioff()

def transit_spectro(pardict, instrument, resol=10):
    '''
    Launches on all spectral bands
    '''

    if pardict['instrument'] == 'NIRSpec_Prism/':
        ldendname = 'prism'
    elif pardict['instrument'] == 'NIRCam/':
        ldendname = 'nircam'
    ldd = open(pardict['project_folder'] \
            + pardict['instrument'] + 'star_' + str(int(pardict['tstar'])) \
                        + 'K/' + 'LDcoeffs_' + ldendname + '.pic', 'rb')
    xobs = pickle.load(ldd)[1][0]
    ldd.close()
    flag = xobs < 6
    expchan = len(xobs[flag])

    # First, fit transit with the smallest error bars, to then fix the spot
    # parameters
    spec = pickle.load(open(pardict['data_folder'] + 'spec_model_' \
                + instrument + '.pic', 'rb'))
    ymoderr = spec[2]
    bestbin = ymoderr.argmin()
    transit_emcee(pardict, bestbin, bestbin)

    for i in np.arange(expchan):
        if i != bestbin:
            transit_emcee(pardict, int(i), bestbin)

    return

def transit_emcee(diz, ind, bestbin):

    print('Channel', str(ind))

    plt.close('all')
    os.system('mkdir ' + diz['chains_folder'])
    lcfile = open(diz['data_folder'] + 'transit_spots_' + str(ind) \
                    + '.pic', 'rb')
    lc = pickle.load(lcfile)
    lcfile.close()
    global wl
    t, y, yerr, wl = lc

    global per_planet
    per_planet = diz['pplanet']

    # From stellar density
    rhostar = 5.14e-5/diz['rstar']*10**diz['loggstar'] #cgs
    global aR
    aR = (6.67408e-11*rhostar*1e3*(per_planet*86400.)**2/(3.*np.pi))**(1./3.)
    global inc
    inc = 88.
    global t0
    t0 = 0.10

    # LM fit boundaries - fit for t0 and x0 only on the first channel
    bounds_model = []
    bounds_model.append((0.01, 0.2))         # rp/r*
    bounds_model.append((0.0, 1.0))          # q1 from Kipping+2013
    bounds_model.append((0.0, 1.0))          # q2
    bounds_model.append((-1., 1.))           # r0
    bounds_model.append((0., 2.))            # r1
    bounds_model.append((1e-6, 1.))          # A
    bounds_model.append((1e-3, 0.11))        # sigma
    bounds_model.append((0.08, 0.12))        # x0 = 0.1051

    # Initial values
    kr, q1, q2, r0, r1 = 0.09, 0.3, 0.3, -1e-3, 1.
    A, wspot_, tspot_ = 1e-3, 0.005, 0.1

    # This will set the fit or fix for tspot and spot size
    # Spot position and size are fitted only on the bluemost wavelength bin
    global modeltype
    if ind == bestbin:
        modeltype = 'fitt0'
        initial_params = kr, q1, q2, r0, r1, A, wspot_, tspot_
    else:
        modeltype = 'fixt0'
        # Extract spot time from first wavelength bin
        ffopen = open(diz['chains_folder'] + 'chains_' + str(bestbin) \
                    + '.pickle', 'rb')
        res = pickle.load(ffopen)
        perc = res['Percentiles'][0]
        # These will be fixed in the fit
        global wspot
        global tspot
        wspot = perc[-2][2]
        tspot = perc[-1][2]
        if wl <= 2.7:
            initial_params = kr, q1, q2, r0, r1, A
            bounds_model = bounds_model[:-2]
        else:
            initial_params = kr, r0, r1, A
            temp = []
            for kk in [0, 3, 4, 5]:
                temp.append(bounds_model[kk])
            bounds_model = temp

    # LM fit
    options= {}
    ftol = 1e-10
    options['ftol'] = ftol
    nll = lambda *args: -lnlike(*args)
    soln = minimize(nll, initial_params, jac=False, method='L-BFGS-B', \
         args=(t, y, yerr), bounds=bounds_model, options=options)
    print('Likelihood maximimazion results:')
    print(soln)
    # This contains the spot signal with the transit model removed
    plot_best(soln, t, y, yerr, wl, \
            diz['chains_folder'] + 'best_LM_' + str(ind) + '.pdf')

    # Now, MCMC starting about the optimized solution
    initial = np.array(soln.x)
    if ind == bestbin:
        ndim, nwalkers = len(initial), 64
    else:
        ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=([t, y, yerr]), threads=8)

    # Variation around LM solution
    p0 = initial + 0.1*(np.random.randn(nwalkers, ndim))*initial \
                + 1e-6*(np.random.randn(nwalkers, ndim))
    # Check condition number (must be < 1e8 to maximise walker linear
    # independence). Problems might be caued by LM results = 0
    cond = np.linalg.cond(p0)
    while cond >= 1e8:
        p0 += 0.1*(np.random.randn(nwalkers, ndim))*initial
        cond = np.linalg.cond(p0)

    print("Running burn-in...")
    nsteps = 250
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width+1)*float(i)/nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(width - n)))
    sys.stdout.write("\n")
    p0, lp, _ = result
    sampler.reset()

    print("Running production...")
    nsteps = 500
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
    except:
        print('The chain is too short')
        pass

    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

    percentiles = [np.percentile(samples[:,i],[4.55, 15.9, 50, 84.1, 95.45]) \
                        for i in np.arange(np.shape(samples)[1])]
    plot_best(best_sol, t, y, yerr, wl, \
            diz['chains_folder'] + 'best_MCMC_' + str(ind) + '.pdf')

    if wl <= 2.7 and ind == bestbin:
        titles = [r'$R_\mathrm{p}/R_\star$', r'$q_1$', r'$q_2$', \
                    r'$r_0$', r'$r_1$', r'$A_\mathrm{spot}$', \
                    r'$w_\mathrm{spot}$', '$t_\mathrm{spot}$']
    elif wl <= 2.7 and ind != bestbin:
        titles = [r'$R_\mathrm{p}/R_\star$', r'$q_1$', r'$q_2$', \
                    r'$r_0$', r'$r_1$', r'$A_\mathrm{spot}$']
    else:
        titles = [r'$R_\mathrm{p}/R_\star$', r'$r_0$', r'$r_1$', \
                    r'$A_\mathrm{spot}$']

    truths = np.concatenate(([diz['rplanet']/diz['rstar']], \
                        [None]*(len(titles) - 1)))
    ranges = [0.999]*samples.shape[1]
    if ind < 2 or ind == bestbin:
        cornerplot.cornerplot(samples, titles, truths, ranges, \
                diz['chains_folder'] + '/corner_' + str(ind) + '.pdf')
    plt.close('all')
    # Starspot size
    #size = starspot_size(diz, samples, str(ind))

    # Save chains
    fout = open(diz['chains_folder'] + '/chains_' + str(ind) \
                                                    + '.pickle', 'wb')
    chains_save = {}
    chains_save['wl'] = wl
    chains_save['LM'] = soln.x
    chains_save['Burn_in'] = [p0, lp]
    chains_save['Chains'] = samples
    #chains_save['Starspot_size'] = size
    chains_save['Mean_acceptance_fraction'] \
                                    = np.mean(sampler.acceptance_fraction)
    #chains_save['Autocorrelation_multiples'] = acor_time
    chains_save["Percentiles"] = [percentiles]
    chains_save["ln_L"] = lnL
    pickle.dump(chains_save, fout)
    fout.close()

    fout = open(diz['chains_folder'] + '/chains_best_' \
                + str(ind) + '.p', 'wb')
    pickle.dump(best_sol, fout)
    fout.close()

    return

def chi2(model, y, yerr):
    return np.sum((model - y)**2/yerr**2)

def plot_best(sol, t, y, yerr, wl, plotname):

    if type(sol) != np.ndarray: # This is the LM minimization
        params = sol.x
    else:
        params = sol
    bestmodel = transit_spot_syst(params, t)

    ln_evidence = -0.5*np.log(np.mean(yerr)) - 0.5*np.log(2.*np.pi) \
                    - 0.5*chi2(bestmodel, y, yerr)
    #BIC = np.log(len(y))*len(params) - 2.*ln_evidence
    rms = np.std((y - bestmodel)/y.max()*1e6)

    xTh = np.linspace(t.min(), t.max(), len(t)*10)
    yTh = transit_spot_syst(params, xTh)
    yTh_t = transit_spot_syst(params, t)
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

def transit_spot_syst(par, t):

    if wl <= 2.7:
        kr = par[0]
        q1 = par[1]
        q2 = par[2]
        r0 = par[3]
        r1 = par[4]
        Aspot = par[5]
        if modeltype == 'fixt0':
            sig = wspot
            x0 = tspot
        else:
            sig = par[6]
            x0 = par[7]
        model = (transit_model([kr, q1, q2], t, wl) \
                + gauss(t, [Aspot, x0, sig]))*np.polyval([r0, r1], t)
    else:
        kr = par[0]
        r0 = par[1]
        r1 = par[2]
        Aspot = par[3]
        model = (transit_model([kr], t, wl) \
                + gauss(t, [Aspot, tspot, wspot]))*np.polyval([r0, r1], t)

    return model

def transit_model(par, t, wl, u1=0, u2=0):
    '''
    # From stellar density to aR*
    #aR = (6.67408e-11*1e-3*par[2]*(per_planet*86400)**2/(3*np.pi))**(1./3.)

    # q1 and q1 (as free parameters) are Kipping's parameters
    '''

    if wl <= 2.7:
        # Back to u1, u2
        u1 = 2.*par[1]**0.5*par[2]
        u2 = par[1]**0.5*(1. - 2.*par[2])

    params = batman.TransitParams()
    params.t0 = t0
    params.per = per_planet
    params.rp = par[0]
    params.a = aR
    params.inc = inc # in degrees
    params.ecc = 0.
    params.w = 0.
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    return flux

def lnprob(params, t, y, yerr):

    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lnlike(params, t, y, yerr) + lp

def lnlike(pars, t, y, yerr):

    model = transit_spot_syst(pars, t)

    sigma = np.mean(yerr)
    lnL = -0.5*len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) \
                - 0.5*chi2(model, y, yerr)

    return lnL

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
    return np.log(np.cos(val)/(np.sin(valmax) - np.sin(np(valmin))))

def lnprior(p):

    if len(p) == 8:
        kr, q1, q2, r0, r1, A, sig, x0 = p
        if np.logical_or.reduce((sig < 0., x0 < 0.08, x0 > 0.12, q1 < 0., \
                    q1 > 1., q2 < 0., q1 > 1.)):
            return -np.inf
    elif len(p) == 6:
        kr, q1, q2, r0, r1, A  = p
        if np.logical_or.reduce((q1 < 0., q1 > 1., q2 < 0., q1 > 1.)):
            return -np.inf
    elif len(p) == 4:
        kr, r0, r1, A = p
        q1, q2 = 0., 0.

    if kr >= 0:
        lnp_kr = np.log(1./(kr*np.log(0.2/0.01))) # Jeffreys prior
        return lnp_kr
    else:
        return -np.inf

def gauss(x, par):
    '''
    Par is defined as [A, x0, sigma]
    '''
    A, x0, sigma = par
    return A*np.exp(-0.5*(x - x0)**2/sigma**2)

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
