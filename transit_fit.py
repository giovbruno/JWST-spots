# GP with Celerite.
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
import emcee
import os, sys, pickle, glob, acor, logging, pathlib
from astropy.io import fits
from emcee.autocorr import integrated_time
#from pytransit import QuadraticModel
import batman
from astropy import constants as const
from text_operations import printlog
import cornerplot

#from spotmodel_twotransits import gauss
per_planet = 2.*np.random.normal(loc=1., scale=1e-4)
# Noise and systematics parameters
psyst = [1e-3, -1e-4, 1e-3, 0.99]
rawfile = 'white_lc_raw.p'
corrected_file = 'corrected_lc.p'
flagext = [0.072, 0.088]
# Initial error for white LC model
ierrw = np.array([1e-1, 1e-1, 1.0, 1.0, 0.5, 0.5, 1., 1., 1., 0.05, 1e-1, \
                1e-2, 1e-2])
# And for spectro channels
ierrs = np.array([1e-3, 0.1, 0.1, 1., 0.05])

# For all spectral bands
def transit_spectro(pardict, expchan, instrument):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(pardict['logfile_chains'])
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Wl bins and LD
    wm, ua, eua, ub, eub = np.loadtxt(pardict['ldfile'], unpack=True, \
                        skiprows=3)

    # Band-integrated transit files in the data folder
    ll = len(glob.glob(pardict['data_folder'] + 'transits_spots*.pic'))
    #ll = len(wm) - 1
    whiteleft = wm.min() - np.diff(wm)[0]/2.
    whiteright = wm.max() + np.diff(wm)[0]/2.
    printlog('Band: ' + str(whiteleft) + '-' + str(whiteright) + ', LD:' \
                    + str(ua[-1]) + '-' + str(ub[-1]), logger)
    global ldw
    ldw = [ua[-1], ub[-1]]
    # Save results
    diz_res = {}
    for i in expchan:
        samples_blue, spot1 = transit_emcee(pardict, logger, int(i))
        diz_res[i] = spot1
    printlog('White light curve fit - OK', logger)

    filespot = open(pardict['chains_folder'] + '/spots_signal.p', 'wb')
    pickle.dump(diz_res, filespot)

    return

def transit_emcee(diz, logger, ind):

    plt.close('all')
    os.system('mkdir ' + diz['chains_folder'])
    lcfile = open(diz['data_folder'] + 'transit_spots_' + str(ind) \
                    + '_jwst.pic', 'rb')
    lc = pickle.load(lcfile)
    lcfile.close()
    t, y, yerr, wl = lc
    t -= t.min()

    # Add systematics and save for common mode correction
    # (noise calculated from PANDEXO)
    # - For computing errors
    spec_obs = pickle.load(open(diz['pandexo_out_jwst'], 'rb'))
    #scale = np.mean(spec_obs[2])/(len(spec_obs[2])**0.5)
    y, yerr = add_rnoise(y, t, np.mean(yerr), psyst)
    yerr /= y.max()
    y /= y.max()
    wfile = open(diz['data_folder'] + rawfile, 'wb')
    pickle.dump([t, y, yerr], wfile)
    wfile.close()

    '''
    # Granulation? Not included
    '''

    flag = np.logical_and(t > flagext[0], t < flagext[1])
    datasav = [t.copy(), y.copy(), yerr.copy()]

    bounds_model = []
    bounds_model.append((0.01, 0.2))
    bounds_model.append((0.08, 0.12))
    bounds_model.append((6., 9.))
    bounds_model.append((85., 90.))
    bounds_model.append((0.01, 0.99))
    bounds_model.append((0.01, 0.99))
    bounds_model.append((-1., 1.))
    bounds_model.append((-1., 1.))
    bounds_model.append((-1,  1.))
    bounds_model.append((0.95, 1.05))
    bounds_model.append((1e-6, 1e-2)) # A
    bounds_model.append((0.07, 0.09)) # x0
    bounds_model.append((1e-6, 1e-2)) # sigma

    kr, t0, aR, i, ua, ub, r0, r1, r2, C = 0.10, 0.08, 8., \
            88., ldw[0], ldw[1], -1e-3, 1e-4, -1e-4, 1.
    A, x0, sigma = 1e-2, 0.08, 1e-2

    initial_params = kr, t0, aR, i, ua, ub, r0, r1, r2, C, A, x0, sigma
    options = {}
    options['maxiter'] = 10000
    nll = lambda *args: -lnlike_white(*args)
    soln = minimize(nll, initial_params, jac=False, method='L-BFGS-B', \
                    args=(t, y, yerr), bounds=bounds_model, options=options)

    print('Likelihood maximimazion results:')
    #print(soln)

    # Initial error on parameters (prior)
    initial_err = np.copy(ierrw)
    # This contains the spot signal with the transit model removed
    spotsig, scalef = plot_best_white(soln, t, y, yerr, datasav, initial_err, \
            diz['chains_folder'] + 'best_LM_' + str(ind) + '.pdf', \
            diz['chains_folder'] + 'entropy.p', logger)

    yerr*= scalef**0.5
    #plt.errorbar(t, y, yerr = yerr, fmt = '.')
    #x = np.linspace(t.min(), t.max(), 1000)
    #plt.plot(x, transit_white(soln.x, x))
    #plt.show()

    # Now, MCMC starting almost from the optimized solution
    initial = np.array(soln.x)
    ndim, nwalkers = len(initial), 200
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_white, \
            args=([t, y, yerr]), threads=8, live_dangerously=False)

    print("Running burn-in...")
    p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
    nsteps = 1000
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width+1)*float(i)/nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#'*n, ' '*(width - n)))
    sys.stdout.write("\n")
    p0, lp, _ = result
    sampler.reset()

    print("Running production...")
    nsteps = 3000
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
        acor_time = emcee.autocorr.integrated_time(samples, c=10)
        acor_multiples = np.shape(samples)[0]/acor_time
        print('Length chains:', np.shape(samples)[0])
        print('Autocorrelation multiples:', acor_multiples)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

    percentiles = [np.percentile(samples[:,i],[4.55, 15.9, 50, 84.1, 95.45]) \
                        for i in np.arange(np.shape(samples)[1])]
    spotsig2, scalef2 = plot_best_white(best_sol, t, y, yerr, datasav, \
            initial_err, \
            diz['chains_folder'] + 'best_MCMC_' + str(ind) + '.pdf', \
            diz['chains_folder'] + 'entropy.p', logger)

    titles = [r'$R_\mathrm{p}/R_\star$', r'$t_0$', r'$a/R_\ast$', r'$i$', \
                    r'$u_a$', r'$u_b$', r'$r_0$', r'$r_1$', r'$r_2$', r'$C$', \
                    r'$A_\mathrm{spot}$', '$x_0$', r'$\sigma^2_\mathrm{spot}$']
    #truths = [0.11, None, 7.97, 90., ldw[0], ldw[1], None, None, None, \
    #                None, None, None, ]
    truths = [None, None, None, None, None, None, None, None, None, \
                    None, None, None, None]
    cornerplot.cornerplot(samples, titles, truths, diz['chains_folder'] \
                        + '/corner_' + str(ind) + '.pdf')
    # Save chains
    fout = open(diz['chains_folder'] + '/chains_' + str(ind) + '.pickle', 'wb')
    chains_save = {}
    chains_save['wl'] = wl
    chains_save['LM'] = soln.x
    chains_save['Burn_in'] = [p0, lp]
    chains_save['Chains'] = samples
    chains_save['Mean_acceptance_fraction'] \
                                        = np.mean(sampler.acceptance_fraction)
    #chains_save['Autocorrelation_multiples'] = acor_time
    chains_save["Percentiles"] = [percentiles]
    chains_save["ln_L"] = lnL
    pickle.dump(chains_save, fout)
    fout.close()

    fout = open(diz['chains_folder'] + '/chains_best_' + str(ind) + '.p', 'wb')
    pickle.dump(best_sol, fout)
    fout.close()

    return samples, spotsig

# Compute ln of Gaussian distribution
def lnp(par, mm, sigma):
    return -1.*(np.log(sigma) + 0.5*np.log(2.*np.pi) \
                + (par - mm)**2/(2.*sigma)**2)

def lnprior(p):

    kr, t0, aR, i, ua, ub, r0, r1, r2, C, A, x0, sigma = p

    # Some restrictions:
    if i <= 90. and 0 <= ua <= 1. and 0 <= ub <= 1:
                #and var > 0: #and sigma < 5e-2 #and A > 1e-5 :
        lnp_kr = lnp(kr, 0.1, ierrw[0])
        lnp_t0 = lnp(t0, 0.1, ierrw[1])
        lnp_aR = lnp(aR, 7.5, ierrw[2])
        lnp_i  = lnp(i, 90., ierrw[3])
        lnp_ua = lnp(ua, ldw[0], ierrw[4])
        lnp_ub = lnp(ub, ldw[1], ierrw[5])
        lnp_r0 = lnp(r0, 0., ierrw[6])
        lnp_r1 = lnp(r1, 0., ierrw[7])
        lnp_r2 = lnp(r2, 0., ierrw[8])
        lnp_C  = lnp(C, 1., ierrw[9])
        lnp_A  = lnp(A, 5e-3, ierrw[10])
        lnp_x0  = lnp(x0, 7e-2, ierrw[11])
        lnp_sigma = lnp(sigma, 1e-3, ierrw[12])

        return lnp_kr + lnp_t0 + lnp_i + lnp_aR + lnp_ua + lnp_ub + lnp_r0 \
                    + lnp_r1 + lnp_r2 + lnp_C + lnp_A + lnp_x0 + lnp_sigma
    else:
        return -np.inf

def chi2(model, y, yerr):
    return np.sum((model - y)**2/yerr**2)

# This calls also the information content computation
def plot_best_white(sol, t, y, yerr, datasav, \
        initial_err, plotname, namentropyf, logger):

    # Compute Jacobian around best solution
    if type(sol) != np.ndarray: # This is the LM minimization,
        params = sol.x
    else:
        params = sol
    jac = approx_derivative(transit_syst_spot, params, args = ([t]))
    bestmodel = transit_syst_spot(params, t)

    ln_evidence = -0.5*np.log(np.mean(yerr)) - 0.5*np.log(2.*np.pi) \
                    - 0.5*chi2(bestmodel, y, yerr)
    BIC = np.log(len(y))*len(params) - 2.*ln_evidence
    rms = np.std((y - bestmodel)/y.max()*1e6)

    plt.close('all')
    fig1 = plt.figure(1, figsize=(9, 7))
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    frame1 = fig1.add_axes((.1,.3,.8,.6), sharex=frame2)
    plt.setp(frame1.get_xticklabels(), visible=False)
    x = np.linspace(t.min(), t.max(), len(t)*10)
    yTh = transit_syst_spot(params, x)
    yTh_tsav = transit_syst_spot(params, datasav[0])
    yTh_t = transit_syst_spot(params, t)
    rms = np.std((y - yTh_t)/y.max()*1e6)
    chisq = chi2(yTh_t, y, yerr)/(len(y) - len(params))
    printlog('Chi2 = ' + str(chisq), logger)
    printlog('RMS = ' + str(rms) + ' ppm', logger)
    tp = (t - params[1])/per_planet
    xp = (x - params[1])/per_planet
    frame1.plot(x, yTh, 'orange')
    frame1.errorbar(t, y, yerr=yerr, fmt='k.')
    frame1.set_xlim(t.min(), t.max())
    frame1.set_ylabel('Relative flux', fontsize=16)
    frame2.errorbar(datasav[0], (datasav[1] - yTh_tsav)*1e6, \
            yerr=datasav[2]*1e6, fmt='k.')
    frame2.plot([datasav[0].min(), datasav[0].max()], [0., 0.], 'k--')
    frame2.set_xlim(t.min() - 0.002, t.max() + 0.002)
    frame2.set_ylabel('Residuals [ppm]', fontsize=16)
    frame2.set_xlabel('Time [days]', fontsize=16)
    plt.title('Transit + starspot', fontsize=18)
    plt.show()
    plt.savefig(plotname)
    #infocontent(t, y, yerr, initial_err, jac, namentropyf, logger)
    #set_trace()
    plt.close('all')

    return datasav[1] - yTh_tsav, chisq

def lnprob_white(params, t, y, yerr):

    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lnlike_white(params, t, y, yerr) + lp

def lnlike_white(p, t, y, yerr):

    model = transit_syst_spot(p, t)
    sigma = np.mean(yerr)
    lnL = -len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) \
                - 0.5*chi2(model, y, yerr)
    return lnL

# Common mode
def correct_spectro(diz):

    if os.path.isfile(corrected_file):
        print('Correction file already present! Proceed?')
        set_trace()

    plt.close('all')

    # "Observed" spectrum
    obs = pickle.load(open(diz['pandexo_out'], 'rb'))
    wl_sol = obs[0]

    residuals, wm = isolate_systematics(diz, wl_sol)
    corrected = clean_bands(diz, wl_sol, residuals, wm)

    cf = open(diz['data_folder'] + corrected_file, 'wb')
    pickle.dump(corrected, cf)
    cf.close()

    return

# Divide the spectroscopic light curves by the error model
def clean_bands(diz, wl_sol, residuals, wm):

    plt.close('all')

    # Find lightcurves in folder
    listlc = []
    for pfile in os.listdir(diz['data_folder']):
        if pfile.endswith(".p"):
            listlc.append(pfile)

    residuals['Channels'] = {}

    for ii, chanwidth in enumerate(wm):
        if ii == len(wm) - 1:
            continue
        print('Corrected binning', chanwidth)
        residuals['Channels'][chanwidth] = {}
        # Find lightcurve
        lcfile = open(diz['data_folder'] + 'transit_spots' + str(ii) + '.p', 'rb')
        lc = pickle.load(lcfile)
        lcfile.close()
        t, y, yerr = lc
        t -= t.min()

        # Add systematics and noise
        # - For computing errors
        spec_obs = pickle.load(open(diz['pandexo_out'], 'rb'))
        scale = np.mean(yerr)#spec_obs[2])*(len(spec_obs[2])**0.5)
        y, yerr = add_rnoise(y, t, scale, psyst)

        # Flag
        flag = np.logical_and(t > flagext[0], t < flagext[1])
        t, y, yerr = t[~flag], y[~flag], yerr[~flag]

        # Model for the residuals
        tr = np.asarray(residuals['Time'])
        fr = np.asarray(residuals['White'][0])[0]
        er = np.asarray(residuals['White'][1])[0]

        # Cleaned flux and error propagation
        fclean = y/(y.max()*fr)
        delta = (np.sqrt((yerr/fr)**2 + (yerr*er/(fr**2))**2))/y.max()

        # Write results
        residuals['Channels'][chanwidth] = [[], []]
        residuals['Channels'][chanwidth][0] = fclean
        residuals['Channels'][chanwidth][1] = delta

        plt.errorbar(t, fclean - ii*0.01, yerr = delta, fmt = '.')

    plt.xlabel('Time [days]', fontsize = 18)
    plt.ylabel('Flux', fontsize = 18)
    plt.show()
    set_trace()

    return residuals

# Recover the systematics form the white LC by using the
# highest likelihood solution for the white one.
def isolate_systematics(diz, wl_sol):

    plots = True

    # Dictionary for residuals & co.
    resids = {}
    resids['Limb_darkening'] = {}
    #for ind in np.arange(len(wl_sol)):
    wm, ua, eua, ub, eub = np.loadtxt(diz['ldfile'], unpack = True, skiprows = 3)
    ind = len(wm) - 1

    t, y, yerr = pickle.load(open(diz['data_folder'] + rawfile, 'rb'))

    # MCMC file
    whitefolder = diz['chains_folder']
    parameters = pickle.load(open(whitefolder + 'chains_best.p', 'rb'))
    print('Best solution')
    print(parameters)
    flag = np.logical_and(t > flagext[0], t < flagext[1])
    t, y, yerr = t[~flag], y[~flag], yerr[~flag]

    # Compute transit model and divide data for white LC residuals
    transit_model = transit_white(parameters[:6], t)
    residuals = y/transit_model
    delta = yerr/transit_model

    # Write result
    if 'Time' not in resids.keys():
        resids['Time'] = np.asarray(t)
    resids['White'] = {}
    resids['White'] = [[], []]
    resids['White'][0].append(np.asarray(residuals))
    resids['White'][1].append(np.asarray(delta))

    if plots:
        plt.close('all')
        plt.figure(1)
        plt.errorbar(t, residuals, yerr = delta, fmt = 'o')
        #plt.xlabel('Time [BJD - ' + str(tzero) + ']', fontsize = 18)
        plt.title('Residuals', fontsize = 18)
        plt.ylabel('Flux', fontsize = 18)
        plt.savefig(diz['chains_folder'] + 'commonmode_residuals.pdf')
        ff = open(diz['chains_folder'] + 'commonmode_residuals.p', 'wb')
        pickle.dump([t, residuals, delta], ff)
        ff.close()

    # Add LD coefficients to dictionary
    wm, ua, eua, ub, eub = np.loadtxt(diz['ldfile'], unpack = True, skiprows = 3)
    resids['LD'] = {}
    for ii, ww in enumerate(wm):
        resids['LD'][ww] = [ua[ii], ub[ii]]

    return resids, wm

def mcmc_spectro(diz, logger):

    # Same as for white channels, but with less free parameters
    # Only kr and linear function
    fspectra_file = open(diz['data_folder'] + corrected_file, 'rb')
    fspectra = pickle.load(fspectra_file, encoding='latin1')
    fspectra_file.close()
    plt.ioff()

    for ii, ww in enumerate(fspectra['Channels'].keys()):
        if ww <= 2.8:
            continue
        chains_folder_ch = diz['chains_folder'] + str(ww) + '/'
        if not pathlib.Path(chains_folder_ch).exists():
            os.system('mkdir ' + chains_folder_ch)

        t    = fspectra['Time']
        y    = fspectra['Channels'][ww][0]
        yerr = fspectra['Channels'][ww][1]
        ld = fspectra['LD'][ww]

        whitefolder = diz['chains_folder']
        kr, t0, aR, i, ua, ub, r0, r1, r2, C = pickle.load(open(whitefolder + 'chains_best.p', 'rb'), encoding='latin1')
        p0 = kr, ld[0], ld[1], 0., 1.
        soln, sampler, yerr = run_emcee(t, y, yerr, p0, t0, aR, i, ld, lnlike_spectro, 100, 200, 1000)
        analyze_chains(soln, sampler, chains_folder_ch, t, t0, aR, i, ld, y, yerr, ww, diz, logger)
    plt.ion()
    return

def run_emcee(t, y, yerr, p0, t0, aR, i, ld, like_funct, nwalkers, it_burnin, it_production):

    bounds_model = []
    bounds_model.append((0.01, 0.2))
    bounds_model.append((0.01, 0.99))
    bounds_model.append((0.01, 0.99))
    bounds_model.append((-1., 1.))
    bounds_model.append((0.5, 1.5))
    nll = lambda *args: -like_funct(*args)
    soln = minimize(nll, p0, jac = False,
                method = 'L-BFGS-B', args = (t, t0, aR, i, y, yerr), bounds = bounds_model)
    print('Likelihood maximimazion results:')
    print(soln)
    print("LM log-likelihood: {0}".format(-soln.fun))
    # Rescale uncertainties
    #chi2r = chi2(transit_slope(soln.x, t, t0, aR, i, ld), y, yerr)/(len(y) - len(p0))
    #yerr = yerr * (chi2r**0.5)
    #print('Uncertainties rescalilng factor: ', str(chi2r**0.5))

    # Now, MCMC starting almost form the optimized solution
    initial = np.array(soln.x)
    ndim = len(initial)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_spectro, args = ([t, t0, aR, i, ld, y, yerr]),
                threads = 8, live_dangerously = False)
    print("Running burn-in...")
    p0 = initial + 1e-2 * np.random.randn(nwalkers, ndim)
    nsteps = it_burnin
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations = nsteps)):
        n = int((width+1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    sys.stdout.write("\n")
    p0, lp, _ = result
    sampler.reset()

    print("Running production...")
    nsteps = it_production
    width = 30
    for i, result in enumerate(sampler.sample(p0, iterations = nsteps, thin = 10)):
        n = int((width+1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
    sys.stdout.write("\n")
    pfin, lpfin, _ = result

    return soln, sampler, yerr

def analyze_chains(soln, sampler, chains_folder_ch, t, t0, aR, i, ld, y, yerr, binwl, diz, logger):

    # Merge in single chains
    samples = sampler.flatchain
    lnL = sampler.flatlnprobability

    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

    # Autocorrelation time
    try:
        acor_time = emcee.autocorr.integrated_time(samples, c = 10)
        acor_multiples = np.shape(samples)[0]/acor_time
        print('Length chains:', np.shape(samples)[0])
        print('Autocorrelation multiples:', acor_multiples)
    except emcee.autocorr.AutocorrError as e:
        print(str(e))

    # Compute percentiles
    percentiles = [np.percentile(samples[:,i],[4.55, 15.9, 50, 84.1, 95.45]) for i in np.arange(np.shape(samples)[1])]
    # Save chains
    fout = open(chains_folder_ch + 'chains.p', 'wb')
    chains_save = {}
    chains_save['max ML']                   = soln.x
    chains_save['chains']                   = samples
    chains_save['Mean_acceptance_fraction'] = np.mean(sampler.acceptance_fraction)
    #chains_save['Autocorrelation_time']     = acor_time
    chains_save["Percentiles"]              = [percentiles]
    chains_save["ln_L"]                     = lnL
    pickle.dump(chains_save, fout)
    fout.close()

    # Corner plot
    titles = ['$R_\mathrm{p}/R_\star$', '$u_a$', '$u_b$', '$r_0$', '$C$']
    truths = [None, ld[0], ld[1], None, None]
    cornerplot.cornerplot(samples, titles, truths, chains_folder_ch + 'corner.pdf')

    # Plot best solution
    best_sol = samples[lnL.argmax()]
    initial_err = np.copy(ierrs)
    plot_best_spectro(best_sol, t, t0, aR, i, y, yerr, initial_err, chains_folder_ch + 'best_MCMC.pdf', binwl, chains_folder_ch + 'entropy.p', logger)

    fout = open(chains_folder_ch + 'chains_best.p', 'wb')
    pickle.dump(best_sol, fout)
    fout.close()

    plt.close('all')
    return

def plot_best_spectro(params, t, t0, aR, i, y, yerr, initial_err, nameplot, binwl, diz, logger):

    plt.close('all')
    bestmodel = transit_slope(params, t, t0, aR, i)
    rms = np.std((y - bestmodel)/y.max()*1e6)
    ln_evidence = chi2(bestmodel, y, yerr)/(len(y) - len(params))
    BIC = np.log(len(y))*len(params) - 2.*ln_evidence
    AIC = 2*(len(params)) - 2./ln_evidence
    #AIC = -2*ln_evidence
    AICc = AIC - 2*len(params)*(len(params) - 2*len(y) - 1)/(len(params) - len(y) - 1)
    printlog('Fit on channel ' + str(binwl), logger)
    printlog('Chi2 = ' + str(chi2(bestmodel, y, yerr)/(len(y) - len(params))), logger)
    printlog('RMS = ' + str(rms) + ' ppm', logger)
    printlog('BIC = ' + str(BIC), logger)
    printlog('AIC = ' + str(AIC), logger)
    tTh = np.linspace(t.min(), t.max(), 500)
    bestmodel = transit_slope(params, t, t0, aR, i)
    bestmodel_Th = transit_slope(params, tTh, t0, aR, i)
    fig1 = plt.figure(1, figsize = (10, 7))
    ax2 = fig1.add_axes((.1,.1,.8,.2))
    ax1 = fig1.add_axes((.1,.3,.8,.6), sharex = ax2)
    plt.setp(ax1.get_xticklabels(), visible = False)
    ax1.errorbar(t, y, yerr = yerr, fmt = 'k.', capsize = 2)
    ax1.plot(tTh, bestmodel_Th, color = "orange")
    #ax1.set_ylim(0.967, 1.003)
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylabel('Normalized flux', fontsize = 18)
    ax2.errorbar(t, (y - bestmodel)*1e6, yerr = yerr*1e6, capsize = 2, fmt = 'k.')
    ax2.plot([t.min(), t.max()], [0., 0.], 'k--', color = 'grey')
    ax2.set_xlim(t.min() - 0.002, t.max() + 0.002)
    #ax2.set_ylim(-1000, 1000)
    ax2.set_xlabel('Time [BJD]', fontsize = 18)
    plt.locator_params(axis='y', nticks=3)
    ax2.set_ylabel('Residuals [ppm]', fontsize = 18)
    plt.savefig(nameplot)

    # Compute jacobian
    jac = approx_derivative(transit_slope, params, args = ([t, t0, aR, i]))
    infocontent(t, y, yerr, initial_err, jac, diz, logger)

    return

def plot_channels():

    # Same as for white channels, but with less free parameters
    # Only kr and linear function

    fspectra_file = open(corrected_file, 'rb')
    fspectra = pickle.load(fspectra_file, encoding='latin1')
    fspectra_file.close()
    plt.ioff()
    fig = plt.figure(figsize = (7, 8))
    ax = plt.subplot(111)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    #ax2.yaxis.tick_right()
    build_histo = []
    kr, t0, aR, i, ua, ub, r0, r1, r2, r3, r4, C = pickle.load(open(folder + target + '/MCMC/ima1_emcee/spotmasked_bkglow_ldnew/chains_best.p', 'rb'), encoding='latin1')
    for j, ff in enumerate(os.listdir(chains_folder)):
        if os.path.isdir(chains_folder + ff):
            t    = fspectra['Time_[BJD]']
            flag = np.logical_and(t > flagext[0], t < flagext[1])
            tTh = np.linspace(t.min(), t.max(), 1000)
            phase = (t - t0)/per_planet
            phaseTh = (tTh - t0)/per_planet
            y    = fspectra[chanwidth][ff][0]
            yerr = fspectra[chanwidth][ff][1]
            ax1.errorbar(phase, y - 0.008*j, yerr = yerr, fmt = 'k.', capsize = 2)
            ld = fspectra['Limb_darkening'][chanwidth][ff]
            params = pickle.load(open(chains_folder + ff + '/chains_best.p', 'rb'), encoding='latin1')
            model = transit_slope(params, tTh, t0, aR, i, ld)
            ax1.plot(phaseTh, model - 0.008*j, color='blue')
            #ax1.locator_params(nbins=3)
            build_histo.append((y - transit_slope(params, t, t0, aR, i, ld))/y.max())
            #plt.figure(3)
            #plt.plot(t, y - transit_slope(params, t, t0, aR, i, ld), 'o')
            ax2.errorbar(phase, ((y - transit_slope(params, t, t0, aR, i, ld)) - 0.008*j) + 1., yerr = yerr, fmt = 'k.', capsize = 2)
            ax2.plot([phase.min(), phase.max()], np.asarray([-0.008*j, -0.008*j]) + 1, '--', color = 'blue')
        else:
            j -= 1
    plt.setp(ax2, yticklabels=[])
    #ax2.locator_params(nbins=3)
    ax1.set_ylabel('Relative flux', fontsize = 18)
    plt.subplots_adjust(wspace = 0.01, hspace = None)
    ax1.set_xlim(-0.045, 0.055)
    ax2.set_xlim(-0.045, 0.055)
    ax1.set_ylim(0.81, 1.0)
    ax2.set_ylim(0.81, 1.0)
    #ax.set_xlabel('ll')
    fig.text(0.5, 0.05, 'Orbital phase', ha='center', va='center', fontsize = 18)
    fig.text(0.5, 0.94, 'Spot masked', ha='center', va='center', fontsize = 20)
    plt.show()
    set_trace()
    plt.close('all')
    reshist = np.concatenate((build_histo))
    #plt.hist(reshist, alpha = 0.5, bins = 20)
    plt.title('Residuals', fontsize = 1)
    plt.show()
    #set_trace()
    fout = open(chains_folder + 'residuals.pic', 'wb')
    pickle.dump(np.concatenate((build_histo)), fout)
    return

def lnprob_spectro(params, t, t0, aR, i, ld, y, yerr):

    kr, ua, ub, r0, C = params

    # Only uniform priors on the transit depth
    if kr > 0.01 and 0.01 < ua < 0.99 and 0.01 < ub < 0.99:
        lnp_kr = lnp(kr, 0.1, ierrs[0])
        lnp_ua = lnp(ua, ld[0], ierrs[1])
        lnp_ub = lnp(ub, ld[1], ierrs[2])
        lnp_r0 = lnp(r0, 0., ierrs[3])
        lnp_C  = lnp(C, 1., ierrs[4])
        return lnlike_spectro(params, t, t0, aR, i, y, yerr) + lnp_kr + lnp_ua + lnp_ub + lnp_r0 + lnp_C
    else:
        return -np.inf

def transit_slope(params, t, t0, aR, i):
    return transit_aRi(params[:-2], t, t0, aR, i)*np.polyval(params[-2:], t)

def lnlike_spectro(params, t, t0, aR, i, y, yerr):

    model = transit_slope(params, t, t0, aR, i)
    sigma = np.mean(yerr)

    return -len(y)*np.log(sigma) - 0.5*len(y)*np.log(2.*np.pi) - 0.5*chi2(model, y, yerr)

# Mandel-Agol (2002) model for the transit (PyTransit/batman)
def transit_aRi(par, t, t0, aR, i):

    params = batman.TransitParams()
    params.t0 = t0                    #time of inferior conjunction
    params.per = per_planet           #orbital period
    params.rp = par[0]                #planet radius (in units of stellar radii)
    params.a = aR                     #semi-major axis (in units of stellar radii)
    params.inc = par[4]               #orbital inclination (in degrees)
    params.ecc = 0.                   #eccentricity
    params.w = 0.                     #longitude of periastron (in degrees)
    params.u = [par[1], par[2]]       #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"    #limb darkening model

    m = batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve


    #k, p, t0, a, i, e, w, u = par[0], per_planet, t0, aR, np.radians(i), \
                                #0., 0., [par[1], par[2]]
    #tm = QuadraticModel()
    #tm.set_data(t)
    #flux = m.evaluate_tm(t, k, u, t0, p, a, i, e, w)

    return flux

def transit_white(par, t):

    #p, a, i, e, w, u = per_planet, par[2], np.radians(par[3]), 0., 0., [par[4], par[5]]#ld
    #k, t0 = par[0], par[1]
    #m = MandelAgol()
    #flux = m.evaluate(t, k, u, t0, p, a, i, e, w)

    params = batman.TransitParams()
    params.t0 = par[1]                #time of inferior conjunction
    params.per = per_planet           #orbital period
    params.rp = par[0]                #planet radius (in units of stellar radii)
    params.a = par[2]                 #semi-major axis (in units of stellar radii)
    params.inc = par[3]               #orbital inclination (in degrees)
    params.ecc = 0.                   #eccentricity
    params.w = 0.                     #longitude of periastron (in degrees)
    params.u = [par[4], par[5]]       #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"    #limb darkening model

    m = batman.TransitModel(params, t)    #initializes model
    flux = m.light_curve(params)          #calculates light curve

    return flux

def expramp_simple(par, t, shift):

    per_ramp = 50.
    r0, r1, r2, r3, r4, C = par
    theta = 2.*np.pi*(t % per_planet)/per_planet
    phi   = 2.*np.pi*((t + shift) % per_ramp)/per_ramp

    S = C*(1 + r0 * theta + r1 * theta**2) * \
            (1 - np.exp(r2 * phi + r3) + r4 * phi)
    return S

def transit_syst(par, t):
    return transit_white(par[:6], t) * np.polyval(par[6:], t)

def transit_syst_spot(par, t):

    model = (transit_white(par[:6], t) \
                    + gauss(t, par[-3:]))*np.polyval(par[6:-3], t)
    return model

# From Line+2012, Batalha+2017
def infocontent(tTh, yTh, yerr, initial_err, jac, namentropyf, logger):

    # 0. Jacobian matrix of transit function wrt time
    #jac = np.transpose(jac)
    #jac = np.gradient(yTh, np.diff(tTh)[0])
    # 1. Data covariance matrix.
    # - Create matrix, assign diagonal elements
    Se = np.matrix( np.zeros(len(yerr)**2).reshape(len(yerr), len(yerr)) )
    diag_e = np.diag_indices(len(yerr))
    Se[diag_e] = yerr**2
    # 2. Prior covariance matrix
    Sa = np.matrix( np.zeros(len(initial_err)**2).reshape(len(initial_err), len(initial_err)) )
    diag_a = np.diag_indices(len(initial_err))
    Sa[diag_a] = initial_err**2
    # 3. Posterior covariance matrix
    Scirc = np.matrix( np.linalg.inv( np.transpose(jac) * np.linalg.inv(Se) * jac + np.linalg.inv(Sa) ) )

    # Entropy
    H = 0.5 * np.log(np.linalg.det( np.linalg.inv(Scirc) * Sa ))
    printlog('H = '+ str(H) + ' nats', logger)
    # Save result
    dictentr = {}
    dictentr['Se'] = Se
    dictentr['Sa'] = Sa
    dictentr['Scirc'] = Scirc
    dictentr['H'] = H
    resfile = open(namentropyf, 'wb')
    pickle.dump(dictentr, resfile)
    resfile.close()

    return

def add_rnoise(x, t, wscale, parsyst):
    x *= np.random.normal(loc=1.0, scale=wscale, size=len(x))
    xerr = wscale*np.random.normal(loc=1.0, scale=0.05, size=len(x))
    # Systematics
    x *= np.polyval(parsyst, t)
    return x, xerr

def gauss(x, par):
    '''
    Par is defined as [A, x0, sigma]
    '''
    A, x0, sigma = par
    return A*np.exp(-0.5*(x - x0)**2/sigma**2)
