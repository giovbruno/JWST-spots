import numpy as np
import simulate_transit
import transit_fit
import spectra_fit
import os
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import interp1d
import scipy
from dynesty import utils as dyfunc
import sys
from pdb import set_trace

def go(magstar, pardict, operation, models, res=10, fittype='grid', \
        spotted_starmodel=True, model='KSint', noscatter=False, \
        tight_ld_prior=False, oldf2f1=False, resume=False, realization=0):

    # Folders & files (add to the input dictionary)
    #pardict = {}
    pardict['homedir'] = os.path.expanduser('~')
    pardict['project_folder'] = pardict['homedir'] \
                                            + '/Projects/jwst_spots/revision2/'
    #pardict['instrument'] = str(instr) + '/'
    if not os.path.exists(pardict['project_folder'] + pardict['instrument'] + '/'):
        os.mkdir(pardict['project_folder'] + pardict['instrument'] + '/')
    if int(pardict['latspot']) == 21:
        latpart = '_lat' + str(int(pardict['latspot']))
    else:
        latpart = ''
    if noscatter:
        scatterpart = '_noscatter'
    else:
        scatterpart = ''
    if tight_ld_prior:
        ldpart = '_tightLD'
    else:
        ldpart = ''
    if oldf2f1:
        mpart = '_oldf2f1'
    else:
        mpart = ''
    if realization != 0:
        rpart = '_' + str(realization)
    else:
        rpart = ''

    pardict['case_folder'] = pardict['project_folder'] \
            + pardict['instrument'] + '/star_' + str(int(pardict['tstar'])) \
            + 'K/p' + str(np.round(pardict['rplanet'], 4)) \
            + '_star' + str(pardict['rstar']) \
            + '_' + str(int(pardict['tstar'])) + '_' \
            + str(pardict['loggstar']) + '_spot' + str(int(pardict['tumbra'])) \
            + '_i' + str(int(pardict['incl'])) + '_a' \
            + str(int(pardict['aumbra'])) + latpart \
            + '_theta' + str(int(pardict['theta'])) \
            + '_mag' + str(magstar) + scatterpart + ldpart + mpart \
            + rpart + '/'

    if not os.path.exists(pardict['case_folder']):
        #if resume:
        #    return pardict
        os.mkdir(pardict['case_folder'])
    pardict['data_folder'] = pardict['case_folder'] + 'simulated_data/'
    if not os.path.exists(pardict['data_folder']):
        os.mkdir(pardict['data_folder'])
    pardict['chains_folder']  = pardict['case_folder'] + 'MCMC/'
    if not os.path.exists(pardict['chains_folder']):
        os.mkdir(pardict['chains_folder'])
    pardict['observatory'] = 'jwst'
    #pardict['pandexo_out_hst'] = pardict['data_folder'] + 'singlerun_hst.p'

    pardict['spotted_starmodel'] = spotted_starmodel
    pardict['magstar'] = magstar
    #pardict['muindex'] = -1
    pardict['pplanet'] = 2. # K, M star

    pardict['planettype'] = 'constant'
    if pardict['instrument'] == 'NIRSpec_Prism':
        pardict['pandexo_out_jwst'] = pardict['data_folder'] + 'singlerun_1transit.p'
        pardict['ldfile_quadratic'] = pardict['project_folder'] + 'ldfiles/' \
            + str(int(pardict['tstar'])) + '_' \
            + str(np.round(pardict['loggstar'], 1)) \
            + '_0.0_tophat_30ch_quadratic_kurucz.txt'
        pardict['ldfile_nonlinear'] = pardict['project_folder'] + 'ldfiles/' \
            + str(int(pardict['tstar'])) + '_' \
            + str(np.round(pardict['loggstar'], 1)) \
            + '_0.0_tophat_30ch_nonlinear_kurucz.txt'
    elif pardict['instrument'] == 'NIRCam':
        #pardict['pandexo_out_jwst'] = \
        #'/home/giovanni/Projects/jwst_spots/NIRCam/star_5000K/PandExo_F322W2_k3.5.pic'
        pardict['ldfile_quadratic_blue'] = pardict['project_folder'] \
            + 'ldfiles/' + str(int(pardict['tstar'])) + '_' \
            + str(np.round(pardict['loggstar'], 1)) \
            + '_0.0_F150W2_15ch_quadratic_phoenix.txt'
        pardict['ldfile_quadratic_red'] = pardict['project_folder'] \
            + 'ldfiles/' + str(int(pardict['tstar'])) + '_' \
            + str(np.round(pardict['loggstar'], 1)) \
            + '_0.0_F322W2_15ch_quadratic_phoenix.txt'

    # Run process for JWST
    if 'simulate_transits' in operation:
        if pardict['instrument'] == 'NIRSpec_Prism':
            simulate_transit.generate_spectrum_jwst(pardict, models)
            totchan = simulate_transit.add_spots(pardict, resol=res, \
                    models=models, noscatter=noscatter)
        elif pardict['instrument'] == 'NIRCam':
            if pardict['tstar'] == 3500:
                rsttemp = 0.3
                rpltemp = 0.3
            elif pardict['tstar'] == 5000:
                rsttemp = 1.0
                rpltemp = 1.0
            uncfile1 = pardict['project_folder'] + pardict['instrument'] \
                + '/dhs_snr_calcs/spec_p' \
                + str(rpltemp) + '_star' + str(rsttemp) \
                + '_' + 'k' + str(magstar).replace('.', 'p') + '.dat'
            w1, y1, yerr1 = np.loadtxt(uncfile1, unpack=True)
            uncfile2 = pardict['project_folder'] + pardict['instrument'] \
                + '/grism_snr_calcs/spec_p' \
                + str(rpltemp) + '_star' + str(rsttemp) \
                + '_' + 'k' + str(magstar).replace('.', 'p') + '.dat'
            w2, y2, yerr2 = np.loadtxt(uncfile2, unpack=True)
            wmod = np.concatenate((w1, w2))
            ymod = np.concatenate((y1, y2))
            yerrmod = np.concatenate((yerr1, yerr2))
            totchan = simulate_transit.add_spots(pardict, resol=res, \
                simultr=list([wmod, ymod, yerrmod]), models=models, \
                noscatter=noscatter)

    # Select channels and fit transit + spots
    if 'fit_transits' in operation:
        transit_fit.transit_spectro(pardict, resol=res, model=model, \
                    tight_ld_prior=tight_ld_prior, resume=resume)
    # Fit derived transit depth rise with stellar models
    if 'fit_spectra' in operation:
        spectra_fit.read_res(pardict, pardict['chains_folder'] \
          + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res_', \
          models, resol=res, model=model, mcmc=False, nest=True, resume=resume)

    # Now, for HST - requires ramp calculation but it's missing in the tutorials
    #simulate_transit.generate_spectrum_hst(pardict)
    #simulate_transit.add_spots(pardict, 'hst')
    #expchan = np.arange(1, 14)
    #transit_fit.transit_spectro(pardict, expchan, 'hst')
    #spectra_fit.read_res(pardict, expchan, 'hst', pardict['chains_folder'] \
    #        + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res.pic')

    return pardict

def cycle(rplanet, rstar, tstar, loggstar, instrum, mags=[4.5], \
            simulate_transits=True, fit_transits=True, fit_spectra=True, \
            models='josh', res=10, fittype='grid', \
            spotted_starmodel=False, inputpars={}, update=True, \
            chi2rplot=False, model='KSint', noscatter=False, \
            tight_ld_prior=False, onlyres=False, oldf2f1=False, type_res=7, \
            resume=False, realization=0):
    '''
    Run simulations for several scenarios.

    Parameters
    ----------
    models: 'phoenix', 'ck04models', or 'josh'
    inputpars: provides incl, mu, spot size

    update: overwrite previous results?
    '''

    # Stellar/planet pars
    ip = {}
    ip['rplanet'] = rplanet #f1.0 for 5000 K star, 0.3 for M star
    ip['rstar'] = rstar
    ip['tstar'] = tstar
    ip['loggstar'] = loggstar
    ip['tpenumbra'] = 5500
    ip['aumbra'] = inputpars['aumbra']
    ip['incl'] = inputpars['incl']
    ip['theta'] = inputpars['theta']
    ip['instrument'] = instrum
    ip['latspot'] = inputpars['latspot']

    if instrum == 'NIRCam':
        mags = [4.5, 6.0, 7.5, 9.0]
        #mags = [4.5]
    elif instrum == 'NIRSpec_Prism':
        #mags = np.linspace(10.5, 14.5, 5)
        mags = np.array([13.5])
    if tstar == 5000:
        # Read all Josh's models, simulte only every other two
        tcontrast = np.arange(-1400, 0, 100)
        #ip['aumbra'] = 3.
        #tcontrast = np.array([-1200.])
    elif tstar == 3500:
        tcontrast = np.arange(-1200, 0, 100)
        #ip['aumbra'] = 3.
        #tcontrast = np.array([-600.])
    opers = []
    if simulate_transits:
        opers.append('simulate_transits')
    if fit_transits:
        opers.append('fit_transits')
    if fit_spectra:
        opers.append('fit_spectra')

    if update:
        homef = os.path.expanduser('~')
        checkf = homef + '/Projects/jwst_spots/revision2/' + instrum \
                + '/star_' + str(int(tstar)) + 'K/accuracy_' + models \
                + '_a' + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pdf'
        if os.path.isfile(checkf):
            return

    if not onlyres:
        dict_starmodel, dict_spotmodels \
                        = ingest_stellarspectra(tstar, tcontrast, loggstar)

        ip['starmodel'] = dict_starmodel
        ip['spotmodels'] = dict_spotmodels

        # Get mu index in Josh models
        if models == 'josh':
            muval = np.cos(np.radians(ip['theta']))
            ip['muindex'] = abs(ip['starmodel']['mus'] - muval).argmin()

    ## Try only one Tspot
    if tstar == 3500:
    # Very cool models will only be used for the fit
        #tcontrast = np.arange(-900, 0, 100)
        tcontrast = np.array([-300.])
    elif tstar == 5000.:
        tcontrast = np.arange(-1200, 0, 100)
        #tcontrast = np.array([-1200.])

    if not onlyres:
        if len(opers) > 0:
            for mag in mags:
                for td in tcontrast[::3]:
                    ip['tumbra'] = tstar + td
                    pardict = go(mag, ip, opers, models, res=res, \
                        fittype=fittype, \
                        spotted_starmodel=spotted_starmodel, model=model, \
                        noscatter=noscatter, tight_ld_prior=tight_ld_prior, \
                        oldf2f1=oldf2f1, resume=resume, realization=realization)

    if type_res == 4:
        plot_res4(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior)
    elif type_res == 5:
        plot_res5(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior)
    elif type_res == 6:
        plot_res6(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior)
    elif type_res == 7:
        plot_res7(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior, \
        noscatter=noscatter)

    #plot_size(ip, mags, tcontrast, models, fittype, chi2rplot=chi2rplot)
    #plot_res2(ip, mags, tcontrast, models, fittype, chi2rplot=chi2rplot)
    #plot_res3(ip, mags, tcontrast, models)
    #plot_res5(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior)
    #plot_res6(ip, mags, tcontrast, models, tight_ld_prior=tight_ld_prior)
    #map_uncertainties(mags, tcontrast, ip)
    #plot_unc_results(instrum, ip)

    plt.close('all')

    return

def plot_res(inputpars, mags, tcontrast, models, fittype, chi2rplot=False):
    '''
    Build a 2d plot containing stellar mag, t diff input as axes and
    tdiff_output as color.
    '''

    #mag = 10.5
    # Arrays for resolution. Here we are just renaming
    if models == 'josh':
        tcontrast = tcontrast[::3]
    if inputpars['tstar'] == 3500:
        tcontrast = np.arange(-900, 0, 300)
        td = -600.
        size = 3.
    elif inputpars['tstar'] == 5000:
        tcontrast = np.arange(-1200, 0, 300)
        size = 5.
        #td = -900.
    ip = inputpars
    xmag = mags
    ytdiff = tcontrast
    if chi2rplot:
        xmag = [4.5]
        ytdiff = [td]
    tdiff_output = np.zeros((len(xmag), len(ytdiff)))
    tdiff_unc = np.zeros(len(xmag))
    sizes = np.arange(2, 6)
    cc = ['g', 'r', 'b', 'm', 'c', 'y']
    fig, ax = plt.subplots()#figsize=[12, 6])
    axins = inset_axes(ax, width=3., height=3., loc=2)  #
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    plt.xticks(visible=False)
    #plt.yticks(visible=False)
    for mod in [models]:
        for i, mag in enumerate(mags):
        #for i, size in enumerate(sizes):
            ii = np.copy(i)
            i = 0
            uncT = []
            diffT = [] # This is for the 2D plot
            for j, td in enumerate(tcontrast): #td
                tumbra = ip['tstar'] + td
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/Projects/jwst_spots/revision2/'
                instrument = ip['instrument'] + '/'
                chains_folder = project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) \
                    + '_i' + str(int(ip['incl'])) \
                    + '_a' + str(int(size)) \
                    + '_theta' + str(int(ip['theta'])) \
                    + '_mag' + str(mag) + '/MCMC/'
                try:
                    resfile = open(chains_folder + 'contrast_res_' \
                            + mod + '_' + fittype + '.pic', 'rb')
                    resdict = pickle.load(resfile)
                    resfile.close()
                    tbest = sorted(zip(resdict[mag][mod].items()), \
                                    key=lambda x: x[0][1])[0][0][0]
                    #tdiff_output[i, j] = -(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output_abs[i, j] = abs(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output[i, j] = -(td - tbest)
                    #tdiff_output[ii, j] = resdict[mag]['Tunc'][0]
                    #if tdiff_output[ii, j] < 0:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][0]
                    #else:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][1]
                    tdiff_output[ii, j] = resdict[mag]['Tunc'][1][1] \
                                - resdict[mag]['Tunc'][1][0]
                    #                - (ip['tstar'] + td)
                    diffT.append(abs(resdict[mag]['Tunc'][4]))
                    uncT.append(resdict[mag]['Tunc'][3])
                    if np.isnan(tdiff_output[i, j]) \
                                or tdiff_output[i, j] == np.inf:
                        tdiff_output[i, j] = 10000
                except FileNotFoundError:
                    tdiff_output[i, j] = -999
                if chi2rplot:
                    chi2r = resdict[mag]['Tunc'][-2]
                    #x, like = resdict[mag]['Tunc'][2], resdict[mag]['Tunc'][3]
                    spotSNR = resdict[mag]['Tunc'][-1]
                    print('Spot SNR = ', spotSNR)
                    ax.plot(tcontrast[:-4], chi2r[:-4]/chi2r.min(), '.-', \
                                label=str(mag), color=cc[ii])
                                #label=str(round(spotSNR)), color=cc[ii], \
                                #markersize=size*2)
                    axins.plot(tcontrast[:-4], np.zeros(len(tcontrast[:-4])) \
                                + 11.62/chi2r.min() + 1., color=cc[ii])
                    axins.plot(tcontrast[:-4], chi2r[:-4]/chi2r.min(), '.-', \
                                label=str(mag), color=cc[ii])
                    ax.plot(x, like/like.max(), '.-', label=str(mag), \
                                color=cc[ii])
                    ax.plot([td, td], [0., 1.], 'k')
                            #label=str(round(spotSNR)), color=cc[ii], \
                            #markersize=size*2)
            tdiff_unc[i] = np.mean(uncT)
            #plt.figure(33)
            #plt.plot(tcontrast, diffT, 'o-', c=cc[i], label=str(mag))
        #plt.legend()
        #plt.plot(tcontrast, np.zeros(len(tcontrast)) + 0.1, 'k--')
        #plt.plot(tcontrast, np.zeros(len(tcontrast)) + 0.3, 'k--')
        if chi2rplot:
            ax.text(-1100, 7, r'True $T_\mathrm{spot} - T_\mathrm{star}$', \
                            fontsize=14)
            #axins.text(-1100, 1.6, '$99.7\%$ CI', fontsize=14)
            ax.legend(frameon=False, loc='upper right', fontsize=14, \
                        title='K mag', title_fontsize=14)
            axins.plot([td, td], [0., 10.], 'k--')
            #ax.text(-300., 28., 'K mag', fontsize=14)
            x1, x2, y1, y2 = -1200., -100., 0.9, 3.
            axins.yaxis.tick_right()
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            ax.set_xlabel(r'$T_\mathrm{spot} - T_\mathrm{star}$ [K]', fontsize=16)
            #ax.set_ylabel('$\chi^2/\chi^2_\mathrm{min}$', fontsize=16)
            ax.set_ylabel('$Likelihood$', fontsize=16)
            ax.set_title(str(int(ip['tstar'])) + ' K star, ' \
                    + str(int(ip['tstar'] + td)) + r' K spot, $\theta=$' \
                    + str(int(ip['theta'])) + ' $\deg$, ' \
                    + ip['instrument'].replace('_', ' '), fontsize=16)
            plt.show()
            plt.savefig(project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/likelihood_map_' + mod + '_' + str(td) \
                    #+ '_asizes' + str(int(ip['aumbra'])) \
                    + '_theta' + str(int(ip['theta'])) + '.pdf')
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.13, 0.77, 0.77])
        mm = ax.imshow(tdiff_output.T[::-1], extent=(min(xmag), \
                    max(xmag) + np.diff(xmag)[0], \
                    ip['tstar'] + ytdiff.min(), ip['tstar']), \
                    # + ytdiff.max()),
                    aspect='auto', \
                    #vmin=0., vmax=1500,
                    cmap=plt.get_cmap('plasma'))
        ax.set_xticks(np.arange(min(xmag), max(xmag) + np.diff(xmag)[0], \
                        np.diff(xmag)[0]))
        ax.set_yticks(np.arange(ip['tstar'] + ytdiff.min(), \
                        ip['tstar'], np.diff(ytdiff)[0]))
        ax.grid(color='k')
        ll = plt.colorbar(mappable=mm, ax=ax)
        ll.set_label(r'Output $\Delta T_\bullet \, [K]$', fontsize=16)
        ax.set_xlabel('$K$ mag', fontsize=16)
        ax.set_ylabel(r'Input $T_\bullet$ [K]', fontsize=16)
        plt.savefig(project_folder + instrument + 'star_' \
                + str(int(ip['tstar'])) + 'K/accuracy_' + mod \
                + '_a' + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pdf')
        #plt.subplot(212)#figure()
        #frame2.plot(mags, tdiff_unc, 'ko-')
        #frame2.set_xlabel('K mag', fontsize=16)
        #frame2.set_ylabel(r'$\sigma(T_\mathrm{spot})$ [K]', fontsize=16)
        #plt.show()
        #plt.savefig(project_folder + instrument + 'star_' \
        #        + str(int(ip['tstar'])) + 'K/accuracy_' + mod + '.pdf')

        # Save uncertainty array
        fout = open(project_folder + instrument + 'star_' \
                + str(int(ip['tstar'])) + 'K/' + 'uncertainty_array_' \
                + 'a' + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pic', 'wb')
        pickle.dump([mags, tdiff_unc], fout)
        fout.close()

    return

def map_uncertainties(mags, tcontrast, ip):
    '''
    Plot median uncertainty for PandExo output spectrum.
    '''
    unc = []
    for mag in mags:
        homedir = os.path.expanduser('~')
        project_folder = homedir + '/Projects/jwst_spots/revision2/'
        instrument = ip['instrument'] + '/'
        data_folder = project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/p' \
            + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
            + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
            + '_spot' + str(int(ip['tstar'] + tcontrast[0])) + '_' \
            + str(int(ip['tpenumbra'])) + '_a' + str(int(ip['aumbra'])) \
            + '_mag' + str(mag) + '/simulated_data/'
        specmodel = pickle.load(open(data_folder \
                                        + 'spec_model_jwst.pic', 'rb'))
        unc.append(np.median(specmodel[2])*1e6)

    plt.plot(mags, unc, 'o-')
    plt.xlabel('$K$ mag', fontsize=14)
    plt.ylabel('Median $D$ uncertainty [ppm]', fontsize=14)
    plt.show()
    plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/map_uncertainties.pdf')

    return

def plot_unc_results(instrument, ip):
    '''
    Plot delta T spot results.
    '''
    homedir = os.path.expanduser('~')
    colr = ['orange', 'royalblue']
    tspot = [3500, 5000]
    plt.figure()
    for i, ti in enumerate(tspot):
        filres = homedir + '/Projects/jwst_spots/revision2/' + instrument \
                + '/star_' + str(ti) + 'K/uncertainty_array_' \
                + 'a' + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pic'
        ff = pickle.load(open(filres, 'rb'))
        plt.plot(ff[0], ff[1], 'o-', label=str(ti) + ' K', color=colr[i])
    plt.legend(frameon=False, fontsize=12)
    plt.xlabel('$K$ mag', fontsize=16)
    plt.ylabel(r'$\sigma(T_\bullet)$ [K]', fontsize=16)
    plt.title(instrument.replace('_', ' '), fontsize=16)
    plt.show()
    plt.savefig(homedir + '/Projects/jwst_spots/revision2/' + instrument \
                + '/result_uncertainties_' + instrument + '_a' \
                + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pdf')

    return

def ingest_stellarspectra(tstar, tcontrast, loggstar, everymu=1):
    '''
    Ingest all stellar spectra needed for a given simulation (so you only have
    to do it once).

    everymu: use only every xxx mu values
    '''

    print('Ingesting stellar models...')
    josh_grid_folder = os.path.expanduser('~') \
                                        + '/Projects/jwst_spots/josh_models/'

    # For star
    dict_starmodel = {}
    tstar_ = format(tstar, '2.2e')
    loggstar_ = format(loggstar, '2.2e')
    filename = josh_grid_folder \
       + 'starspots.teff=' + tstar_ + '.logg=' + loggstar_ + '.z=0.0.irfout.csv'
    g = np.genfromtxt(filename, delimiter=',')
    mus = g[0][1:][::everymu]
    dict_starmodel['mus'] = mus
    dict_starmodel['wl'] = [g[i][0] for i in range(2, len(g))]
    dict_starmodel['spec'] = []
    for j, mm in enumerate(mus):
        spec = [g[i][j + 1] for i in range(2, len(g))]
        dict_starmodel['spec'].append(np.array(spec))

    # For starspots
    dict_spotmodels = {}
    loggspot = format(loggstar - 0.5, '2.2e')
    moldefiles = glob.glob(josh_grid_folder + '*' + loggspot + '*')
    for tc_ in tstar + tcontrast:
        tc = format(tc_, '2.2e')
        if tc_ != 2900:
            filename = josh_grid_folder \
            + 'starspots.teff=' + tc + '.logg=' + loggspot + '.z=0.0.irfout.csv'
        else:
            filename = josh_grid_folder + 'starspots.teff=' \
               + tc + '.logg=' + loggspot + '.z=0.0.irfout3.FIXED.csv'

        dict_spotmodels[tc] = {}
        g = np.genfromtxt(filename, delimiter=',')
        mus = g[0][1:][::everymu]
        dict_spotmodels[tc]['mus'] = np.array(mus)
        dict_spotmodels[tc]['wl'] = np.array([g[i][0] for i in range(2,len(g))])
        dict_spotmodels[tc]['spec'] = []
        for j, mm in enumerate(mus):
            #if j == len(mus) - 1: # Take only mu = 1
            # the spectrum at 2900 K has a different wl axis
            spec = [g[i][j + 1] for i in range(2, len(g))]
            if tc_ == 2900.:
                modint = interp1d(dict_spotmodels[tc]['wl'], spec, \
                        bounds_error=False, fill_value='extrapolate')
                spec = modint(dict_starmodel['wl'])
                # Replace wl axis just once
                if j == len(mus) - 1:
                    dict_spotmodels[tc]['wl'] = np.copy(dict_starmodel['wl'])
            dict_spotmodels[tc]['spec'].append(np.array(spec))

    return dict_starmodel, dict_spotmodels

def plot_res2(inputpars, mags, tcontrast, models, fittype, chi2rplot=False):
    '''
    Build a 2d plot containing stellar mag, t diff input as axes and
    tdiff_output as color.
    '''

    #mag = 10.5
    # Arrays for resolution. Here we are just renaming
    if models == 'josh':
        tcontrast = tcontrast[::3]
    if inputpars['tstar'] == 3500:
        tcontrast = np.arange(-900, 0, 300)
        td = -600.
        size = 3.
    elif inputpars['tstar'] == 5000:
        tcontrast = np.arange(-1200, 0, 300)
        size = 5.
        #td = -1200.
    ip = inputpars
    xmag = mags
    ytdiff = tcontrast
    #if chi2rplot:
    #    xmag = [4.5]
    #    ytdiff = [td]
    tdiff_output = np.zeros((len(xmag), len(ytdiff)))
    tdiff_unc = np.zeros(len(xmag))
    sizes = np.arange(2, 6)
    cc = ['g', 'r', 'b', 'm', 'c', 'y']
    #axins = inset_axes(ax, width=3., height=3., loc=2)  #
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    #mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    #plt.xticks(visible=False)
    #plt.yticks(visible=False)
    for mod in [models]:
        for j, td in enumerate(tcontrast): #td
            fig, ax = plt.subplots()#figsize=[12, 6])
            #for i, size in enumerate(sizes):
            uncT = []
            diffT = [] # This is for the 2D plot
            for i, mag in enumerate(mags):
                ii = np.copy(i)
                i = 0
                tumbra = ip['tstar'] + td
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/Projects/jwst_spots/revision2/'
                instrument = ip['instrument'] + '/'
                chains_folder = project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) \
                    + '_i' + str(int(ip['incl'])) \
                    + '_a' + str(int(size)) \
                    + '_theta' + str(int(ip['theta'])) \
                    + '_mag' + str(mag) + '/MCMC/'
                try:
                    resfile = open(chains_folder + 'contrast_res_' \
                            + mod + '_' + fittype + '.pic', 'rb')
                    resdict = pickle.load(resfile)
                    resfile.close()
                    tbest = sorted(zip(resdict[mag][mod].items()), \
                                    key=lambda x: x[0][1])[0][0][0]
                    #tdiff_output[i, j] = -(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output_abs[i, j] = abs(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output[i, j] = -(td - tbest)
                    #tdiff_output[ii, j] = resdict[mag]['Tunc'][0]
                    #if tdiff_output[ii, j] < 0:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][0]
                    #else:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][1]
                    tdiff_output[ii, j] = resdict[mag]['Tunc'][1][1] \
                                - resdict[mag]['Tunc'][1][0]
                    #                - (ip['tstar'] + td)
                    diffT.append(abs(resdict[mag]['Tunc'][4]))
                    uncT.append(resdict[mag]['Tunc'][3])
                    if np.isnan(tdiff_output[i, j]) \
                                or tdiff_output[i, j] == np.inf:
                        tdiff_output[i, j] = 10000
                except FileNotFoundError:
                    tdiff_output[i, j] = -999
                if chi2rplot:
                    #chi2r = resdict[mag]['Tunc'][-2]
                    x, like = resdict[mag]['Tunc'][2], resdict[mag]['Tunc'][3]
                    #like = -2.*np.log(like)
                    spotSNR = resdict[mag]['Tunc'][-1]
                    print('Spot SNR = ', spotSNR)
                    ax.plot(x, like/like.max(), '.-', label=str(mag), \
                                    color=cc[ii])
                    ax.plot([td, td], [0., 1.], 'k')
                            #label=str(round(spotSNR)), color=cc[ii], \
                            #markersize=size*2)
            #tdiff_unc[i] = np.mean(uncT)
            #plt.figure(33)
            #plt.plot(tcontrast, diffT, 'o-', c=cc[i], label=str(mag))
            #ax.text(-1100, 7, r'True $T_\mathrm{spot} - T_\mathrm{star}$', \
            #                fontsize=[14])
            #axins.text(-1100, 1.6, '$99.7\%$ CI', fontsize=14)
            ax.legend(frameon=False, loc='best', fontsize=14, \
                        title='K mag', title_fontsize=14)
            #axins.plot([td, td], [0., 10.], 'k--')
            #ax.text(-300., 28., 'K mag', fontsize=14)
            #x1, x2, y1, y2 = -1200., -100., 0.9, 3.
            #axins.yaxis.tick_right()
            #ax.set_ylim(0, 5)
            #axins.set_ylim(y1, y2)
            ax.set_xlabel(r'$T_\mathrm{spot} - T_\mathrm{star}$ [K]', fontsize=16)
            #ax.set_ylabel('$\chi^2/\chi^2_\mathrm{min}$', fontsize=16)
            ax.set_ylabel('$\mathcal{L}/\mathcal{L}_\mathrm{max}$', fontsize=16)
            #ax.set_ylabel('$\mathcal{L}$', fontsize=16)
            ax.set_title(str(int(ip['tstar'])) + ' K star, ' \
                    + str(int(ip['tstar'] + td)) + r' K spot, $\theta=$' \
                    + str(int(ip['theta'])) + ' $\deg$, ' \
                    + ip['instrument'].replace('_', ' '), fontsize=16)
            plt.savefig(project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/likelihood_map_' \
                    + ip['instrument'] \
                    + '_' + mod + '_' + str(ip['tstar']) + '_' + str(td) \
                    #+ '_asizes' + str(int(ip['aumbra'])) \
                    + '_theta' + str(int(ip['theta'])) + '.pdf')
            plt.close('all')

    return

def plot_size(inputpars, mags, tcontrast, models, fittype, chi2rplot=False):
    '''
    Build a 2d plot containing stellar mag, t diff input as axes and
    tdiff_output as color. Spot SNR goes as marker size
    '''

    mags = [10.5]
    # Arrays for resolution. Here we are just renaming
    if models == 'josh':
        tcontrast = tcontrast[::3]
    if inputpars['tstar'] == 3500:
        tcontrast = [-600]
        td = -600.
        size = [3., 4., 5.]
    elif inputpars['tstar'] == 5000:
        tcontrast = np.arange(-1200, 0, 300)
        size = 5.
        #td = -1200.
    ip = inputpars
    xmag = mags
    ytdiff = tcontrast
    #if chi2rplot:
    #    xmag = [4.5]
    #    ytdiff = [td]
    tdiff_output = np.zeros((len(size), len(ytdiff))).T
    tdiff_unc = np.zeros(len(xmag))
    sizemarker = [3, 5, 8]
    cc = ['g', 'r', 'b', 'm', 'c', 'y']
    fig, (ax1) = plt.subplots(1, 1)
    #axins = inset_axes(ax, width=3., height=3., loc=2)  #
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    #mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    #plt.xticks(visible=False)
    #plt.yticks(visible=False)
    for mod in [models]:
        for j, sz in enumerate(size): #td
            #for i, size in enumerate(sizes):
            uncT = []
            diffT = [] # This is for the 2D plot
            for i, mag in enumerate(mags):
                ii = np.copy(i)
                i = 0
                tumbra = ip['tstar'] + td
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/Projects/jwst_spots/revision2/'
                instrument = ip['instrument'] + '/'
                chains_folder = project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) \
                    + '_i' + str(int(ip['incl'])) \
                    + '_a' + str(int(sz)) \
                    + '_theta' + str(int(ip['theta'])) \
                    + '_mag' + str(mag) + '/MCMC/'
                try:
                    resfile = open(chains_folder + 'contrast_res_' \
                            + mod + '_' + fittype + '.pic', 'rb')
                    resdict = pickle.load(resfile)
                    resfile.close()
                    tbest = sorted(zip(resdict[mag][mod].items()), \
                                    key=lambda x: x[0][1])[0][0][0]
                    #tdiff_output[i, j] = -(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output_abs[i, j] = abs(td - tbest)/(ip['tstar'] + td)*100
                    #tdiff_output[i, j] = -(td - tbest)
                    #tdiff_output[ii, j] = resdict[mag]['Tunc'][0]
                    #if tdiff_output[ii, j] < 0:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][0]
                    #else:
                    #    tdiff_output[ii, j] /= resdict[mag]['Tunc'][1][1]
                    tdiff_output[ii, j] = resdict[mag]['Tunc'][1][1] \
                                - resdict[mag]['Tunc'][1][0]
                    #                - (ip['tstar'] + td)
                    diffT.append(abs(resdict[mag]['Tunc'][4]))
                    uncT.append(resdict[mag]['Tunc'][3])
                    if np.isnan(tdiff_output[i, j]) \
                                or tdiff_output[i, j] == np.inf:
                        tdiff_output[i, j] = 10000
                except FileNotFoundError:
                    tdiff_output[i, j] = -999
                if chi2rplot:
                    #chi2r = resdict[mag]['Tunc'][-2]
                    x, like = resdict[mag]['Tunc'][2], resdict[mag]['Tunc'][3]
                    #like = -2.*np.log(like)
                    spotSNR = resdict[mag]['Tunc'][-2]
                    print('Spot SNR = ', spotSNR)
                    ax1.plot(x, like, '.-', color=cc[j], \
                        markersize=sizemarker[j]*2, label=str(round(spotSNR)))
                    ax1.plot([td, td], [0., 1.], 'k')
                            #label=str(round(spotSNR)), color=cc[ii], \
                            #markersize=size*2)
            #tdiff_unc[i] = np.mean(uncT)
            #plt.figure(33)
            #plt.plot(tcontrast, diffT, 'o-', c=cc[i], label=str(mag))
            #ax.text(-1100, 7, r'True $T_\mathrm{spot} - T_\mathrm{star}$', \
            #                fontsize=[14])
            #axins.text(-1100, 1.6, '$99.7\%$ CI', fontsize=14)
            ax1.legend(frameon=False, loc='best', fontsize=14, \
                        title='Spot SNR', title_fontsize=14)
            #axins.plot([td, td], [0., 10.], 'k--')
            #ax.text(-300., 28., 'K mag', fontsize=14)
            #x1, x2, y1, y2 = -1200., -100., 0.9, 3.
            #axins.yaxis.tick_right()
            #ax.set_ylim(0, 5)
            #axins.set_ylim(y1, y2)
            ax1.set_xlabel(r'$T_\mathrm{spot} - T_\mathrm{star}$ [K]', fontsize=16)
            #ax1.set_ylabel('$\chi^2/\chi^2_\mathrm{min}$', fontsize=16)
            ax1.set_ylabel('$\mathcal{L}/\mathcal{L}_\mathrm{max}$', fontsize=16)
            #ax1.set_ylabel('$\mathcal{L}$', fontsize=16)
            ax1.set_title(str(int(ip['tstar'])) + ' K star, ' \
                    + str(int(ip['tstar'] + td)) + r' K spot, $\theta=$' \
                    + str(int(ip['theta'])) + ' $\deg$, ' \
                    + str(mag) + ' K$_\mathrm{mag}$',fontsize=16)
            plt.savefig(project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/likelihood_map_size_' \
                    + ip['instrument'] \
                    + '_' + mod + '_' + str(ip['tstar']) + '_' + str(td) \
                    #+ '_asizes' + str(int(ip['aumbra'])) \
                    + '_theta' + str(int(ip['theta'])) + '.pdf')
    #plt.show()
    #set_trace()
    plt.close('all')

    return

def plot_res3(ip, mags, tcontrast, models, fittype='LM'):
    '''
    Represent the results of the LM fits.
    '''

    plt.close('all')
    tcontrast = tcontrast[::3]
    tdiff_output = np.zeros((len(mags), len(tcontrast)))
    ffact_map = np.zeros((len(mags), len(tcontrast)))
    betaplanet = np.zeros((len(mags), len(tcontrast)))
    if ip['tstar'] == 5000:
        size = 5
    elif ip['tstar'] == 3500.:
        size = 3
    #fig, ax = plt.subplots()
    for i, mag in enumerate(mags):
        sizes = []
        for j, td in enumerate(tcontrast): #td
            #for i, size in enumerate(sizes):
            uncT = []
            diffT = [] # This is for the 2D plot
            tumbra = ip['tstar'] + td
            homedir = os.path.expanduser('~')
            project_folder = homedir + '/Projects/jwst_spots/revision2/'
            instrument = ip['instrument'] + '/'
            chains_folder = project_folder + instrument + 'star_' \
                + str(int(ip['tstar'])) + 'K/p' \
                + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                + '_spot' + str(int(ip['tstar'] + td)) \
                + '_i' + str(int(ip['incl'])) \
                + '_a' + str(int(size)) \
                + '_theta' + str(int(ip['theta'])) \
                + '_mag' + str(mag) + '/MCMC/'
            resfile = open(chains_folder \
                        + 'contrast_LMfit_josh_jwst.pic', 'rb')
            resdict = pickle.load(resfile)
            resfile.close()
            tdiff_output[i, j] = resdict[0].x[0] - tumbra
                                    #resdict[0].params['Tspot'] - tumbra
            #ffact_map[i, j] = resdict[0].params['delta']
            betaplanet[i, j] = resdict[0].x[1]#.params['beta']
            sizes.append(resdict[1])
        plt.plot(tcontrast, sizes, 'o-', label=str(mag))
    plt.xlabel(r'$T_\bullet - T_\star$', fontsize=16)
    plt.ylabel('Minimum spot size [deg]', fontsize=16)
    plt.legend(title='K mag', title_fontsize=14, frameon=False)
    plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
            + '_spotsize_a' + str(int(ip['aumbra'])) \
            + '_theta' + str(int(ip['theta'])) + '.pdf')

    plt.close('all')

    labels = [r'Output $\Delta T_\bullet$ [K]', r'$\beta$']#, r'$\delta$']
    outname = ['Tspot', 'beta']#, 'delta']
    for k, tab in enumerate([tdiff_output, betaplanet]):#, ffact_map]):
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.13, 0.77, 0.77])
        mm = ax.imshow(tab.T[::-1], extent=(min(mags), \
                    max(mags) + np.diff(mags)[0], \
                    ip['tstar'] + tcontrast.min(), ip['tstar']), \
                    # + ytdiff.max()),
                    aspect='auto', \
                    #vmin=0., vmax=1500,
                    cmap=plt.get_cmap('plasma'))
        ax.set_xticks(np.arange(min(mags), max(mags) + np.diff(mags)[0], \
                        np.diff(mags)[0]))
        ax.set_yticks(np.arange(ip['tstar'] + tcontrast.min(), \
                        ip['tstar'], np.diff(tcontrast)[0]))
        ax.grid(color='k')
        ll = plt.colorbar(mappable=mm, ax=ax)
        ll.set_label(labels[k], fontsize=16)
        ax.set_xlabel('$K$ mag', fontsize=16)
        ax.set_ylabel(r'Input $T_\bullet$ [K]', fontsize=16)
        ax.set_title(str(instrument).replace('_', ' ').replace('/', '') \
                + ', ' + str(int(ip['tstar'])) + ' K star,' \
                + r' $\theta =$' + str(int(ip['theta'])) + '$^\circ$', \
                    fontsize=16)
        plt.savefig(project_folder + instrument + 'star_' \
                + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
                + '_map_' \
                + outname[k] + '_a' + str(int(ip['aumbra'])) \
                + '_theta' + str(int(ip['theta'])) + '.pdf')

    plt.close('all')

    return

def plot_res4(ip, mags, tcontrast, models, fittype='LM', tight_ld_prior=True):

    plt.close('all')
    tcontrast = tcontrast[::3]
    #tdiff_output = np.zeros((len(mags), len(tcontrast)))
    #ffact_map = np.zeros((len(mags), len(tcontrast)))
    #betaplanet = np.zeros((len(mags), len(tcontrast)))
    if ip['tstar'] == 5000:
        size = 3
    elif ip['tstar'] == 3500.:
        size = 3

    if tight_ld_prior:
        ldpart = '_tightLD'
    else:
        ldpart = ''

    #fig, ax = plt.subplots()
    marker = ['o', '*', '^', 's', 'x']
    colour = ['g', 'r', 'c', 'y', 'brown']
    for i, mag in enumerate(mags):
        sizes = []
        for j, td in enumerate(tcontrast): #td
            #for i, size in enumerate(sizes):
            uncT = []
            diffT = [] # This is for the 2D plot
            tumbra = ip['tstar'] + td
            homedir = os.path.expanduser('~')
            project_folder = homedir + '/Projects/jwst_spots/revision2/'
            instrument = ip['instrument'] + '/'
            chains_folder = project_folder + instrument + 'star_' \
                + str(int(ip['tstar'])) + 'K/p' \
                + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                + '_spot' + str(int(ip['tstar'] + td)) \
                + '_i' + str(int(ip['incl'])) \
                + '_a' + str(int(size)) \
                + '_theta' + str(int(ip['theta'])) \
                + '_mag' + str(mag) + ldpart + '/MCMC/'
            #resfile = open(chains_folder + 'chains_spec.pic', 'rb')
            resfile = open(chains_folder + 'nested_spec.pic', 'rb')
            resdict = pickle.load(resfile)
            resfile.close()
            #tspot_chain = resdict['Chains'][:, 0]
            #percentiles = np.percentile(tspot_chain, [15.9, 50., 84.1])

            # Quantiles
            planet_angle = ip['rplanet']*0.1005/ip['rstar']
            samples = resdict.samples  # samples
            flag = samples[:, 1] <= 1.
            samples = samples[flag]
            weights = np.exp(resdict.logwt - resdict.logz[-1])[flag]  # normalized weights
            quantiles = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                        weights=weights) for samps in samples.T]
            #quantiles = [dyfunc.quantile(samps, [0.05, 0.50, 0.95], \
            #            weights=weights) for samps in samples.T]

            yerrup = np.diff(quantiles[0])[1]
            yerrdown = np.diff(quantiles[0])[0]
            toutput = quantiles[0][1]

            if instrument == 'NIRSpec_Prism/':
                errup = np.sqrt(yerrup**2)# + 79.**2)
                errdown = np.sqrt(yerrdown**2)# + 79.**2)
            elif instrument == 'NIRCam/':
                errup = np.sqrt(yerrup**2)# + 62.**2)
                errdown = np.sqrt(yerrdown**2)# + 62.**2)
            if j == 0:
                plt.scatter(-(ip['tstar'] - tumbra) + i*20, \
                    -(ip['tstar'] - toutput), marker=marker[i], s=100, \
                    c=colour[i], label='{}'.format(mag))
            else:
                plt.scatter(-(ip['tstar'] - tumbra) + i*20, \
                    -(ip['tstar'] - toutput), marker=marker[i], s=100, \
                    c=colour[i])
            plt.errorbar(-(ip['tstar'] - tumbra) + i*20, \
                    -(ip['tstar'] - toutput), yerr=([errdown], [errup]), \
                    fmt='-', color=colour[i], capsize=2)

    plt.xlabel(r'True $T_\bullet - T_\star$ [K]', fontsize=16)
    plt.ylabel(r'Recovered $T_\bullet - T_\star$ [K]', fontsize=16)
    plt.legend(title='K mag', title_fontsize=14, frameon=False)
    plt.title('{}'.format(int(ip['tstar'])) + ' K star, ' \
                + instrument.replace('/', '').replace('_', ' ') \
                + r', $\theta={}^\circ$'.format(int(ip['theta'])), fontsize=16)
    if ip['tstar'] == 5000:
        xx = -(ip['tstar'] - np.linspace(3800., 5000., 100))
        plt.xlim(-1300., -100)
    elif ip['tstar'] == 3500.:
        xx = -(ip['tstar'] - np.linspace(2300., 3500., 100))
        plt.xlim(-1000, -100)
    plt.plot(xx, xx, 'k--')
    plt.tight_layout()
    plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
            + '_' + str(int(ip['tstar'])) + '_a' + str(int(ip['aumbra'])) \
            + '_theta' + str(int(ip['theta'])) + '_Tscatter.pdf')
    plt.close('all')

    return

def plot_res5(ip, mags, tcontrast, models, fittype='LM', tight_ld_prior=True):

    plt.close('all')
    aumbras = np.arange(2, 6)
    if ip['tstar'] == 5000:
        tcontrast = [-900.]
    elif ip['tstar'] == 3500.:
        tcontrast = [-600.]

    if tight_ld_prior:
        ldpart = '_tightLD'
    else:
        ldpart = ''

    marker = ['o', '*', '^', 's', 'x']
    colour = ['g', 'r', 'c', 'y', 'brown']
    for i, size in enumerate(aumbras):
        sizes = []
        for k, mag in enumerate(mags):
            for j, td in enumerate(tcontrast):
                if int(ip['latspot']) == 21:
                    latpart = '_lat' + str(int(ip['latspot']))
                else:
                    latpart = ''
                tumbra = ip['tstar'] + td
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/Projects/jwst_spots/revision2/'
                instrument = ip['instrument'] + '/'
                chains_folder = project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) \
                    + '_i' + str(int(ip['incl'])) \
                    + '_a' + str(int(size)) + latpart \
                    + '_theta' + str(int(ip['theta'])) \
                    + '_mag' + str(mag) + ldpart + '/MCMC/'
                print(chains_folder)
                #resfile = open(chains_folder + 'chains_spec.pic', 'rb')
                resfile = open(chains_folder + 'nested_spec.pic', 'rb')
                resdict = pickle.load(resfile)
                resfile.close()

                # Quantiles
                samples = resdict.samples  # samples
                weights = np.exp(resdict.logwt - resdict.logz[-1])  # normalized weights
                quantiles = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                            weights=weights) for samps in samples.T]
                #quantiles = [dyfunc.quantile(samps, [0.05, 0.50, 0.95], \
                #            weights=weights) for samps in samples.T]

                yerrup = np.diff(quantiles[0])[1]
                yerrdown = np.diff(quantiles[0])[0]
                toutput = quantiles[0][1]

                if instrument == 'NIRSpec_Prism/':
                    errup = np.sqrt(yerrup**2)# + 79.**2)
                    errdown = np.sqrt(yerrdown**2)# + 79.**2)
                elif instrument == 'NIRCam/':
                    errup = np.sqrt(yerrup**2)# + 62.**2)
                    errdown = np.sqrt(yerrdown**2)# + 62.**2)

                if i == 0:
                    plt.scatter(size + k*0.1, \
                        -(ip['tstar'] - toutput), marker=marker[k], \
                        s=100, c=colour[k], label='{}'.format(mag))
                else:
                    plt.scatter(size + k*0.1, \
                        -(ip['tstar'] - toutput), marker=marker[k], \
                        s=100, c=colour[k])
                plt.errorbar(size + k*0.1, \
                  -(ip['tstar'] - toutput), \
                  yerr=([errdown], [errup]), fmt='-', color=colour[k], capsize=2)

    plt.plot(np.linspace(1.5, 5.5, len(aumbras)), \
                np.zeros(len(aumbras)) + tcontrast[0], 'k--')
    if ip['tstar'] == 3500.:
        yloc = -1100
    elif ip['tstar'] == 5000.:
        yloc = -1000
    plt.text(4., yloc, '--- True value', fontsize=16)
    plt.xlabel('Starspot size [deg]', fontsize=16)
    plt.ylabel(r'$T_\bullet - T_\star$ [K]', fontsize=16)
    plt.legend(title='K mag', title_fontsize=14, frameon=False)
    plt.title('{}'.format(int(ip['tstar'])) + ' K star, ' + instrument.replace('/', \
                '').replace('_', ' ') \
                + r', $\theta={}^\circ$'.format(int(ip['theta'])), fontsize=16)
    plt.tight_layout()
    plt.savefig(project_folder + instrument + 'star_' \
        + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
        + '_' + str(int(ip['tstar'])) + '_avary_theta' + str(int(ip['theta'])) \
        + '_Tscatter.pdf')
    plt.close('all')

    return

def plot_res6(ip, mags, tcontrast, models, fittype='LM', tight_layout=True, \
            tight_ld_prior=True):

    instrument = ip['instrument'] + '/'

    if 'NIRCam' in instrument:
        mags = np.arange(4.5, 10.5, 1.5)
    elif 'NIRSpec_Prism' in instrument:
        mags = np.arange(10.5, 15.0, 1.0)
    plt.close('all')
    aumbras = [3.]
    if ip['tstar'] == 5000:
        tcontrast = np.arange(-1200., 0., 300)
    elif ip['tstar'] == 3500.:
        tcontrast = np.arange(-900., 0., 300.)

    if tight_ld_prior:
        ldpart = '_tightLD'
    else:
        ldpart = ''

    lat = '21'
    marker = ['o', '*', '^', 's', 'x']
    colour = ['g', 'r', 'c', 'y', 'brown']
    for i, size in enumerate(aumbras):
        sizes = []
        for k, mag in enumerate(mags):
            for j, td in enumerate(tcontrast):
                if int(ip['latspot']) == 21:
                    latpart = '_lat' + str(int(ip['latspot']))
                else:
                    latpart = ''
                tumbra = ip['tstar'] + td
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/Projects/jwst_spots/revision2/'
                chains_folder = project_folder + instrument + 'star_' \
                    + str(int(ip['tstar'])) + 'K/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) \
                    + '_i' + str(int(ip['incl'])) \
                    + '_a' + str(int(size)) + latpart \
                    + '_theta' + str(int(ip['theta'])) \
                    + '_mag' + str(mag) + ldpart + '/MCMC/'
                print(chains_folder)
                resfile = open(chains_folder + 'nested_spec.pic', 'rb')
                resdict = pickle.load(resfile)
                resfile.close()

                # Quantiles
                samples = resdict.samples  # samples
                weights = np.exp(resdict.logwt - resdict.logz[-1])  # normalized weights
                #quantiles = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                #            weights=weights) for samps in samples.T]
                quantiles = [dyfunc.quantile(samps, [0.003, 0.50, 0.997], \
                            weights=weights) for samps in samples.T]

                yerrup = np.diff(quantiles[0])[1]
                yerrdown = np.diff(quantiles[0])[0]
                toutput = quantiles[0][1]

                if instrument == 'NIRSpec_Prism/':
                    errup = np.sqrt(yerrup**2)# + 79.**2)
                    errdown = np.sqrt(yerrdown**2)# + 79.**2)
                elif instrument == 'NIRCam/':
                    errup = np.sqrt(yerrup**2)# + 62.**2)
                    errdown = np.sqrt(yerrdown**2)# + 62.**2)

                if j == 0:
                    plt.scatter(-(ip['tstar'] - tumbra) + k*20, \
                        -(ip['tstar'] - toutput), marker=marker[k], s=100, \
                        c=colour[k], label='{}'.format(mag))
                else:
                    plt.scatter(-(ip['tstar'] - tumbra) + k*20, \
                        -(ip['tstar'] - toutput), marker=marker[k], s=100, \
                        c=colour[k])
                plt.errorbar(-(ip['tstar'] - tumbra) + k*20, \
                        -(ip['tstar'] - toutput), yerr=([errdown], [errup]), \
                        fmt='-', color=colour[k], capsize=2)

    if ip['tstar'] == 5000:
        xx = -(ip['tstar'] - np.linspace(3800., 5000., 100))
    elif ip['tstar'] == 3500.:
        xx = -(ip['tstar'] - np.linspace(2300., 3500., 100))
    plt.plot(xx, xx, 'k--')
    plt.xlabel(r'True $T_\bullet - T_\star$ [K]', fontsize=16)
    plt.legend(title='K mag', title_fontsize=14, frameon=False)
    plt.ylabel(r'Recovered $T_\bullet - T_\star$ [K]', fontsize=14)
    plt.title('{}'.format(int(ip['tstar'])) + ' K star, ' \
                + instrument.replace('/', '').replace('_', ' ') \
                + r', $\theta={}^\circ$'.format(int(ip['theta'])) \
                + r', lat $=' + lat + '^\circ$', fontsize=16)
    plt.tight_layout()
    if ip['tstar'] == 5000:
        plt.xlim(-1300., -100)
    elif ip['tstar'] == 3500.:
        plt.xlim(-1000, -100)
    plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
            + '_' + str(int(ip['tstar'])) + '_a' + str(int(ip['aumbra'])) \
            + '_lat' + lat + '_theta' + str(int(ip['theta'])) + '_Tscatter.pdf')
    plt.close('all')

    return

def plot_res7(ip, mags, tcontrast, models, fittype='LM', tight_ld_prior=True, \
        noscatter=False, scalehisto=100):

    #aumbras = np.arange(2, 6)
    plothistos = [True, False]
    for plothisto in plothistos:
        aumbras = [3.]
        if ip['tstar'] == 5000:
            #tcontrast = [-600.]
            tcontrast = np.arange(-1200, -100, 300)
        elif ip['tstar'] == 3500.:
            tcontrast = np.arange(-900, -100, 300)
            #tcontrast = [-300.]
        if tight_ld_prior:
            ldpart = '_tightLD'
        else:
            ldpart = ''
        if noscatter:
            scatterpart = '_noscatter'
        else:
            scatterpart = ''

        marker = ['o', '*', '^', 's', 'x']
        colour = ['g', 'r', 'b', 'orange']#, 'brown']
        SNRmin = np.inf
        SNRmax = -np.inf
        tcmax = -np.inf
        tcmin = np.inf
        snr, tdiff, tu, yerru, yerrd, mmags, flatness, betamin \
                    = [], [], [], [], [], [], [], []
        for i, size in enumerate(aumbras):
            sizes = []
            for k, mag in enumerate(mags):
                for m, theta in enumerate([ip['theta']]):
                    for j, td in enumerate(tcontrast):
                        if ip['latspot'] == 21:
                            latpart = '_lat' + str(ip['latspot'])
                            theta = 21
                            if ip['tstar'] == 5000:
                                ip['incl'] = 88.1
                            else:
                                ip['incl'] = 88.75
                        else:
                            latpart = ''
                        tumbra = ip['tstar'] + td
                        homedir = os.path.expanduser('~')
                        project_folder = homedir + '/Projects/jwst_spots/revision2/'
                        instrument = ip['instrument'] + '/'
                        chains_folder = project_folder + instrument + 'star_' \
                            + str(int(ip['tstar'])) + 'K/p' \
                            + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                            + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                            + '_spot' + str(int(ip['tstar'] + td)) \
                            + '_i' + str(int(ip['incl'])) \
                            + '_a' + str(int(size)) + latpart \
                            + '_theta' + str(int(theta)) \
                            + '_mag' + str(mag) + scatterpart + ldpart + '/MCMC/'
                        if not os.path.exists(chains_folder):
                            print(chains_folder, 'does not exist')
                            continue
                        transits = glob.glob(chains_folder + 'transit*pic')
                        tmax = len(transits) - 2
                        #print(chains_folder, tumbra)
                        A, Aunc = [], []
                        for index in [1, tmax - 1, -1]:
                            ffopen = open(chains_folder \
                                + 'transit_' + str(index) + '_nested.pic', 'rb')
                            sresults = pickle.load(ffopen)
                            ffopen.close()
                            samples = sresults['samples']
                            weights = np.exp(sresults.logwt - sresults.logz[-1])
                            perc = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                                        weights=weights) for samps in samples.T]
                            #A = perc[6][1]
                            samples_equal = dyfunc.resample_equal(samples, weights)
                            # Evaluate fitted n, to decide whether the
                            # solution is reliable
                            if index == -1:
                                flatness.append(perc[7][2] - perc[7][1])
                                #print('n = ', perc[7][2] - perc[7][1])
                                #flatness.append(scipy.stats.kurtosis(samples_equal[:, 7]))
                                #print(scipy.stats.kurtosis(samples_equal[:, 7]))
                            A.append(perc[6][1])
                            Aunc.append(np.mean(np.diff(perc[6])))
                        data_folder = project_folder + instrument + 'star_' \
                            + str(int(ip['tstar'])) + 'K/p' \
                            + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                            + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                            + '_spot' + str(int(ip['tstar'] + td)) \
                            + '_i' + str(int(ip['incl'])) \
                            + '_a' + str(int(size)) + latpart \
                            + '_theta' + str(int(theta)) \
                            + '_mag' + str(mag) + scatterpart + ldpart \
                                        + '/simulated_data/'
                        #lcfile = open(data_folder + 'transit_spots_-1.pic', 'rb')
                        #lc = pickle.load(lcfile)
                        #lcfile.close()
                        #t, y, yerr, wl = lc
                        #SNR = A*3.**0.5/np.mean(yerr)
                        SNR = np.sqrt((A[0] - A[1])**2 + A[2]**2)/Aunc[1]
                        print(chains_folder.split('star_')[1], SNR)
                        #resfile = open(chains_folder + 'chains_spec.pic', 'rb')
                        resfile = open(chains_folder + 'nested_spec.pic', 'rb')
                        resdict = pickle.load(resfile)
                        resfile.close()

                        # Quantiles
                        samples = resdict.samples  # samples
                        weights = np.exp(resdict.logwt - resdict.logz[-1])  #        normalized     weights
                        #quantiles = [dyfunc.quantile(samps, [0.159, 0.50, 0.841], \
                        #            weights=weights) for samps in samples.T]
                        quantiles = [dyfunc.quantile(samps, [0.025, 0.50, 0.975], \
                                    weights=weights) for samps in samples.T]

                        #quantiles = [dyfunc.quantile(samps, [0.003, 0.50, 0.997], \
                        #            weights=weights) for samps in samples.T]
                        yerrup = np.diff(quantiles[0])[1]
                        yerrdown = np.diff(quantiles[0])[0]
                        toutput = quantiles[0][1]

                        if instrument == 'NIRSpec_Prism/':
                            errup = np.sqrt(yerrup**2)# + 79.**2)
                            errdown = np.sqrt(yerrdown**2)# + 79.**2)
                        elif instrument == 'NIRCam/':
                            errup = np.sqrt(yerrup**2)# + 62.**2)
                            errdown = np.sqrt(yerrdown**2)# + 62.**2)
                        if SNR < SNRmin:
                            SNRmin = SNR
                        if SNR > SNRmax:
                            SNRmax = SNR
                        if toutput - tumbra < tcmin:
                            tcmin = toutput - tumbra
                        if toutput - tumbra > tcmax:
                            tcmax = toutput - tumbra

                        # Resample weighted samples.
                        samples_equal = dyfunc.resample_equal(samples, \
                                weights)
                        # Get beta_min
                        betamin.append(samples_equal[:, 1].min())
                        #betamin.append( \
                        #            scipy.stats.kurtosis(samples_equal[:, 1]))
                        if plothisto:
                            n, bins, _ = \
                             plt.hist(samples_equal[:, 0] - (ip['tstar'] + td), \
                             orientation='horizontal', bottom=SNR*scalehisto, \
                             alpha=0.7, density=False, bins=20, \
                             color=colour[j], histtype='step')
                            plt.plot([SNR*scalehisto, SNR*scalehisto], \
                                    [bins.min() - 20, \
                                    bins.max() + 20], c=colour[j])
                        snr.append(SNR)
                        tdiff.append(toutput - tumbra)
                        tu.append(tumbra)
                        yerru.append(yerrup)
                        yerrd.append(yerrdown)
                        mmags.append(mag)

        snr = np.array(snr)
        tdiff = np.array(tdiff)
        tu = np.array(tu)
        yerru = np.array(yerru)
        yerrd = np.array(yerrd)
        mmags = np.array(mmags)
        flatness = np.array(flatness)
        betamin = np.array(betamin)
        for j, mm in enumerate(mags):
            for i, tc in enumerate(tcontrast):
                #rand = np.random.uniform(low=-50, high=50)
                flag = np.logical_and(tu - ip['tstar'] == tc, mmags == mm)
                if not plothisto:
                    #flatflag = betamin[flag] < 0.5
                    flatflag = flatness[flag] < 2
                    plt.errorbar(snr[flag][flatflag], tdiff[flag][flatflag], \
                            yerr=([yerrd[flag][flatflag], \
                            yerru[flag][flatflag]]), fmt='.', color=colour[i], \
                            capsize=2)
                    plt.errorbar(snr[flag][~flatflag], \
                            tdiff[flag][~flatflag], \
                            yerr=([yerrd[flag][~flatflag], \
                            yerru[flag][~flatflag]]), capsize=2, \
                            color=colour[i])
                    plt.errorbar(snr[flag][~flatflag], \
                            tdiff[flag][~flatflag], \
                            yerr=([yerrd[flag][~flatflag], \
                            yerru[flag][~flatflag]]), \
                            color=colour[i], uplims=True)

                    #eb[-1][0].set_linestyle('-.')
                    plt.scatter(snr[flag], tdiff[flag], marker=marker[j], \
                       s=20*size, c=colour[i])
                if i == 2:
                    plt.scatter([], [], marker=marker[j], \
                       s=100., c='k', \
                       #colour[i], \
                       label='{}'.format(mm))

        for i, tc in enumerate(tcontrast):
            # Stellar Teff
            if not plothisto:
                plt.plot([max([SNRmin - 10, 0]), SNRmax + 5], \
                    [-tc - 100, -tc - 100], alpha=0.5, color=colour[i])
            else:
                plt.plot([0., SNRmax*scalehisto + 50*scalehisto], \
                    [-tc - 100, -tc - 100], alpha=0.5, color=colour[i])
            text_kwargs = dict(color=colour[i])
            if not plothisto:
                plt.text(snr.max()*0.8, -500 - 100*i, r'$T_\bullet=' \
              + str(int(ip['tstar'] + tc)) + '$ K', fontsize=12, **text_kwargs)
            else:
                if int(ip['tstar']) == 5000:
                    tstart = -500
                elif int(ip['tstar']) == 3500:
                    tstart = -700
                plt.text(snr.max()*(scalehisto - scalehisto/5), tstart - 100*i, \
                r'$T_\bullet=' \
            + str(int(ip['tstar'] + tc)) + '$ K', fontsize=12, **text_kwargs)

        if not plothisto:
            plt.plot([0, SNRmax + 5], [0., 0.], 'k--')
            plt.xlim(max([SNRmin - 2, 0]), SNRmax + 1)
        else:
            plt.plot([0., SNRmax*scalehisto + 100*scalehisto], \
                        [0., 0.], 'k--')
            #plt.xlim(min(50., scalehisto), SNRmax*scalehisto + 100)
            plt.xlim(200., SNRmax*scalehisto + 100)
        plt.ylim(-1000, 1000)
        if not plothisto:
            plt.xlabel('Occultation SNR', fontsize=16)
            plt.ylabel(r'$T_\bullet$ (meas) $- T_\bullet$ (real) [K] $(2\sigma)$', \
                        fontsize=16)
        else:
            plt.xlabel(r'Occultation SNR$\times$' + str(scalehisto), \
                        fontsize=16)
            plt.ylabel(r'$T_\bullet$ (meas) $- T_\bullet$ (real) [K]', \
                    fontsize=16)
        if abs(tcmin) > abs(tcmax):
            ll = 'lower right'
        else:
            ll = 'upper left'
        if not plothisto:
            plt.legend(title='K mag', title_fontsize=12, frameon=False, \
                    loc='lower left')
        plt.title('{}'.format(int(ip['tstar'])) + ' K star, ' \
                    + instrument.replace('/', '').replace('_', ' ') \
                    + r', $\theta={}^\circ$'.format(int(ip['theta'])), fontsize=16)
        plt.tight_layout()
        if not plothisto:
            plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
            + '_' + str(int(ip['tstar'])) + 'SNRvar_theta' + str(int(ip['theta'])) \
            + '_Tscatter.pdf')
        else:
            plt.savefig(project_folder + instrument + 'star_' \
            + str(int(ip['tstar'])) + 'K/' + instrument.replace('/', '') \
            + '_' + str(int(ip['tstar'])) + 'SNRvar_theta' + str(int(ip['theta'])) \
            + '_Tscatter_histo.pdf')
        plt.close('all')

    return


def main2(spotsize, instruments, thetas, stars, latspot=0., noscatter=False, \
        tight_ld_prior=True, onlyres=False, oldf2f1=False, typeres=7, \
        realization=0):
    '''
    Parameters
    ----------

    Instruments: list for NIRCam and/or NIRSpec_Prism
    thetas: list for limb-angle values
    stars: list for K and M star
    '''

    incl = 90.

    for m, instrum in enumerate(instruments):
        for j, incl in enumerate([incl]):
            for k, theta in enumerate(thetas): # mu angle 0.., 40
                for m, star in enumerate(stars):
                    if star == 'K':
                        rp = 0.75
                        rs = 0.75
                        ts = 5000.
                        logg = 4.5
                        if latspot == 21:
                            incl = 88.1#87.6
                    elif star == 'M':
                        rp = 0.25
                        rs = 0.47
                        ts = 3500.
                        logg = 5.0
                        if latspot == 21:
                            incl = 88.75#88.7
                    if latspot == 21:
                        theta = 21
                    inputpars = {}
                    inputpars['incl'] = incl
                    inputpars['theta'] = theta
                    inputpars['aumbra'] = spotsize
                    inputpars['latspot'] = latspot
                    cycle(rp, rs, ts, logg, instrum, \
                        simulate_transits=False, fit_transits=False, \
                        fit_spectra=True, spotted_starmodel=False, \
                        inputpars=inputpars, update=False, chi2rplot=True, \
                        model='batman', noscatter=noscatter, \
                        tight_ld_prior=tight_ld_prior, onlyres=onlyres, \
                        oldf2f1=False, type_res=typeres, resume=False, \
                        realization=realization)

    return

def main(tstar, instrument, theta, spotsize, latspot, outfile):

    sys.stdout \
        = open('/home/giovanni/Projects/jwst_spots/revision2/logs/' \
            + outfile, 'wt')
    tstar = int(tstar)
    theta = int(theta)
    latspot = int(latspot)
    inputpars = {}
    if latspot == 0:
        inputpars['incl'] = 90.
    elif latspot == 21 and tstar == 3500:
        inputpars['incl'] = 88.75#88.7
    elif latspot == 21 and tstar == 5000:
        inputpars['incl'] = 88.1#87.6
    inputpars['theta'] = theta
    inputpars['aumbra'] = float(spotsize)
    inputpars['latspot'] = int(latspot)
    if tstar == 5000:
        rstar = 0.75
        rplanet = 0.75
        loggstar = 4.5
    elif tstar == 3500:
        rstar = 0.47
        rplanet = 0.25
        loggstar = 5.0
    cycle(rplanet, rstar, tstar, loggstar, instrument, \
            simulate_transits=True, fit_transits=True, \
            fit_spectra=True, spotted_starmodel=False, \
            inputpars=inputpars, update=False, chi2rplot=True, model='batman', \
            tight_ld_prior=True, noscatter=True, resume=False)

    return

if __name__ == '__main__':
    tstar_ = sys.argv[1]
    instrument_ = sys.argv[2]
    theta_ = sys.argv[3]
    spotsize_ = sys.argv[4]
    latspot_ = sys.argv[5]
    outfile = sys.argv[6]
    main(tstar_, instrument_, theta_, spotsize_, latspot_, outfile)
