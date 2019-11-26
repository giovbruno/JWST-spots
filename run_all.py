import numpy as np
import simulate_transit
import transit_fit
import spectra_fit
import os
import pathlib
import pickle
import matplotlib.pyplot as plt
from pdb import set_trace

def go(magstar, rplanet, tstar, tumbra, tpenumbra, loggstar, rstar, instr):

    # Folders & files
    pardict = {}
    pardict['homedir'] = os.path.expanduser('~')
    pardict['project_folder'] = pardict['homedir'] + '/projects/jwst_spots/'
    pardict['instrument'] = str(instr) + '/'
    if not os.path.exists(pardict['project_folder'] + pardict['instrument']):
        os.mkdir(pardict['project_folder'] + pardict['instrument'])
    pardict['case_folder'] = pardict['project_folder'] \
            + pardict['instrument'] + 'star_' + str(int(tstar)) + 'K/p' \
            + str(np.round(rplanet, 4)) \
            + '_star' + str(rstar) + '_' + str(int(tstar)) + '_' \
            + str(loggstar) + '_spot' + str(int(tumbra)) + '_' \
            + str(int(tpenumbra)) + '_mag' + str(magstar) + '/'
    if not os.path.exists(pardict['case_folder']):
        os.mkdir(pardict['case_folder'])
    pardict['data_folder'] = pardict['case_folder'] + 'simulated_data/'
    if not os.path.exists(pardict['data_folder']):
        os.mkdir(pardict['data_folder'])
    pardict['chains_folder']  = pardict['case_folder'] + 'MCMC/'
    if not os.path.exists(pardict['chains_folder']):
        os.mkdir(pardict['chains_folder'])
    pardict['pandexo_out'] = pardict['data_folder'] + 'singlerun_3transits.p'
    #pardict['pandexo_out_hst'] = pardict['data_folder'] + 'singlerun_hst.p'
    pardict['logfile_data'] = pardict['data_folder'] + 'logfile.log'
    pardict['logfile_chains'] = pardict['chains_folder'] + 'logfile.log'

    pardict['flag'] = [1.90, 1.97]        # Time intervals for normalization

    pardict['magstar'] = magstar
    pardict['rstar'] = rstar
    pardict['tstar'] = tstar
    pardict['tumbra'] = tumbra
    pardict['tpenumbra'] = tpenumbra
    pardict['loggstar'] = loggstar
    pardict['rplanet'] = rplanet
    pardict['planettype'] = 'constant'
    pardict['ldfile'] = pardict['project_folder'] + 'ldfiles/' \
        + str(int(pardict['tstar'])) + '_' \
        + str(np.round(pardict['loggstar'], 1)) \
        + '_0.0_tophat_14ch_quadratic_kurucz.txt'

    # Check
    if pathlib.Path(pardict['logfile_data']).exists() or \
    pathlib.Path(pardict['logfile_chains']).exists():
        print('Log file already exists. Continue?')
        set_trace()
        os.system('rm ' + pardict['logfile_data'])
        os.system('rm ' + pardict['logfile_chains'])

    # Run process for JWST
    simulate_transit.generate_spectrum_jwst(pardict)
    totchan = simulate_transit.add_spots(pardict, 'jwst')
    # Channels to perform fit
    expchan = np.arange(totchan)
    # Select channels and fit transit + spots
    #transit_fit.transit_spectro(pardict, expchan, 'jwst')
    # Fit derived transit depth rise with stellar models
    #spectra_fit.read_res(pardict, expchan, 'jwst', pardict['chains_folder'] \
    #    + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res_')

    # Now, for HST - requires ramp calculation but it's missing in the tutorials
    #simulate_transit.generate_spectrum_hst(pardict)
    #simulate_transit.add_spots(pardict, 'hst')
    #expchan = np.arange(1, 14)
    #transit_fit.transit_spectro(pardict, expchan, 'hst')
    #spectra_fit.read_res(pardict, expchan, 'hst', pardict['chains_folder'] \
    #        + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res.pic')

    return pardict

def cycle(rplanet, rstar, tstar, loggstar, tcontrast):
    '''
    Run the simulations for several scenarios.
    '''

    # Stellar/planet pars
    ip = {}
    ip['rplanet'] = rplanet #1.0 for 5000 K star
    ip['rstar'] = rstar #1.0
    #ip['rplanet'] = 0.3
    #ip['rstar'] = 0.3
    ip['tstar'] = tstar #5000
    ip['loggstar'] = loggstar #4.5
    ip['tpenumbra'] = 5500
    ip['instrument'] = 'NIRSpec_Prism'
    mags = np.linspace(10.5, 14.5, 5)
    #tcontrast = np.linspace(-1750, -250, 5) #for 5000 K, v1
    #tcontrast = np.arange(-1500, 0, 200) for 5000 K, v2
    #tcontrast = np.arange(-1000, 0, 200) #for 3500 K

    for mag in mags:
        for td in tcontrast:
            pardict = go(mag, ip['rplanet'], ip['tstar'], ip['tstar'] + td , \
                ip['tpenumbra'], ip['loggstar'], ip['rstar'], ip['instrument'])

    #plot_res(ip, mags, tcontrast)

    return

def plot_res(inputpars, mags, tcontrast):
    '''
    Build a 2d plot containing stellar mag, t diff input as axes and
    tdiff_output as color.
    '''

    # Arrays for resolution. Here we are just renaming
    ip = inputpars
    xmag = mags
    ytdiff = tcontrast
    tdiff_output = np.zeros((len(xmag), len(ytdiff)))
    #tdiff_output_abs = np.copy(tdiff_output)
    for mod in ['phoenix', 'k93models']:
        for i, mag in enumerate(mags):
            for j, td in enumerate(tcontrast):
                tumbra = ip['tstar'] + tcontrast
                homedir = os.path.expanduser('~')
                project_folder = homedir + '/projects/jwst_spots/'
                instrument = ip['instrument'] + '/'
                chains_folder = project_folder + instrument + 'tests_v2/p' \
                    + str(ip['rplanet']) + '_star' + str(ip['rstar']) + '_' \
                    + str(int(ip['tstar'])) + '_' + str(ip['loggstar']) \
                    + '_spot' + str(int(ip['tstar'] + td)) + '_' \
                    + str(int(ip['tpenumbra'])) + '_mag' + str(mag) + '/MCMC/'
                resfile = open(chains_folder + 'contrast_res_' \
                                                        + mod + '.pic', 'rb')
                resdict = pickle.load(resfile)
                resfile.close()
                tbest = sorted(zip(resdict[mag][mod].items()), \
                                    key=lambda x: x[0][1])[0][0][0]
                tdiff_output[i, j] = (td - tbest)/(ip['tstar'] + td)*100
                #tdiff_output_abs[i, j] = abs(td - tbest)/(ip['tstar'] + td)*100
        plt.figure()
        plt.imshow(tdiff_output.T, extent=(xmag.min(), xmag.max(), \
                    ip['tstar'] + ytdiff.min(), ip['tstar'] + ytdiff.max()), \
                    aspect='auto', interpolation='hanning', vmin=-15, vmax=15)
        ll = plt.colorbar()
        ll.set_label(r'$\Delta T_\mathrm{spot} [\%]$', fontsize=16)
        plt.xlabel('H mag', fontsize=16)
        plt.ylabel('$T_\mathrm{spot}$ [K]', fontsize=16)
        plt.title('5000 K star,' + mod, fontsize=16)
        plt.show()
        plt.savefig(project_folder + instrument + 'tests_v2/accuracy_' \
                                                                + mod + '.pdf')

    return
