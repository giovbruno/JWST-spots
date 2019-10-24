import numpy as np
import simulate_transit
import transit_fit
import spectra_fit
import os
import pathlib
from pdb import set_trace

def go(magstar, rstar, tstar, tumbra, tpenumbra, loggstar, rplanet, instr):

    # Folders & files
    pardict = {}
    pardict['homedir'] = os.path.expanduser('~')
    pardict['project_folder'] = pardict['homedir'] + '/projects/jwst_spots/'
    pardict['instrument'] = str(instr) + '/'
    if not os.path.exists(pardict['project_folder'] + pardict['instrument']):
        os.mkdir(pardict['project_folder'] + pardict['instrument'])
    pardict['case_folder'] = pardict['project_folder'] \
            + pardict['instrument'] + 'p' + str(np.round(rplanet, 4)) \
            + '_star' + str(rstar) + '_' + str(tstar) + '_' \
            + str(loggstar) + '_spot' + str(tumbra) + '_' + str(tpenumbra) \
            + '_mag' + str(magstar) + '/'
    if not os.path.exists(pardict['case_folder']):
        os.mkdir(pardict['case_folder'])
    pardict['data_folder'] = pardict['case_folder'] + 'simulated_data/'
    if not os.path.exists(pardict['data_folder']):
        os.mkdir(pardict['data_folder'])
    #pardict['attempt_mcmc'] = pardict['attempt_data'] + '/'
    pardict['chains_folder']  = pardict['case_folder'] + 'MCMC/'
    if not os.path.exists(pardict['chains_folder']):
        os.mkdir(pardict['chains_folder'])
    pardict['pandexo_out'] = pardict['data_folder'] + 'singlerun_3transits.p'
    #pardict['pandexo_out_hst'] = pardict['data_folder'] + 'singlerun_hst.p'
    pardict['logfile_data'] = pardict['data_folder'] + 'logfile.log'
    pardict['logfile_chains'] = pardict['chains_folder'] + 'logfile.log'
    pardict['ldfile'] = pardict['project_folder'] \
                + 'ldfiles/w52_NIRSpecG235_quadratic.txt'
    pardict['flag'] = [1.90, 1.97]        # Time intervals for normalization

    pardict['magstar'] = magstar
    pardict['rstar'] = rstar
    pardict['tstar'] = tstar
    pardict['tumbra'] = tumbra
    pardict['tpenumbra'] = tpenumbra
    pardict['loggstar'] = loggstar
    pardict['rplanet'] = rplanet
    pardict['planettype'] = 'constant'

    expchan = np.arange(10, 70, 10)
    '''
    # Check
    if pathlib.Path(pardict['logfile_data']).exists() or \
    pathlib.Path(pardict['logfile_chains']).exists():
        print('Log file already exists. Continue?')
        set_trace()
        os.system('rm ' + pardict['logfile_data'])
        os.system('rm ' + pardict['logfile_chains'])

    # Run process for JWST
    simulate_transit.generate_spectrum_jwst(pardict)
    simulate_transit.add_spots(pardict, 'jwst')

    # Select channels and fit transit + spots
    transit_fit.transit_spectro(pardict, expchan, 'jwst')
    '''
    # Fit derived transit depth rise with stellar models
    spectra_fit.read_res(pardict, expchan, 'jwst', pardict['chains_folder'] \
        + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res.pic')

    # Now, for HST - requires ramp calculation but it's missing in the tutorials
    #simulate_transit.generate_spectrum_hst(pardict)
    #simulate_transit.add_spots(pardict, 'hst')
    #expchan = np.arange(1, 14)
    #transit_fit.transit_spectro(pardict, expchan, 'hst')
    #spectra_fit.read_res(pardict, expchan, 'hst', pardict['chains_folder'] \
            + 'contrast_plot_', pardict['chains_folder'] + 'contrast_res.pic')

    return pardict

    # Then, spot models
        #spotmodels_IC.transit_spectro(pardict)
        #for i in np.arange(1, 74, 10):
    #for multfact in [1.]:#, 2.5, 5., 7.5]:
                #continue
        #        print(i)
    #    spotmodel_twotransits.transit_spectro(pardict, multfact, \
    #                   channels=np.arange(3, 38, 5))

    #else:
    #    print('Mode?')
    #    return
    #spectro_mcmc.run_mcmcs(pardict)
    #spectro_mcmc.plot_spectra(pardict)

    # Results
    #folder_root = '/Users/gbruno/projects/jwst_spots/NIRSpec_G235H/MCMC/'
    #comparisons.comp_entropy([folder_root + 'case1_masked/', folder_root \
    #                   + 'case1_spots/'], ['masked', 'spots'])
    #comparisons.plot_contrasts(folder_root + 'case1_spots/')
    #comparisons.plot_spectra([folder_root + 'case1_masked/', folder_root \
    #           + 'case1_spots/'], ['masked', 'spots'])

def cycle():
    '''
    Run the simulations for several scenarios.
    '''
    #mags = np.concatenate((np.arange(9., 13., 0.5), np.arange(15., 16., 0.5)))
    #for mag in mags:
    #    pardict = go(mag, 1., 5700, 5000, 5500, 4.5, 1., 'NIRSpec_Prism')
    #tumbras = [-1300, -1000, -600, -300]
    #for td in tumbras:
    #    pardict = go(10, 1., 5000, 5000 + td , 5500, 4.5, 1., 'NIRSpec_Prism')
    #for mag in mags:
    #    pardict = go(mag, 1., 4000, 3500, 5500, 4.5, 1., 'NIRSpec_Prism')
    radii = np.linspace(0.01, 0.11, 5)
    for i, rprst in enumerate(radii):
        pardict = go(10., 1., 5000, 4500, 5500, 4.5, rprst*9.95, 'NIRSpec_Prism')

    spectra_fit.plot_precision(pardict, radii*9.95, 'rp_rstar')

    return
