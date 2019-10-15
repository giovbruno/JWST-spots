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
                + pardict['instrument'] + 'p' + str(rplanet) + '_star' \
                + str(rstar) + '_' + str(tstar) + '_' + str(loggstar) \
                + '_spot' + str(tumbra) + '_' + str(tpenumbra) + '_mag' \
                + str(magstar) + '/'
    if not os.path.exists(pardict['case_folder']):
        os.mkdir(pardict['case_folder'])
    pardict['data_folder'] = pardict['case_folder'] + 'simulated_data/'
    if not os.path.exists(pardict['data_folder']):
        os.mkdir(pardict['data_folder'])
    #pardict['attempt_mcmc'] = pardict['attempt_data'] + '/'
    pardict['chains_folder']  = pardict['case_folder'] + 'MCMC/'
    if not os.path.exists(pardict['chains_folder']):
        os.mkdir(pardict['chains_folder'])
    pardict['pandexo_out'] = pardict['data_folder'] + 'singlerun.p'
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

    # Check
    if pathlib.Path(pardict['logfile_data']).exists() or \
    pathlib.Path(pardict['logfile_chains']).exists():
        print('Log file already exists. Continue?')
        set_trace()
        os.system('rm ' + pardict['logfile_data'])
        os.system('rm ' + pardict['logfile_chains'])

    #simulate_transit.generate_spectrum(pardict)
    #simulate_transit.add_spots(pardict, 20)
    #neptune_mdwarf.channelmerge(pardict, 3, 33)

    # Select channels and fit transit + spots
    expchan = np.arange(40, 70, 10)
    transit_fit.transit_spectro(pardict, expchan)

    # Fit derived transit depth rise with stellar models
    spectra_fit.read_res(pardict, expchan, pardict['chains_folder'] \
                       + 'contrast_plot_kurucz.pdf')

    return

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
