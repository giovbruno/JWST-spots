import numpy as np
import neptune_mdwarf
import spotfree_IC
import spectra_fit
import comparisons
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
    pardict['attempt_data'] = 'p' + str(rplanet) + '_star' + str(rstar) + '_' \
        + str(tstar) + '_' + str(loggstar) + '_spot' + str(tumbra) \
        + '_' + str(tpenumbra) + '_mag' + str(magstar)
    pardict['data_folder'] = pardict['project_folder'] + pardict['instrument'] \
                            + 'simulated_data/' + pardict['attempt_data'] + '/'
    if not os.path.exists(pardict['data_folder']):
        os.mkdir(pardict['data_folder'])
    pardict['attempt_mcmc'] = pardict['attempt_data'] + '/'
    #if not os.path.exists(pardict['attempt_mcmc']):
    #    os.mkdir(pardict['attempt_mcmc'])
    pardict['chains_folder']  = pardict['project_folder'] \
            + pardict['instrument'] + 'MCMC/' + pardict['attempt_mcmc']
    if not os.path.exists(pardict['chains_folder']):
        os.mkdir(pardict['chains_folder'])
    pardict['pandexo_out'] = pardict['data_folder'] + 'singlerun.pic'
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
    #if pathlib.Path(pardict['logfile_data']).exists() or \
    #pathlib.Path(pardict['logfile_chains']).exists():
    #    print('Log file already exists. Continue?')
    #    pdb.set_trace()
    #    os.system('rm ' + pardict['logfile_data'])
    #    os.system('rm ' + pardict['logfile_chains'])

    # These require python 2 (maybe)
    #

    res = 20
    #neptune_mdwarf.generate_spectrum(pardict)
    #lastchan = neptune_mdwarf.add_spots(pardict, res)
    #neptune_mdwarf.channelmerge(pardict, 3, 33)

    # Create logfile for chains
    #GP_white.GP_transit_spectro(pardict)

    # Select channels
    expchan = np.arange(22, 37, 3)
    #expchan = np.arange(1, 78, 3)
    #expchan = [1, 4, 8, 11, 15, 18, 22, 25, 28, 31, 34]
    # First, spot-free fit
    #spotfree_IC.transit_spectro(pardict, expchan)
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
    #comparisons.plot_spectra([folder_root + 'case1_masked/', folder_root \\
    #           + 'case1_spots/'], ['masked', 'spots'])

    spectra_fit.read_res(pardict, expchan, pardict['chains_folder'] \
                       + 'contrast_plot_kurucz.pdf')

    return
