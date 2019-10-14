# Compare spot fits with stellar stellar models.
import numpy as np
import glob
import pickle
from pdb import set_trace
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import pysynphot
import plot_spectra

def read_res(pardict, ind, plotname):
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
    yerr = []
    for i in np.arange(len(A[:, 2])):
        if A[i, 1] > 0:
            yerr.append(max(abs(A[i, 2] - A[i, 1]), abs(A[i, 3] - A[i, 2])))
        else:
            yerr.append(abs(A[i, 3] - A[i, 2]))
    yerr = np.array(yerr)
    plt.figure()
    plt.errorbar(wl, sigma[:, 2], yerr=sigma[:, 2] - sigma[:, 0], fmt='ko')
    plt.title('$\sigma$', fontsize=16)
    plt.figure()
    plt.errorbar(wl, A[:, 2], yerr=yerr, fmt = 'ko')
    plt.title('$A$', fontsize=16)
    #plt.xlabel('Channel', fontsize=16)
    #plt.ylabel('Flux rise [ppm]', fontsize=16)
    '''
    plt.figure()
    plt.errorbar(ind, x0[:, 2], yerr=x0[:, 3] - x0[:, 2], fmt = 'ko')
    plt.title('$x_0$', fontsize=16)
    plt.figure()
    plt.errorbar(ind, sigma[:, 2], yerr=sigma[:, 3] - sigma[:, 2], fmt = 'ko')
    plt.title('$\sigma$', fontsize=16)
    '''

    # Get spectra with starspot contamination and perform LM fit
    for temp in np.arange(3000, 6000, 250):
        ww, spec = plot_spectra.combine_spectra(pardict, [temp], 0.05)
        ww/= 1e4
        spec*= A[:, 2].max()
        specint = interp1d(ww, spec)
        specA = specint(wl)
        soln = least_squares(scalespec, 1., args=(specA, \
                            A[:, 2], yerr))
        print('Spectrum scaling factor:', soln.x)
        plt.plot(ww, soln.x*spec, label=str(temp))
    plt.legend(frameon=False, loc='best', fontsize=16)
    plt.xlabel('Wavelength [$\mu m$]', fontsize=16)
    plt.ylabel('Transit depth rise [ppm]', fontsize=16)
    plt.xlim(0.3, 5.0)
    plt.show()
    plt.savefig(plotname)

    return np.array(wl), np.array(A), np.array(x0), np.array(sigma)

def scalespec(x, spec, y, yerr):
    '''
    Distance between model spectrum and data.
    '''
    return (x*spec - y)**2/yerr**2
