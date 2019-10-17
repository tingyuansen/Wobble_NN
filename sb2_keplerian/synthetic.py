#%% import packages
import numpy as np
import torch
import exoplanet as xo
const_c=2.99792458e5

#%%
import matplotlib.pyplot as pl
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
from matplotlib import rc
rc('text', usetex=True)

#%%
%matplotlib inline

#%% import bin spec packages
from binspec import utils
from binspec import spectral_model
from binspec import fitting
from binspec.spectral_model import get_unnormalized_spectrum_single_star,\
get_normalized_spectrum_single_star, get_Teff2_logg2_NN

#%%
flum = 1e-2
fnoise = 1e-2
num_obs = 300


# read in the standard wavelength grid onto which we interpolate spectra.
wavelength = utils.load_wavelength_array()
num_pixel = wavelength.size

# define pixels for continuum normalization
cont_pixels = utils.load_cannon_contpixels()

# read in all individual neural networks we'll need.
NN_coeffs_norm = utils.read_in_neural_network(name = 'normalized_spectra')
NN_coeffs_flux = utils.read_in_neural_network(name = 'unnormalized_spectra')
NN_coeffs_R = utils.read_in_neural_network(name = 'radius')
NN_coeffs_Teff2_logg2 = utils.read_in_neural_network(name = 'Teff2_logg2')

#%%
from PyAstronomy import pyasl
def rvs_keplerian(times, K, period, ecc, omega, t0=None, tau=None):
    if t0 is None and tau is None:
        print ('# give either t0 or rau.')
        return None
    if tau is None:
        E0 = 2*np.arctan(np.sqrt((1.-ecc)/(1.+ecc))*np.tan(0.25*np.pi-0.5*omega))
        M = E0 - ecc*np.sin(E0) + 2*np.pi*(times-t0)/period
    elif t0 is None:
        M = 2*np.pi*(times-tau)/period
    ks = pyasl.MarkleyKESolver()
    E = np.array([ks.getE(_m, ecc) for _m in M])
    f = 2*np.arctan(np.sqrt((1.+ecc)/(1.-ecc))*np.tan(0.5*E))
    rvs = K*(np.cos(omega+f)+ecc*np.cos(omega))
    return rvs

#%%
t=np.linspace(0, 200, 1000)
K1, K2, period, ecc, omega, tau=10, 100, 52, 0.6, 0.3*np.pi, 0
pl.xlabel('time')
pl.ylabel('RV')
pl.plot(t, rvs_keplerian(t, K1, period, ecc, omega, tau=tau), label='star')
pl.plot(t, rvs_keplerian(t, K2, period, ecc, omega+np.pi, tau=tau), label='planet')
pl.legend();


#%%
# make RV for the two stars differently
times = np.linspace(0, 100, num_obs)
K1, K2, period, ecc, omega, tau=10, 100, 32, 0.6, 0.3*np.pi, 0
RV_array_1 = rvs_keplerian(times, K1, period, ecc, omega, tau=tau)
RV_array_2 = rvs_keplerian(times, K2, period, ecc, omega+np.pi, tau=tau)
pl.ylabel('RV')
pl.plot(times, RV_array_1, '.', label='star 1')
pl.plot(times, RV_array_2, '.', label='star 2')
pl.legend(bbox_to_anchor=(1,1));

#%%--------------------------------------------------------------------------------------------------
# make rest frame spectrum for spectrum 1
Teff1, logg1, feh, alphafe, vmacro = 4750., 2.5, 0., 0., 2.,
labels1 = [Teff1, logg1, feh, alphafe, vmacro, 0]
spec_rest_1 = get_normalized_spectrum_single_star(labels = labels1,
                    NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)

# for spectrum 2
Teff2, logg2, feh, alphafe, vmacro = 6000., 4.0, 0., 0., 2.
labels2 = [Teff2, logg2, feh, alphafe, vmacro, 0]
spec_rest_2 = get_normalized_spectrum_single_star(labels = labels2,
                    NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)
spec_rest_2 = (spec_rest_2-1)*flum+1

#--------------------------------------------------------------------------------------------------
#%% radial velocity shift
spec_shifted_1 = []
for i in range(RV_array_1.size):
    doppler_factor = np.sqrt((1 - RV_array_1[i]/const_c)/(1 + RV_array_1[i]/const_c))
    new_wavelength = wavelength*doppler_factor
    ind = np.searchsorted(wavelength[:-1], new_wavelength) - 1
    slopes = (spec_rest_1[1:] - spec_rest_1[:-1])/(wavelength[1:]-wavelength[:-1])
    spec_shifted_1.append(spec_rest_1[ind] + slopes[ind]*(new_wavelength - wavelength[ind]))
spec_shifted_1 = np.array(spec_shifted_1)

spec_shifted_2 = []
for i in range(RV_array_2.size):
    doppler_factor = np.sqrt((1 - RV_array_2[i]/const_c)/(1 + RV_array_2[i]/const_c))
    new_wavelength = wavelength*doppler_factor
    ind = np.searchsorted(wavelength[:-1], new_wavelength) - 1
    slopes = (spec_rest_2[1:]-spec_rest_2[:-1])/(wavelength[1:]-wavelength[:-1])
    spec_shifted_2.append(spec_rest_2[ind] + slopes[ind]*(new_wavelength - wavelength[ind]))
spec_shifted_2 = np.array(spec_shifted_2)

# combine two normalized flux (ignoring flux ratio)
# to mock up observations
spec_shifted = spec_shifted_1*spec_shifted_2

noise = fnoise*np.random.randn(num_obs, num_pixel)/np.sqrt(spec_shifted)
spec_shifted += noise

#%%==================================================================================================
# plot the spectrum
lambda_min, lambda_max = 16000, 16050#  wavelength range for plotting
nomit = 20
pl.figure(figsize=(12,9))
pl.plot(wavelength[nomit:-nomit], spec_shifted.T[nomit:-nomit]*0.8, lw=0.5)
pl.xlim(lambda_min, lambda_max)
pl.plot(wavelength, spec_rest_1*1.2, lw=1, label='star 1') ## rest frame spectrum 1
pl.plot(wavelength, spec_rest_2*1.4, ls="-", lw=1, label='star 2') ## rest frame spectrum 2
pl.legend();
pl.show()

#%%
pl.figure(figsize=(12,9))
pl.xlim(lambda_min, lambda_max)
pl.plot(wavelength, spec_rest_2);

#%%
(wavelength[-1]-wavelength[0])/num_pixel/np.median(wavelength)*const_c

#%%==================================================================================================
# save array
flum
# cull the few last few pixels because interpolation there tend to extrapolate to weird values
np.savez("synthetic_f1e-2_n300.npz",\
         epochs = times,\
         spec_shifted = spec_shifted[:,nomit:-nomit],\
         spec_shifted_1 = spec_shifted_1[:,nomit:-nomit],\
         spec_shifted_2 = spec_shifted_2[:,nomit:-nomit],\
         RV_array_1 = RV_array_1,\
         RV_array_2 = RV_array_2,\
         spec_rest_1 = spec_rest_1[nomit:-nomit],\
         spec_rest_2 = spec_rest_2[nomit:-nomit],\
         wavelength = wavelength[nomit:-nomit])
