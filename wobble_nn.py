# import packages
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
from torchsearchsorted import searchsorted


#========================================================================================================
# restore training set
temp = np.load("fitting_spectra.npz")
spec_shifted = temp["spec_shifted"]
RV_array = temp["RV_array"]
spectra_rest = temp["spectra_rest"]
wavelength = temp["wavelength"]

# number of trianing epoch
num_epoch = 1e3

#========================================================================================================
# number of pixesls
num_pixel = 7214
num_obs = 30

#----------------------------------------------------------------------------------------------------------
# make a rest frame model
class rest_spec(torch.nn.Module):
    def __init__(self):
        super(rest_spec, self).__init__()

        ## initialize with an array of [0,1] uniform numbers
        self.spec = torch.nn.Parameter(torch.rand(num_pixel))

    def forward(self):
        y_pred = self.spec
        return y_pred

#----------------------------------------------------------------------------------------------------------
# make radial velocity prediction
class radial_velocity(torch.nn.Module):
    def __init__(self):
        super(radial_velocity, self).__init__()
        self.rv = torch.nn.Parameter(torch.rand(num_obs))

    def forward(self):
        y_pred = self.rv
        return y_pred

#----------------------------------------------------------------------------------------------------------
# initiate the model
rest_spec_model = rest_spec()
rv_model = radial_velocity()
rest_spec_model.cuda()
rv_model.cuda()

#========================================================================================================
# now optimize
### run on GPU ###
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#---------------------------------------------------------------------------------------------------------
# assume L2 loss
loss_fn = torch.nn.L1Loss()

# make pytorch variables
wave = torch.from_numpy(wavelength).type(dtype)
wave_minus_1 = wave[:-1].clone()

# set the limits to extreme to make sure that it bracket the new wavelength grid
# during interpolation
wave_cat = wave_minus_1.repeat(num_obs).view((num_obs,wave_minus_1.shape[0]))
spec_shifted_torch = torch.from_numpy(spec_shifted).type(dtype)

# light speed for doppler shift
c = 3e5 #km/s

# optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam([{'params': rest_spec_model.parameters(), "lr": learning_rate},\
                              {'params': rv_model.parameters(), "lr": learning_rate}])

# initiate training
loss_data = 10**8
training_loss = []

#---------------------------------------------------------------------------------------------------------
# optimize
for i in range(int(num_epoch)):
    if i % 10**1 == 0:
        print('Step ' + str(i) \
                + ': Training set loss = ' + str(int(loss_data*1e5)/1e5))

    # doppler shift
    spec = rest_spec_model.spec
    RV_pred = rv_model.rv
    doppler_shift = torch.sqrt((1 - RV_pred/c)/(1 + RV_pred/c))

    # torch.ger = np.outer, outer product
    new_wavelength = torch.t(torch.ger(wave, doppler_shift))
    new_wavelength_1 = new_wavelength.clone()

    ind = searchsorted(wave_cat, new_wavelength_1).type(torch.LongTensor)
    print('pass')

    # fix a border index problem
    ind[ind == num_pixel - 1] = num_pixel - 2
    slopes = (spec[1:] - spec[:-1])/(wave[1:]-wave[:-1])
    spec_shifted_recovered = spec[ind] + slopes[ind]*(new_wavelength - wave[ind])

    # the loss function is simply comparing the reconstructed spectra vs. spectrum
    loss = loss_fn(spec_shifted_recovered, spec_shifted_torch)

#---------------------------------------------------------------------------------------------------------
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record training loss
    loss_data = loss.data.item()
    training_loss.append(loss_data)
