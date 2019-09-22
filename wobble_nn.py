# import packages
import numpy as np
import torch
from torchsearchsorted import searchsorted


#========================================================================================================
# restore training set
temp = np.load("fitting_spectra.npz")
spec_shifted = temp["spec_shifted"]
wavelength = temp["wavelength"]

# number of pixesls and epoch
num_pixel = spec_shifted.shape[1]
num_obs = spec_shifted.shape[0]

# number of trianing epoch
num_epoch = 1e4


#========================================================================================================
# make a rest frame spectral model
class rest_spec(torch.nn.Module):
    def __init__(self):
        super(rest_spec, self).__init__()

        # initialize with an array of uniform random number
        self.spec = torch.nn.Parameter(torch.rand(num_pixel))

    def forward(self):
        y_pred = self.spec
        return y_pred

#----------------------------------------------------------------------------------------------------------
# radial velocity model
class radial_velocity(torch.nn.Module):
    def __init__(self):
        super(radial_velocity, self).__init__()
        self.rv = torch.nn.Parameter(torch.rand(num_obs))

    def forward(self):
        y_pred = self.rv
        return y_pred

#----------------------------------------------------------------------------------------------------------
# initiate the model
rest_spec_model_1 = rest_spec()
rv_model_1 = radial_velocity()

rest_spec_model_2 = rest_spec()
rv_model_2 = radial_velocity()

# make it GPU accessible
rest_spec_model_1.cuda()
rv_model_1.cuda()

rest_spec_model_2.cuda()
rv_model_2.cuda()


#========================================================================================================
# assume L2 loss
loss_fn = torch.nn.L1Loss()

# make pytorch variables
wave = torch.from_numpy(wavelength).type(torch.cuda.FloatTensor)
spec_shifted_torch = torch.from_numpy(spec_shifted).type(torch.cuda.FloatTensor)

# make a wavelength grid to allow for mutliple RV shifts simultaneously
# during interpolation
wave_minus = wave[:-1].clone()
wave_cat = wave_minus.repeat(num_obs).view((num_obs,wave_minus.shape[0]))

#-------------------------------------------------------------------------------------------------------
# light speed for RV shift
c = 3e5 #km/s

# optimizer hyperparameters
learning_rate_spec = 1e-2
learning_rate_rv = 1e-2
optimizer = torch.optim.Adam([{'params': rest_spec_model_1.parameters(), "lr": learning_rate_spec},\
                              {'params': rv_model_1.parameters(), "lr": learning_rate_rv},\
                              {'params': rest_spec_model_2.parameters(), "lr": learning_rate_spec},\
                              {'params': rv_model_2.parameters(), "lr": learning_rate_rv}])


#========================================================================================================
# initiate training
loss_data = 10**8
training_loss = []

#---------------------------------------------------------------------------------------------------------
# optimize
for i in range(int(num_epoch)):
    if i % 10**2 == 0:
        print('Step ' + str(i) \
                + ': Training set loss = ' + str(int(loss_data*1e5)/1e5))

#---------------------------------------------------------------------------------------------------------
    # spectrum 1
    # extract model
    spec_1 = rest_spec_model_1.spec
    RV_pred_1 = rv_model_1.rv

    # RV shift
    doppler_shift_1 = torch.sqrt((1 - RV_pred_1/c)/(1 + RV_pred_1/c))
    new_wavelength_1 = torch.t(torch.ger(wave, doppler_shift_1)).contiguous() # torch.ger = outer product
    ind_1 = searchsorted(wave_cat, new_wavelength_1).type(torch.LongTensor)

    # fix border indexing problem
    ind_1[ind_1 == num_pixel - 1] = num_pixel - 2

    # calculate adjacent gradient
    slopes_1 = (spec_1[1:] - spec_1[:-1])/(wave[1:]-wave[:-1])

    # linear interpolate
    spec_shifted_recovered_1 = spec_1[ind] + slopes_1[ind]*(new_wavelength_1 - wave[ind])

#---------------------------------------------------------------------------------------------------------
    # spectrum 2
    spec_2 = rest_spec_model_2.spec
    RV_pred_2 = rv_model_2.rv

    # RV shift
    doppler_shift_2 = torch.sqrt((1 - RV_pred_2/c)/(1 + RV_pred_2/c))
    new_wavelength_2 = torch.t(torch.ger(wave, doppler_shift_2)).contiguous() # torch.ger = outer product
    ind_2 = searchsorted(wave_cat, new_wavelength_2).type(torch.LongTensor)

    # fix border indexing problem
    ind_2[ind_2 == num_pixel - 1] = num_pixel - 2

    # calculate adjacent gradient
    slopes_2 = (spec_2[1:] - spec_2[:-1])/(wave[1:]-wave[:-1])

    # linear interpolate
    spec_shifted_recovered_2 = spec_2[ind] + slopes_2[ind]*(new_wavelength_2 - wave[ind])

#---------------------------------------------------------------------------------------------------------
    # combine prediction
    spec_shifted_recovered = spec_shifted_recovered_1*spec_shifted_recovered_2

#---------------------------------------------------------------------------------------------------------
    # the loss function is simply comparing the reconstructed spectra vs. obs spectra
    loss = loss_fn(spec_shifted_recovered, spec_shifted_torch)

    # back propagation to optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record training loss
    loss_data = loss.data.item()
    training_loss.append(loss_data)

#---------------------------------------------------------------------------------------------------------
    # save results
    np.savez("../results.npz",\
             spec_shifted_recovered = spec_shifted_recovered.cpu().detach().numpy(),\
             spec_shifted_recovered_1 = spec_shifted_recovered_1.cpu().detach().numpy(),\
             spec_rest_recovered_1 = spec_1.cpu().detach().numpy(),\
             rv_recovered_1 = RV_pred_1.cpu().detach().numpy(),\
             spec_shifted_recovered_2 = spec_shifted_recovered_2.cpu().detach().numpy(),\
             spec_rest_recovered_2 = spec_2.cpu().detach().numpy(),\
             rv_recovered_2 = RV_pred_2.cpu().detach().numpy())
