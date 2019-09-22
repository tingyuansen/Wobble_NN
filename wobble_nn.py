# import packages
import numpy as np
import torch
from torchsearchsorted import searchsorted


#========================================================================================================
# restore training set
temp = np.load("fitting_spectra.npz")
spec_shifted = temp["spec_shifted"]
RV_array = temp["RV_array"]
spec_rest = temp["spec_rest"]
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
rest_spec_model = rest_spec()
rv_model = radial_velocity()

# make it GPU accessible
rest_spec_model.cuda()
rv_model.cuda()


#========================================================================================================
# assume L2 loss
loss_fn = torch.nn.L1Loss()

# make pytorch variables
wave = torch.from_numpy(wavelength).type(torch.cuda.FloatTensor)
spec_shifted_torch = torch.from_numpy(spec_shifted).type(torch.cuda.FloatTensor)

# make a wavelength grid to allow for mutliple RV shifts simultaneously
# during interpolation
wave_minus_1 = wave[:-1].clone()
wave_cat = wave_minus_1.repeat(num_obs).view((num_obs,wave_minus_1.shape[0]))

#-------------------------------------------------------------------------------------------------------
# light speed for RV shift
c = 3e5 #km/s

# optimizer hyperparameters
learning_rate_spec = 1e-2
learning_rate_rv = 1e-2
optimizer = torch.optim.Adam([{'params': rest_spec_model.parameters(), "lr": learning_rate_spec},\
                              {'params': rv_model.parameters(), "lr": learning_rate_rv}])


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

    # extract model
    spec = rest_spec_model.spec
    RV_pred = rv_model.rv

    # RV shift
    doppler_shift = torch.sqrt((1 - RV_pred/c)/(1 + RV_pred/c))
    new_wavelength = torch.ger(wave, doppler_shift).T # torch.ger = outer product
    #new_wavelength_1 = new_wavelength.clone() ## not sure why I need this line.. else searhsorted complaint
    ind = searchsorted(wave_cat, new_wavelength).type(torch.LongTensor)

    # fix border indexing problem
    ind[ind == num_pixel - 1] = num_pixel - 2

    # calculate adjacent gradient
    slopes = (spec[1:] - spec[:-1])/(wave[1:]-wave[:-1])

    # linear interpolate
    spec_shifted_recovered = spec[ind] + slopes[ind]*(new_wavelength - wave[ind])

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
             spec_rest_recovered = spec.cpu().detach().numpy(),\
             rv_recovered = RV_pred.cpu().detach().numpy())
