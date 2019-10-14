#%% import packages
import numpy as np
import torch
from torchsearchsorted import searchsorted
const_c=2.99792458e5

#%%
import matplotlib.pyplot as pl
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='sans-serif')
from matplotlib import rc
rc('text', usetex=True)

#%%
%matplotlib inline

#%%========================================================================================================
# load data
data = np.load('spectra_jupiter_hds.npz')
spec_shifted, wavelength = data['fluxes'], data['wavelengths']

#
num_pixel_omit = 100
num_pixel = spec_shifted.shape[1]
num_obs = spec_shifted.shape[0]

#%%========================================================================================================
# make a rest frame spectral model
class rest_spec(torch.nn.Module):
    def __init__(self):
        super(rest_spec, self).__init__()
        # initiate with a random epoch observation to facilitate convergence
        self.spec = torch.nn.Parameter(torch.from_numpy(spec_shifted[0,:]).type(dtype))

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
# no radial velocity (e.g. telluric)
class telluric_velocity(torch.nn.Module):
    def __init__(self):
        super(telluric_velocity, self).__init__()
        self.rv = torch.nn.Parameter(torch.from_numpy(np.zeros(num_obs)).type(dtype))

    def forward(self):
        y_pred = self.rv
        return y_pred

#----------------------------------------------------------------------------------------------------------
# radial velocity model (rotation)
from torch.autograd import Variable
class rotation(torch.nn.Module):
    def __init__(self):
        super(rotation, self).__init__()
        self.rv = torch.nn.Parameter(torch.from_numpy(np.array([15.])).type(dtype))

    def forward(self):
        vrot = self.rv
        return torch.cat([vrot*0, vrot*1, -vrot]), torch.cat([vrot*0, vrot*2, -2*vrot])

npoly = 1
class continuum(torch.nn.Module):
    def __init__(self):
        super(continuum, self).__init__()
        self.coeff = torch.nn.Parameter(torch.from_numpy(np.array([1]+[0]*npoly)).type(dtype).repeat(num_obs).view((num_obs,-1)))

    def forward(self, x):
        c = self.coeff
        x = x - torch.mean(x)
        _x = torch.cat([x ** i for i in range(npoly+1)]).view(npoly+1,-1)
        return torch.cat([torch.mv(_x.t(), c[i]) for i in range(num_obs)]).view(num_obs,-1)

#%%========================================================================================================
# cpu or gpu?
#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor

# make pytorch variables
wave = torch.from_numpy(wavelength).type(dtype)
spec_shifted_torch = torch.from_numpy(spec_shifted).type(dtype)

# make a wavelength grid to allow for mutliple RV shifts simultaneously
# during interpolation
wave_minus = wave[:-1].clone()
wave_cat = wave_minus.repeat(num_obs).view((num_obs,wave_minus.shape[0]))

#%%========================================================================================================
# initiate the model
rest_spec_model_1 = rest_spec()
rest_spec_model_2 = rest_spec()
rest_spec_model_3 = rest_spec()
#rv_model_1 = radial_velocity()
#rv_model_2 = radial_velocity()
rv_model_rot = rotation()
rv_model_3 = telluric_velocity()
cont_model = continuum()

# make it GPU accessible
if 'cuda' in str(dtype):
    rest_spec_model_1.cuda()
    rest_spec_model_2.cuda()
    rest_spec_model_3.cuda()
    rv_model_1.cuda()
    rv_model_2.cuda()
    rv_model_3.cuda()
    rv_model_rot.cuda()

#%%========================================================================================================
# optimizer hyperparameters
learning_rate_spec = 1e-2
learning_rate_rv = 1e-2
optimizer = torch.optim.Adam([{'params': rest_spec_model_1.parameters(), "lr": learning_rate_spec},\
                              {'params': rest_spec_model_2.parameters(), "lr": learning_rate_spec},\
                              #{'params': rv_model_1.parameters(), "lr": learning_rate_rv},\
                              {'params': cont_model.parameters(), "lr": learning_rate_spec},\
                              {'params': rv_model_rot.parameters(), "lr": learning_rate_rv},\
                              {'params': rest_spec_model_3.parameters(), "lr": learning_rate_spec}])

# define loss function
loss_fn = torch.nn.MSELoss()

# L1 regularizer
lambda_L1=1e-6*5

#%%---------------------------------------------------------------------------------------------------------
# initiate training
num_train = 1e4
loss_data = 10**8
training_loss = []
vrot_arr = []

# optimize
for i in range(int(num_train/5)):
    if i % 10**2 == 0:
        print('Step ' + str(i) \
                + ': Training set loss = ' + str(int(loss_data*1e5)/1e5))

#---------------------------------------------------------------------------------------------------------
    # spectrum 1
    # extract model
    spec_1 = rest_spec_model_1.spec
    #RV_pred_1 = rv_model_1.rv

    RV_pred_1, RV_pred_2 = rv_model_rot.forward()

    # RV shift
    doppler_shift = torch.sqrt((1 - RV_pred_1/const_c)/(1 + RV_pred_1/const_c))
    new_wavelength = torch.t(torch.ger(wave, doppler_shift)).contiguous() # torch.ger = outer product
    ind = searchsorted(wave_cat, new_wavelength).type(torch.LongTensor)

    # fix border indexing problem
    ind[ind == num_pixel - 1] = num_pixel - 2

    # calculate adjacent gradient
    slopes = (spec_1[1:] - spec_1[:-1])/(wave[1:]-wave[:-1])

    # linear interpolate
    spec_shifted_recovered_1 = spec_1[ind] + slopes[ind]*(new_wavelength - wave[ind])

#---------------------------------------------------------------------------------------------------------
    # spectrum 2
    spec_2 = rest_spec_model_2.spec
    #RV_pred_2 = rv_model_2.rv
    doppler_shift = torch.sqrt((1 - RV_pred_2/const_c)/(1 + RV_pred_2/const_c))
    new_wavelength = torch.t(torch.ger(wave, doppler_shift)).contiguous() # torch.ger = outer product
    ind = searchsorted(wave_cat, new_wavelength).type(torch.LongTensor)
    ind[ind == num_pixel - 1] = num_pixel - 2
    slopes = (spec_2[1:] - spec_2[:-1])/(wave[1:]-wave[:-1])
    spec_shifted_recovered_2 = spec_2[ind] + slopes[ind]*(new_wavelength - wave[ind])

    # spectrum 3
    spec_3 = rest_spec_model_3.spec
    RV_pred_3 = rv_model_3.rv
    doppler_shift = torch.sqrt((1 - RV_pred_3/const_c)/(1 + RV_pred_3/const_c))
    new_wavelength = torch.t(torch.ger(wave, doppler_shift)).contiguous() # torch.ger = outer product
    ind = searchsorted(wave_cat, new_wavelength).type(torch.LongTensor)
    ind[ind == num_pixel - 1] = num_pixel - 2
    slopes = (spec_3[1:] - spec_3[:-1])/(wave[1:]-wave[:-1])
    spec_shifted_recovered_3 = spec_3[ind] + slopes[ind]*(new_wavelength - wave[ind])

#---------------------------------------------------------------------------------------------------------
    # combine prediction
    spec_shifted_recovered = spec_shifted_recovered_1*spec_shifted_recovered_2*spec_shifted_recovered_3
    spec_shifted_recovered *= cont_model.forward(new_wavelength[0])

#---------------------------------------------------------------------------------------------------------
    # the loss function is simply comparing the reconstructed spectra vs. obs spectra
    ### add aditional condition to regularize with physical intuitions if needed ###
    loss = loss_fn(spec_shifted_recovered, spec_shifted_torch)

    loss += lambda_L1*torch.norm(torch.cat([x.view(-1)-1 for x in rest_spec_model_1.parameters()]), 1)
    loss += lambda_L1*torch.norm(torch.cat([x.view(-1)-1 for x in rest_spec_model_2.parameters()]), 1)
    loss += lambda_L1*torch.norm(torch.cat([x.view(-1)-1 for x in rest_spec_model_3.parameters()]), 1)

#---------------------------------------------------------------------------------------------------------
    # back propagation to optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record training loss
    loss_data = loss.data.item()
    training_loss.append(loss_data)
    vrot_arr.append(rv_model_rot.rv.data.item())

print ()
print ('L2 norm:', torch.norm(spec_shifted_recovered-spec_shifted_torch, 2).data.item())
print ('rotation velocity:', rv_model_rot.rv.data.item())

#%%
pl.plot(vrot_arr);

#%%
rawmodels=[spec_1.detach().numpy(), spec_2.detach().numpy(), spec_3.detach().numpy()]
models=[spec_shifted_recovered_1.detach().numpy(), spec_shifted_recovered_2.detach().numpy(), spec_shifted_recovered_3.detach().numpy()]

#%%
rvs=[RV_pred_1.cpu().detach().numpy(), RV_pred_2.cpu().detach().numpy(), RV_pred_3.cpu().detach().numpy()]
#print (rvs)

#%%
conts=cont_model.forward(new_wavelength[0])
conts=[conts[i].detach().numpy() for i in range(num_obs)]

#%%
mlabels=['planetary', 'stellar', 'telluric']
dlabels=['center', 'west', 'east']
num_pix_drop=100
wmin, wmax=np.min(wavelength[num_pix_drop:-num_pix_drop]), np.max(wavelength[num_pix_drop:-num_pix_drop])
for i in range(num_obs):
    pl.figure(figsize=(12,9))
    pl.xlim(wmin, wmax)
    pl.xlabel('Wavelength (angstrom)')
    pl.ylabel('Normalized flux')
    pl.plot(wavelength[num_pix_drop:-num_pix_drop], (spec_shifted[i]/conts[i])[num_pix_drop:-num_pix_drop], '.', color='gray', lw=6, alpha=0.6, label='data')
    for j in range(3):
        pl.plot(wavelength[num_pix_drop:-num_pix_drop], models[j][i][num_pix_drop:-num_pix_drop], '-', alpha=0.8, label='model (%s)'%mlabels[j])
    pl.plot(wavelength[num_pix_drop:-num_pix_drop], np.prod(models, axis=0)[i][num_pix_drop:-num_pix_drop], '-', color='gray', lw=3, alpha=0.6, label='model sum')
    y0=np.min(spec_shifted[i][num_pix_drop:-num_pix_drop])-0.05
    pl.plot(wavelength[num_pix_drop:-num_pix_drop], y0+spec_shifted[i][num_pix_drop:-num_pix_drop]-(np.prod(models, axis=0)[i]*conts[i])[num_pix_drop:-num_pix_drop], '.',  color='gray', lw=2, alpha=0.8)
    pl.axhline(y=y0, lw=1, color='gray')
    pl.legend(loc='best', bbox_to_anchor=(1,1))
    pl.savefig('data_%s.png'%dlabels[i], dpi=200, bbox_inches='tight')
    pl.show()

#%%---------------------------------------------------------------------------------------------------------
    # save results
    """
    np.savez("../results.npz",\
             spec_shifted_recovered = spec_shifted_recovered.cpu().detach().numpy(),\
             spec_shifted_recovered_1 = spec_shifted_recovered_1.cpu().detach().numpy(),\
             spec_rest_recovered_1 = spec_1.cpu().detach().numpy(),\
             rv_recovered_1 = RV_pred_1.cpu().detach().numpy(),\
             spec_shifted_recovered_2 = spec_shifted_recovered_2.cpu().detach().numpy(),\
             spec_rest_recovered_2 = spec_2.cpu().detach().numpy(),\
             rv_recovered_2 = RV_pred_2.cpu().detach().numpy())
    """
