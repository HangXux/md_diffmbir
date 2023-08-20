import spekpy as sp
import matplotlib.pyplot as plt
import numpy as np


# Generate unfiltered spectrum
sl = sp.Spek(kvp=70, th=12, physics='spekcalc')
# Filter the spectrum
sl.filter('Be', 1.0).filter('Al', 4.0).filter('Air', 1000)
# Get energy values array and fluence arrays (return values at bin-edges)
karr_low, spkarr_low = sl.get_spectrum(edges=False)
norm_fact_l = (0.5 * spkarr_low).sum()
spkarr_low = spkarr_low / norm_fact_l   # normalize spectrum

# Generate unfiltered spectrum
sh = sp.Spek(kvp=120, th=12, physics='spekcalc')
# Filter the spectrum
sh.filter('Be', 1.0).filter('Al', 4.0).filter('Air', 1000)
# Get energy values array and fluence arrays (return values at bin-edges)
karr_high, spkarr_high = sh.get_spectrum(edges=False)
norm_fact_h = (0.5 * spkarr_high).sum()
spkarr_high = spkarr_high / norm_fact_h
spec_high = np.array([karr_high, spkarr_high], dtype='float32')

# normalize spectrum
spectnorm_1 = (0.5*spkarr_low).sum()
spectnorm_2 = (0.5*spkarr_high).sum()

print('70kvp spectrum normalization:', spectnorm_1)
print('120kvp spectrum normalization:', spectnorm_2)

# 70 kvp mass attenuation coefficient
# soft tissue
# load arrays with data points
from numpy import loadtxt
waterdata_low = np.transpose(loadtxt('data/NIST_mu/70kvp_mu_soft_tissue.txt'))
water_energy_low = np.array(waterdata_low[0])
mu_water_low = np.array(waterdata_low[1])
# interp() to interpolate y values
mu_water_low_interp = np.interp(karr_low, water_energy_low, mu_water_low)

# bone
bonedata_low = np.transpose(loadtxt('data/NIST_mu/70kvp_mu_bone.txt'))
bone_energy_low = np.array(bonedata_low[0])
mu_bone_low = np.array(bonedata_low[1])
mu_bone_low_interp = np.interp(karr_low, bone_energy_low, mu_bone_low)

# plt.plot(water_energy_low, mu_water_low)
# plt.plot(bone_energy_low, mu_bone_low)
# plt.show()

# 120 kvp mass attenuation coefficient
# soft tissue
waterdata_high = np.transpose(loadtxt('data/NIST_mu/120kvp_mu_soft_tissue.txt'))
water_energy_high = np.array(waterdata_high[0])
mu_water_high = np.array(waterdata_high[1])
mu_water_high_interp = np.interp(karr_high, water_energy_high, mu_water_high)

# bone
bonedata_high = np.transpose(loadtxt('data/NIST_mu/120kvp_mu_bone.txt'))
bone_energy_high = np.array(bonedata_high[0])
mu_bone_high = np.array(bonedata_high[1])
mu_bone_high_interp = np.interp(karr_high, bone_energy_high, mu_bone_high)

# save spectrum and mass attenuation coefficient data for low and high energy
data_low = np.array([karr_low, spkarr_low, mu_bone_low_interp, mu_water_low_interp], dtype='float32')
data_high = np.array([karr_high, spkarr_high, mu_bone_high_interp, mu_water_high_interp], dtype='float32')
np.save('data/70kvp_data', data_low)
np.save('data/120kvp_data', data_high)

# Plot spectrum
# plt.plot(karr_low, spkarr_low)
# plt.plot(karr_high, spkarr_high)
plt.figure(1)
plt.plot(data_low[0], data_low[1], 'b', label='70 kVp')
plt.plot(data_high[0], data_high[1], 'r', label='120 kVp')
plt.xlabel('Energy [keV]')
# plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
plt.ylabel('Normalized fluence')
plt.title('An example of x-ray spectrum')
plt.legend()

plt.figure(2)
plt.subplot(121), plt.title('(a) Mass attenuation at 70 kVp'), plt.xlabel('Energy [keV]'), plt.ylabel('$cm^2/g$')
plt.plot(data_low[0], data_low[2], 'b', label='bone')
plt.plot(data_low[0], data_low[3], 'r', label='soft tissue')
plt.legend()
plt.subplot(122), plt.title('(b) Mass attenuation at 120 kVp'), plt.xlabel('Energy [keV]'), plt.ylabel('$cm^2/g$')
plt.plot(data_high[0], data_high[2], 'b', label='bone')
plt.plot(data_high[0], data_high[3], 'r', label='soft tissue')
plt.legend()
plt.show()

