import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import griddata
from sklearn.preprocessing import minmax_scale

import pywt

raw_data = pd.read_csv('sliced_FS2e1.csv', sep=',', header=None)
mother_wavelet = pywt.Wavelet('db3')

# A10 Coefficient
coeffs_ten = pywt.wavedec(raw_data.iloc[:,4], mother_wavelet, level=10)
coeffs_eight =pywt.wavedec(raw_data.iloc[:,4], mother_wavelet, level=8)
coeffs_six =pywt.wavedec(raw_data.iloc[:,4], mother_wavelet, level=6)

print('size of A10 coeff:::', coeffs_ten[0].size,'size of raw coeff:::', raw_data.iloc[:,4].size)

#upsample coeffs_ten to have 1000 data points
raw_data = np.array(raw_data.iloc[:,4])
coeffs_ten = np.array(coeffs_ten[0])
coeffs_eight = np.array(coeffs_eight[0])
coeffs_six = np.array(coeffs_six[0])

print('A10 Size->',coeffs_ten.size,'///A8 Size->',coeffs_eight.size,'///A6 Size->', coeffs_six.size)

srate = 1
npnts = len(coeffs_ten)
timeA10 = np.arange(0, npnts) / srate
timeA8 = np.arange(0, npnts) / srate
timeA6 = np.arange(0, npnts) / srate

print('time:::', timeA10)

upsampleFactor = 1000 / len(coeffs_ten)
newNpnts = npnts * upsampleFactor
print('newNPnts:::', newNpnts)

newTime = np.arange(0, newNpnts) / (upsampleFactor * srate)

downdataRaw = signal.resample(raw_data, int(newNpnts))
updataA10 = griddata(timeA10, coeffs_ten, newTime, method='cubic')
updataA8 = griddata(timeA8, coeffs_eight, newTime, method='cubic')
updataA6 = griddata(timeA6, coeffs_six, newTime, method='cubic')

## Cut the last few samples to match the data length of 1000
# SCVSR = A8 - A10
# SCSR = A6 - A10
phasic = []
scvsr = []
scsr = []
for i in range(0, 900):
    phasic.append(downdataRaw[i] - updataA10[i])
    scvsr.append(updataA8[i] - updataA10[i])
    scsr.append(updataA6[i] - updataA10[i])

norm_phasic = minmax_scale(phasic)
norm_scvsr =  minmax_scale(scvsr)
norm_scsr = minmax_scale(scsr)

for i in range(0, len(norm_scvsr)):
    norm_phasic[i] = norm_phasic[i] - 0.5
    norm_scvsr[i] = norm_scvsr[i] - 0.5
    norm_scsr[i] = norm_scsr[i] - 0.5

plt.plot(norm_phasic)
plt.plot(norm_scvsr)
plt.plot(norm_scsr)
plt.show()

# number of zero crossings counted for each coefficient
zero_cross_phasic = np.nonzero(np.diff(norm_phasic > 0))[0]
zero_cross_scvsr = np.nonzero(np.diff(norm_scvsr > 0))[0]
zero_cross_scsr = np.nonzero(np.diff(norm_scsr > 0))[0]

rate_zero_cross_phaic = zero_cross_phasic.size
rate_zero_cross_scvsr = zero_cross_scvsr.size
rate_zero_cross_scsr = zero_cross_scsr.size















