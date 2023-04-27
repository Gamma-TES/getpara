# numpyモジュールを使う場合

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
from natsort import natsorted
import libs.getpara as gp
import libs.fft_spectrum as sp
import json

# ------parameter-------------
ch = 0
length = 10000
output = 'output'

set = gp.loadJson()
os.chdir(set["path"])
rate,samples,presamples,threshold = set["rate"],set["samples"],set["presamples"],set["threshold"]


path = natsorted(glob.glob(f'CH{ch}_noize/rawdata/CH{ch}_*.dat'))
arr = np.arange(len(path)) 
np.random.shuffle(arr) # 配列をシャッフルする
arr_select = arr[:length]

extruct = []

cnt = 0
for i in arr_select:
    path = f'CH{ch}_noize/rawdata/CH{ch}_{i}.dat'
    extruct.append(path)
extruct = natsorted(extruct)
print(extruct)

np.savetxt(f'CH{ch}_noize/random_noise.txt',extruct,fmt="%s")
