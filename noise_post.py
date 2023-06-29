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
import pprint
import sys
import re
import matplotlib.cm as cm


path = "E:/matsumi/data/noise"
noise_post = np.loadtxt(f'{path}/modelnoise_post.txt')
noise_single = np.loadtxt(f'{path}/modelnoise.txt')

plt.plot(noise_post,label = 'spiral post')
plt.plot(noise_single,label = 'single')
plt.legend()
plt.grid()
plt.loglog()
plt.show()