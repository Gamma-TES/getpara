
import numpy as np
import scipy.optimize as opt
import scipy.fftpack as fft
import os
import matplotlib.pyplot as plt
import libs.getpara as gp
import pandas as pd
from scipy.optimize import curve_fit

def main():
    #----------- パルスハイトの読み込み ----------------
    set = gp.loadJson()
    path = "/Volumes/Extreme 1TB/Matsumi/data/20230418/room2-2_140mK_870uA_gain10_trig0.1_10kHz"
    os.chdir(path)
    df = pd.read_csv((f'CH{set["channel"]}_pulse/output/output.csv'),index_col=0)
    rate,samples,presamples,threshold,ch = set["rate"],set["samples"],set["presamples"],set["threshold"],set["channel"]
    pulseheight = df["height_opt_temp"]
