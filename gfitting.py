#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 11-6D1
#Tsuruta Tetsuya 2019.06.14
#川上さんのを参考に作りました。
#python3に書き換え済

#----------- モジュールのインポート --------------
import math
import glob
import numpy as np
import scipy.optimize as opt
import random
from array import array
import os
import os.path
import matplotlib.pyplot as plt
import libs.getpara as gp
import pandas as pd
from scipy.optimize import curve_fit

#-------------- ガウスフィッティング ----------------

def gausse(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def main():
	set = gp.loadJson()
	ch,path = set["Config"]["channel"],set["Config"]["path"]
	rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]
	os.chdir(path)
	df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
	df = gp.select_condition(df,set)
	energy = float(input('input Energy(keV): '))
	min = float(input('min: '))
	max = float(input('max: '))
	bins = int(input('bins: '))
	
	pulseheight = df[(df['height_opt_temp'] > min)&(df['height_opt_temp'] < max)]
	hist, bin_edges = np.histogram(pulseheight['height_opt_temp'], bins=bins)
	p0 = [np.max(hist),(max-min)/6,bin_edges[np.argmax(hist)]]
	
	params,cov = curve_fit(gausse,bin_edges[:-1],hist,p0=p0,maxfev=10000)
	a,mu,sigma=params[0],params[1],np.abs(params[2])
	x_fit = np.linspace(min,max,1000)
	y_fit = gausse(x_fit,*params)
	fwhw = gp.FWHW(sigma)
	dE = energy*fwhw/mu

	print(f'\nFWHW: {fwhw}\nEnergy resoltion: {dE} keV')
	

	plt.hist(pulseheight['height_opt_temp'], bins =bins)
	plt.title(f'dE = {dE:.3f}keV @{energy} keV')
	plt.plot(x_fit, y_fit, color='red',linewidth=1,linestyle='-')
	plt.xlabel('Channel',fontsize = 14)
	plt.ylabel('Count',fontsize = 14)
	plt.tick_params(axis='both', which='both', direction='in',\
                    bottom=True, top=True, left=True, right=True)
	plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
	plt.savefig(f'CH{ch}_pulse/output/select/E_resolution_{energy}.png')
	plt.show()


if __name__ == '__main__':
	main()
