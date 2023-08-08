

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
import sys
import json
12.16,12.28
#-------------- ガウスフィッティング ----------------
BINS = 2048
# K_x  = -77keV

def gausse(x,A,mu,sigma):
	return A*np.exp(-(x-mu)**2/(2.0*sigma**2))

def main():

	para = 'height_opt_temp'
	set = gp.loadJson()
	ch,path = set["Config"]["channel"],set["Config"]["path"]
	output = f'CH{set["Config"]["channel"]}_pulse/output/{set["Config"]["output"]}'
	
	os.chdir(path)
	df = pd.read_csv((f'{output}/output.csv'),index_col=0)
	df_sel = gp.select_condition(df,set)

	plt.hist(df_sel[para], bins = BINS)
	plt.xlabel('Channel',fontsize = 14)
	plt.ylabel('Count',fontsize = 14)
	plt.tick_params(axis='both', which='both', direction='in',\
					bottom=True, top=True, left=True, right=True)
	plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
	plt.show()


	energy = float(input('input Energy(keV): '))
	min = float(input('min: '))
	max = float(input('max: '))
	bins = int(input('bins: '))
	
	pulseheight = df_sel[(df_sel['height_opt_temp'] > min)&(df_sel['height_opt_temp'] < max)]
	hist, bin_edges = np.histogram(pulseheight['height_opt_temp'], bins=bins)
	p0 = [np.max(hist),(max-min)/6,bin_edges[np.argmax(hist)]]
	
	params,cov = curve_fit(gausse,bin_edges[:-1],hist,p0=p0,maxfev=10000)
	a,mu,sigma=params[0],params[1],np.abs(params[2])
	x_fit = np.linspace(min,max,1000)
	y_fit = gausse(x_fit,*params)
	fwhw = gp.FWHW(sigma)
	dE = energy*fwhw/mu
	
	fit_para = {energy:
			{
			"A":a,
			"mu":mu,
			"sigma":sigma,
			"fwhw":fwhw,
			"dE":dE
			}
		}
	
	if os.path.exists(f'{output}/{set["select"]["output"]}/gfit.json'):
		with open(f'{output}/{set["select"]["output"]}/gfit.json') as f:
			jsn = json.load(f)
		jsn.update(fit_para)
		jsn = json.dumps(jsn,indent=4)
	else:
		jsn = json.dumps(fit_para,indent=4)
	with open(f'{output}/{set["select"]["output"]}/gfit.json', 'w') as file:
		file.write(jsn)

	print(f'\nA: {a}\nmu: {mu}\nsigma: {sigma}')
	print(f'\nFWHW: {fwhw}\nEnergy resoltion: {dE} keV')

	plt.hist(pulseheight[para], bins =bins)
	plt.title(f'dE = {dE:.3f}keV @{int(energy)} keV')
	plt.plot(x_fit, y_fit, color='red',linewidth=1,linestyle='-')
	plt.xlabel('Channel',fontsize = 14)
	plt.ylabel('Count',fontsize = 14)
	plt.tick_params(axis='both', which='both', direction='in',\
					bottom=True, top=True, left=True, right=True)
	plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.2)
	plt.savefig(f'{output}/{set["select"]["output"]}/E_{int(energy)}keV.png')
	plt.show()


if __name__ == '__main__':
	main()
