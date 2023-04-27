
# Matsumi
# Temperature Calibration

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import libs.getpara as gp

# Give textfile name as argment when you run this python file.
# ex)
# python .\temp_com_curve.py baseline.txt pulseheight_opt.txt


# pO is initialize parameter to fitting polynomial.
# Make sure the (length of p0 - 1) is the degree of the polynomial.
p0=[0.01,0.01,0.01,0.01,0.01,0.01]
# Plot xlim range

cal_range = [-0.6,0.2]
minBase = -0.5
maxBase = 2.0
     





#Investigate Which plot is in area sellected or out.

def func(X, *params):
    Y = np.zeros_like(X)
    for i, param in enumerate(params):
        Y = Y + np.array(param * X ** i)
    return Y


def Calibration(x,params):
	array = np.zeros(len(params))
	for i,param in enumerate(params):
		term = param * x ** i
		array[i] = term
		sum = np.sum(array)
	return sum
		

def PlotFitting(x,y,x_fit,y_fit):
	plt.plot(x,y,'o',color='blue',markersize=3,label='a')
	plt.plot(x_fit,y_fit,color='red',linewidth=1.0,linestyle='-')
	plt.xlim(cal_range)
	plt.ylim(0,2)
	plt.xlabel('baseline [V]',fontsize = 16)
	plt.ylabel('pulseheight [V]',fontsize = 16)
	plt.grid()
	plt.show()
	plt.cla()

def PlotCalibration(x1,y1,x2,y2):
	plt.plot(x1,y1,'o',color='tab:blue',markersize=0.7,label='a')
	plt.plot(x2,y2,'o',color='tab:red',markersize=0.7,label='a')
	plt.xlim(cal_range)
	plt.ylim(0,2)
	plt.xlabel('baseline [V]',fontsize = 16)
	plt.ylabel('pulseheight [V]',fontsize = 16)
	plt.grid()
	plt.show()
	plt.cla()


def SelectBase(minBase,maxBase,x,y):
	sel = []
	for i in range(len(x)):
		if minBase <= x[i] <= maxBase:
			sel.append(y[i])
	return sel


def main():
	
	set = gp.loadJson()
	ch,path = set["channel"],set["path"]
	os.chdir(path) 
	df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
	#df = df[(df['base']<0.1)&(df['rise']>4e-05)]
	#Pick Samples
	picked = gp.pickSamples("base","pulse_height_opt",df) 
	baseline = df.loc[picked,"base"]
	pulseheight = df.loc[picked,"pulse_height_opt"]
	print(f"Selected {len(picked)} Samples.")
	
	#Fitting
	popt,pcov = curve_fit(func,baseline,pulseheight,p0)
	x_fit = np.linspace(cal_range[0],cal_range[1],100000)
	fitted = func(x_fit,*tuple(popt))
	PlotFitting(baseline,pulseheight,x_fit,fitted)        

	#Calibration
	st = np.mean(pulseheight)
	pulseheight_cal = np.zeros(len(df))
	i = 0
	for base,height in zip(df["base"],df["pulse_height_opt"]):
		if base > cal_range[0] and base < cal_range[1]:
			pulseheight_cal[i] = height/Calibration(base,popt)
		else:
			pulseheight_cal[i] = height
		i+=1
	pulseheight_cal = pulseheight_cal*st
	df["height_opt_temp"] = pulseheight_cal

	PlotCalibration(df["base"],df["height_opt_temp"],df.loc[picked,"base"],df.loc[picked,"height_opt_temp"])

	df.to_csv(os.path.join(f'CH{ch}_pulse/output',"output.csv"))

if __name__ == '__main__':
	main()
