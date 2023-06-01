
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

cal_range = [1.0,1.6]






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
	plt.xlabel('baseline [V]',fontsize = 16)
	plt.ylabel('pulseheight [V]',fontsize = 16)
	plt.grid()
	plt.show()
	plt.cla()

def PlotCalibration(x1,y1,x2,y2):
	plt.plot(x1,y1,'o',color='tab:blue',markersize=0.7,label='a')
	plt.plot(x2,y2,'o',color='tab:red',markersize=0.7,label='a')
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
	ch,path = set["Config"]["channel"],set["Config"]["path"]
	os.chdir(path) 
	df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
	x,y = gp.extruct(df,"base","height_opt")


	df_sel = gp.select_condition(df,set)
	print(df_sel)
	picked = gp.pickSamples(df_sel,"base","height_opt") 
	baseline = df_sel.loc[picked,"base"]
	pulseheight = df_sel.loc[picked,"height_opt"]
	print(f"Selected {len(picked)} Samples.")
	
	#Fitting
	popt,pcov = curve_fit(func,baseline,pulseheight,p0)
	x_fit = np.linspace(set['select']['base->'],set['select']['base-<'],100000)
	fitted = func(x_fit,*tuple(popt))
	PlotFitting(baseline,pulseheight,x_fit,fitted)        

	#Calibration
	st = np.mean(pulseheight)
	pulseheight_cal = np.zeros(len(df_sel))


	for index,row in df_sel.iterrows():
		df.at[index,"height_opt_temp"] = row['height_opt']/Calibration(row['base'],popt)*st

	PlotCalibration(df["base"],df["height_opt_temp"],df.loc[picked,"base"],df.loc[picked,"height_opt_temp"])
	np.savetxt(f'CH{ch}_pulse/output/select/pulseheight_opt_temp.txt',pulseheight_cal)
	df.to_csv(f'CH{ch}_pulse/output/output.csv')

if __name__ == '__main__':
	main()
