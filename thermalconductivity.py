import math
import glob
from sys import argv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import getpara as gp
from natsort import natsorted

R_n = 0.023
R_SH = 3.9e-3
PoR_n = 40.0
eta = 100
remove = [140,160,200]
p0 = [3e-9,190]

def func(x,G,T_c):
	n =3.5
	y = (G*T_c/n)*(1.0-(np.array(x))/T_c)**n
	return y

	

def main():
	path = "/Volumes/Extreme Pro/matsumi/data/20230616/room1-ch2"
	print(path)

	temps = natsorted(glob.glob(f'{path}/calibration/IV_*mK.txt'))

	R_set = R_n*PoR_n/100
	P_j_list = []
	T_bath_list = []

	for i in temps:
		data = np.loadtxt(i)
		I_bias = data[0]
		V_out = data[1]
		I_tes = eta * V_out
		I_sh = I_bias - I_tes
		V_tes = I_sh * R_SH
		R_tes = V_tes[1:] / I_tes[1:]
		R_tes =  np.append(0.0,R_tes)
		T_bath = int(gp.num(os.path.basename(i))[0])

		if T_bath in remove:
			continue
		r = np.argmax(R_tes >= R_set) # the most close R_tes to bias point 
	
		P_j = eta*V_out[r]*R_SH*(I_bias[r]-eta*V_out[r])*1e-12 #[pW]
		P_j_list.append(P_j)
		T_bath_list.append(T_bath*1e-3)

		plt.scatter(T_bath,P_j,c="tab:blue")

	popt,pcov = curve_fit(func,T_bath_list,P_j_list,p0=p0)
	print(popt)
	x_fit = np.arange(100,300,1)
	y_fit = func(x_fit,*popt)
	plt.plot(x_fit,y_fit)
	plt.show()



		

	
main()