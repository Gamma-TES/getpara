# -*- coding: utf-8 -*-
#Calibration
# 11-6D1

import glob
import numpy as np
import scipy.optimize as opt
import scipy.fftpack as fft
from array import array
import os
import os.path
import time
import matplotlib.pyplot as plt
import libs.getpara as gp
import pandas as pd
from scipy.optimize import curve_fit

#----------- パルスハイトの読み込み ----------------

set = gp.loadJson()
os.chdir(set["Config"]['path'])
df = pd.read_csv((f'CH{set["Config"]["channel"]}_pulse/output/output.csv'),index_col=0)
rate,samples,presamples,threshold,ch = set["Config"]["rate"],set["Config"]["samples"],set["Config"]["presamples"],set["Config"]["threshold"],set["Config"]["channel"]

pulseheight = df["height_opt_temp"]

ap = len(pulseheight)


plt.hist(pulseheight, bins =1500, range=(70,110))
plt.xlabel('Pulseheight[V]')
plt.ylabel('Count[-]')
plt.show()


	#------------ フィッティング ---------------
pe = int(input('ピークはいくつありますか？:'))
#フィッティングするガウス関数の設定
def gauss(p,x,y):
	residual=y-(p[0]*np.exp(-(x-p[2])**2/(2*p[1]**2)))
	return residual

ch = [0.0]
energy = [0.0]
for i in range(pe):
	print ('peak',i+1)
	pulse_fmin1 = float(input('パルス高の最小値[V]:'))
	pulse_fmax1 = float(input('パルス高の最大値[V]:'))
	pulseheight_f = []
	for i in range(ap):
		if pulse_fmin1 <= pulseheight[i] <= pulse_fmax1:
			pulseheight_f.append(pulseheight[i])

	hist, bin_edges = np.histogram(pulseheight_f, bins=40)
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

	p0=[np.max(hist),(pulse_fmax1 - pulse_fmin1)/6,bin_centers[np.argmax(hist)]]
	print ('初期値',p0)

	output=opt.leastsq(gauss,p0,args=(bin_centers,hist))
	a = output[0][0]
	sigma = output[0][1]
	mu = output[0][2]
	ch.append(mu)
	x = np.arange(pulse_fmin1,pulse_fmax1,0.00001)
	y = a*np.exp(-(x-mu)**2/(2*sigma**2))
	E = 2*sigma*np.sqrt(2*np.log(2))

	print ('FWHM :',E,'Energy resolution ',E/mu*100,'%')

	plt.hist(pulseheight_f, bins =50)
	plt.plot(x, y, color='red',linewidth=1,linestyle='-')
	plt.xlabel('pulseheight')
	plt.ylabel('count')
	plt.show()

	E = float(input('このピークのエネルギーはいくつですか？[keV]:'))
	energy.append(E)

ch = np.array(ch)
energy = np.array(energy)

#----------------- 2次曲線でfitting --------------------

p0 = [0.2,0.0,0.0]
def fit(p,x,y):
	residual=y-(p[0]*x**2+p[1]*x+p[2])
	return residual

output=opt.leastsq(fit,p0,args=(ch,energy))
a = output[0][0]
b = output[0][1]
c = output[0][2]
print ('a=',a,'b=',b,'c=',c)
x = np.arange(0,100,0.01)
y = a*x**2+b*x+c



'''
#直線で校正
a = (energy[1]-energy[0])/(ch[1]-ch[0])
x = np.arange(0,18,0.01)
y = a*x
'''


#校正曲線の表示
plt.xlim(0,700)#ここの単位はkeV
plt.ylim(0,10.0)#ここの単位はV（パルス高）
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Pulseheight[V]',fontsize=14, fontname='serif')
plt.xlabel('Energy[keV]',fontsize=14, fontname='serif')
plt.plot(energy,ch,'ro',color='blue',markersize=6)
plt.plot(y,x,color='red',linewidth=1,linestyle='-')
#plt.grid()
plt.savefig("cal_curve.pdf",format = 'pdf')
plt.show()



#---------- キャリブレーション後のヒストグラム表示 ---------
#pulseheight_fc = a*pulseheight**2+b*pulseheight+c
pulseheight_fc = a*pulseheight

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,1500)
#plt.ylim(0,450)
plt.hist(pulseheight_fc, bins =1500, range=(0,1500))
plt.xlabel('Energy[keV]',fontsize=14, fontname='serif')
plt.ylabel('Count[-]',fontsize=14, fontname='serif')
#plt.grid()
plt.savefig("spectrum_cal_last.pdf",format = 'pdf')
plt.show()


#------------- ファイルに出力 ---------------
f = open("calibration_curve_last.txt","w")			#校正曲線の出力
f.write(f"{a}\n{b}\n{c}")                           #+str(b)+'\n'+str(c))
f.flush()
f.close()

f = open("pulseheight_cal_last.txt","w")				#キャリブレーション後のpulseheightの出力
for x in range(len(pulseheight_fc)):
	f.write(str(pulseheight_fc[x])+'\n')
f.flush()
f.close()
#os.chdir("..")									#1つ上のディレクトリに移動

#--------- キャリブレーション後のデータをフィッティング ------------
pulse_fmin = float(input('ピークパルスの最小値[keV]::'))
pulse_fmax = float(input('ピークパルスの最大値[keV]::'))
pulseheight_f = []
for i in range(ap):
	if pulse_fmin <= pulseheight_fc[i] <= pulse_fmax:
		pulseheight_f.append(pulseheight_fc[i])

bin_n = 150
hist, bin_edges = np.histogram(pulseheight_f, bins=bin_n)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2			#ビンの中心値を取得

#フィッティングの初期値
p0 = [0,0,0]
p0[0]= np.max(hist)										#hist内の最大値を返す
p0[1] = (pulse_fmax - pulse_fmin)/6
p0[2] = bin_centers[np.argmax(hist)]					#argmaxは最大値の添字を返す
print ('初期値',p0)

#フィッティング関数の定義
def gauss(p,x,y):
	residual=y-(p[0]*np.exp(-(x-p[2])**2/(2*p[1]**2)))
	return residual

output=opt.leastsq(gauss,p0,args=(bin_centers,hist))
a = output[0][0]
sigma = output[0][1]
mu = output[0][2]

x2 = np.arange(pulse_fmin,pulse_fmax,0.00001)
fit = a*np.exp(-(x2-mu)**2/(2*sigma**2))
dE = 2*sigma*np.sqrt(2*np.log(2))							#半値幅の計算

print ('energy resolution :',dE,'keV',E/mu*100,'%')

#フィッティング結果の表示
plt.xlim(1330,1350) #ここの単位はkeV
#plt.ylim(0,400) #ここの単位はカウント
plt.hist(pulseheight_f, bins =bin_n)
plt.plot(x2, fit, color='red',linewidth=1,linestyle='-',label='')
plt.title('dE= '+'{:.2f}'.format(dE)+'keV') #'{}:.3f}'は少数部分を3桁に指定する意味
plt.xlabel('pulseheight')
plt.ylabel('count')
plt.show()
