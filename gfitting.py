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

#-------------- ガウスフィッティング ----------------
x_ax = "height_opt_temp"

set = gp.loadJson()
ch,path = set["Config"]["channel"],set["Config"]["path"]
os.chdir(path)
df = pd.read_csv((f'CH{ch}_pulse/output/output.csv'),index_col=0)
pulseheight = np.array(gp.extruct(df,x_ax))[0]
print(pulseheight)


energy = float(input('エネルギー(keV)を入れてください:'))

pulse_fmin = float(input('パルス高の最小値:'))
pulse_fmax = float(input('パルス高の最大値:'))
pulseheight_f = []
for i in range(len(pulseheight)):
	if pulse_fmin <= pulseheight[i]  <= pulse_fmax:
		pulseheight_f.append(pulseheight[i])

bin_n = 100		#ビンの数 （細かいヒストグラムにガウスフィッティングを描きたい場合は増やす）
hist, bin_edges = np.histogram(pulseheight_f, bins=bin_n)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2			#ビンの中心値を取得

#フィッティングの初期値
p0 = [0]*3
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
E = 2*sigma*np.sqrt(2*np.log(2))							#半値幅の計算

print ('半値幅:',E,'エネルギー分解能:',E/mu*100,'%')
dE = energy*E/mu

#-------------- グラフ描画--------------

plt.hist(pulseheight_f, bins =bin_n)
plt.title('dE= '+'{:.3f}'.format(dE)+'keV')
plt.plot(x2, fit, color='red',linewidth=1,linestyle='-',label='')
plt.xlabel('Pulseheight[keV]',fontsize = 14)
plt.ylabel('Count',fontsize = 14)
plt.savefig("FWHM_opt.png",format = 'png', dpi=150)
plt.show()
