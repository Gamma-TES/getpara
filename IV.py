# -*- coding: utf-8 -*-

import math
import glob
from sys import argv
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path

#-----------パラメタの設定----------------------

filename_1 = 'CH1_'				#ファイル名前半
filename_2 = 'mK_'				#ファイル名中間
filename_3 = 'uA.dat'			#ファイル名後半

R_sh = 3.9e-3					#シャント抵抗[Ω]

path = '../data/20230410/room2_ch2'

#----------ファイル名から温度とバイアス電流値を得る---------

os.chdir(path) 

folder = glob.glob("*")
print(folder)




def main():
	T = []
	Ib = []
	for file in glob.glob('*.dat'):					#datファイルを読み込む
		file_i = file.replace(filename_1, '')		#CH1_ → ''
		file_i = file_i.replace(filename_2, ',')	#mK_ → ','
		file_i = file_i.replace(filename_3, '')		#uA.dat → ''
		T_I = file_i.split(',')						#T,I → [T,I]
		T.append(int(T_I[0]))
		Ib.append(int(T_I[1]))

	Ib = np.array(Ib)			#Tを配列化
	Ib.sort()					#Tを昇べきの順に並べる

	print ('T(mK)=',T[0])
	print ('Ib(uA)=',Ib)
	print ('len_Ib=',len(Ib))

	#---------SQUIDアンプの出力電圧の補正を行う----------------------
	Vout = []
	Vout_1 = []
	Vout_2 = []

	for i in range(len(Ib)):
		filename = filename_1 + str("{0:g}".format(T[0])) + filename_2 + str(Ib[i]) + filename_3
		#print(filename)
		data = np.loadtxt(filename,comments = '#',delimiter = ',')
		Vout_1.append(np.mean(data))

	#print(Vout_1)

	imax = Vout_1.index(max(Vout_1))
	imin = Vout_1.index(min(Vout_1))

	#print(imax)
	#print(imin)

	if (max(Vout_1) - min(Vout_1))/(imax - imin) > 0:
		for i in range(len(Ib)):
			if i == 0:
				Vout.append(0)
			else:
				Vout.append(Vout_1[i]-Vout_1[0])
	else:
		for i in range(len(Ib)):
			if i == 0:
				Vout.append(0)
			else:
				Vout_2 = -np.array(Vout_1)
				Vout.append(Vout_2[i]-Vout_2[0])

	#--------SQUIDアンプの出力電圧とTESに流れる電流の比（eta）を求める---------
	if imax > imin :
		Ib_s = np.array(Ib[0:5])      #etaが極端に大きい場合、imaxを超電導状態における傾きが最大になるデータ番号に書き換える
		Vout_s = np.array(Vout[0:5]) #上に同じ
	else:
		Ib_s = np.array(Ib[0:5])     #etaが極端に大きい場合、iminを超電導状態における傾きが最大になるデータ番号に書き換える
		Vout_s = np.array(Vout[0:5]) #上に同じ

	#print(len(Ib_s))
	#print(len(Vout_s))

	def fit_func(para,x,y):					#fit関数の定義。paraはパラメタ列。
		a = para[0]							#パラメタ列の1番目をa
		b = para[1]							#パラメタ列の2番目をb
		residual = y-(a*x + b)
		return(residual)

	para0 = [0.0,0.0]
	result = opt.leastsq(fit_func,para0,args = (Ib_s,Vout_s))
	a_fit = result[0][0]
	b_fit = result[0][1]
	print ('fit_func =',a_fit,'X +',b_fit)

	eta = 1/a_fit
	print ('eta(uA/V) =',eta)

	#-------------------------抵抗値等の計算-----------------------
	I_TES = eta*np.array(Vout)

	print(I_TES)
	num1 = len(I_TES)
	#print(num1)

	I_sh = Ib - I_TES
	V_TES = I_sh*R_sh

	I_TES0=list(I_TES)
	V_TES0=list(V_TES)

	I_TES0 = np.array(I_TES[1:num1])
	V_TES0 = np.array(V_TES[1:num1])

	R_TES0 = V_TES0/I_TES0
	num2 = len(R_TES0)
	#print(num2)
	R_TES=list(R_TES0)
	R_TES.insert(0,0.0)
	#---------------------ファイルに出力-------------------
	folder = os.path.exists('./output')
	if not folder:
		os.mkdir('output')
	os.chdir('./output')

	f = open('IV_'+str(T[0])+'mK'+'.txt','w')
	f.write('#Ib[uA]'+'\t'+'Vout[V]'+'\t'+'R_TES[ohm]'+'\t'+'eta[uA/V]'+'\n')
	for i in range(len(Ib)):
		f.write(str(Ib[i])+'\t'+ str(Vout[i])+'\t'+ str(R_TES[i])+'\t'+ str(eta)+'\n')
	f.flush()
	f.close()

	#------------------グラフとして出力-----------------------
	#plt.figure(figsize=(8,5))#
	plt.title('I-V at '+str(T[0])+' [mK]')
	plt.plot(Ib,Vout,marker='o',color ='red',linewidth=1,markersize = 2)
	plt.xlabel('Ib[uA]',fontsize = 16)
	plt.ylabel('Vout[V]',fontsize = 16)
	plt.grid(True)
	#plt.legend(loc ='best',fancybox = True,shadow = True)
	plt.savefig("IV.png",format = 'png')
	plt.show()

	plt.title('I-R at '+str(T[0])+' [mK]')
	plt.plot(Ib,R_TES,marker='o',color ='red',linewidth=1,markersize = 2)
	plt.xlabel('Ib[uA]',fontsize = 16)
	plt.ylabel('R_TES[$\Omega$]',fontsize = 16)
	plt.grid(True)
	#plt.legend(loc ='best',fancybox = True,shadow = True)
	plt.savefig("IR.png",format = 'png')
	plt.show()

	os.chdir('..')

