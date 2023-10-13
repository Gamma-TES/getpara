import matplotlib.pyplot as plt
import getpara as gp
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import os
import tqdm
import glob

# parameter 
para = "height" 
n_block = 11
a_ini = 1.5 # y = ax + 1
resolution = 0.00001 # a decrement

cmap = cm.get_cmap("hsv")

def func(x,a):
	return a*x




def main():

	setting = gp.loadJson()
	
	config = setting["Config"]
	select = setting["select"]
	os.chdir(config["path"])

	ch = config["channel"]

	output_0 = f'CH{0}_pulse/output/{config["output"]}'
	output_1 = f'CH{1}_pulse/output/{config["output"]}'

	os.makedirs(f"{output_0}/block",exist_ok=True)

	df_0 = pd.read_csv((f'{output_0}/output.csv'),index_col=0)
	df_1 = pd.read_csv((f'{output_1}/output.csv'),index_col=0)
	
	df_0 = gp.select_condition(df_0,select=select)
	df_1 = gp.select_condition(df_1,select=select)

	df_0_lap,df_1_lap = gp.overlap(df_0,df_1)

	length = len(df_0_lap)
	block = int(length/n_block)

	print(f"length: {length}")
	print(f"1-block: {block}")


	x,y = df_0_lap[para],df_1_lap[para]


	a = a_ini
	
	a_array = np.arange(a,-0.1,resolution*-1)

	x_line = np.arange(0,1,0.001)
	

	a_line = []
	sel = []
	sel = set(sel)

	n = 0
	y_line_down = func(x_line,a)
	plt.scatter(x,y,s=2,color = cmap((float(n))/float(n_block)))
	plt.plot(x_line,y_line_down,"--",linewidth=1,markersize=1,color="orange",alpha=0.7)
	plt.show()
	plt.cla()

	plt.plot(x_line,y_line_down,"--",linewidth=1,markersize=1,color="orange",alpha=0.7)
	for i in tqdm.tqdm(a_array):

		for index in df_0_lap.index:

			# check up or down
			if (func(df_0_lap.at[index,para],i) < df_1_lap.at[index,para]) \
				and func(df_0_lap.at[index,para],a) > df_1_lap.at[index,para]:

				sel.add(int(index))

				if len(sel) == block:
					a_line.append(i)

					a = i
					y_line_down = func(x_line,a)

					# cast set to int list
					indexs = [int(float(i)) for i in list(sel)]
					# save text
					np.savetxt(f"CH{ch}_pulse/output/{config['output']}/block/block_{n}.txt",indexs)

					# block increment
					n+=1
					if n == n_block-1:
						block = length%block+block
					
					# initialize sel
					sel = []
					sel = set(sel)

					# plot block
					df_sel_0,df_sel_1 = df_0_lap.loc[indexs],df_1_lap.loc[indexs]
					x,y = df_sel_0[para],df_sel_1[para]

					plt.scatter(x,y,s=2,color = cmap((float(n))/float(n_block)))
					plt.plot(x_line,y_line_down,"--",linewidth=1,markersize=0.7,color="orange",alpha=0.7)
		if len(a_line) == n_block:
			break
	
	print(a_line)
			
	#plt.scatter(x,y,s=2,alpha=0.7)
	plt.grid()
	plt.xlabel(f"Pulse Height (CH0) [V]")
	plt.ylabel(f"Pulse Height (CH1) [V]")
	plt.savefig(f"CH{ch}_pulse/output/{config['output']}/block/blocks.png")
	plt.show()
	
	plt.cla()




	return 0

	

if __name__ == "__main__":
	main()








