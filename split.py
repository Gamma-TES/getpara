import matplotlib.pyplot as plt
import getpara as gp
import numpy as np
import pandas as pd

import os

para = "height"
n_block = 11

def func(x,a):
	return a*x




def main():
	print("start")

	setting = gp.loadJson()
	
	config = setting["Config"]
	os.chdir(config["path"])
	output_0 = f'CH{0}_pulse/output/{config["output"]}'
	output_1 = f'CH{1}_pulse/output/{config["output"]}'

	df_0 = pd.read_csv((f'{output_0}/output.csv'),index_col=0)
	df_1 = pd.read_csv((f'{output_1}/output.csv'),index_col=0)

	df_0 = gp.select_condition(df_0,setting)
	df_1 = gp.select_condition(df_1,setting)

	df_0_lap,df_1_lap = gp.overlap(df_0,df_1)

	length = len(df_0_lap)
	block = int(length/n_block)

	print(f"length: {length}")
	print(f"1-block: {block}")

	x,y = df_0_lap[para],df_1_lap[para]



	a = 4
	a_array = np.arange(a,-0.1,-0.0001)

	x_line = np.arange(0,1,0.001)
	
	a_line = []
	sel = []
	sel = set(sel)

	n = 0
	y_line_down = func(x_line,a)
	plt.plot(x_line,y_line_down,markersize=1,color="orange")
	for i in a_array:

		for index in df_0_lap.index:
			if (func(df_0_lap.at[index,para],i) < df_1_lap.at[index,para]) \
				and func(df_0_lap.at[index,para],a) > df_1_lap.at[index,para]:
				
					
				sel.add(int(index))

				if len(sel) == block:
					
					print(i)
					a_line.append(i)

					a = i
					y_line_down = func(x_line,a)

					np.savetxt(f"block_{n}.txt",[int(i) for i in list(sel)])
					n+=1
					if n == n_block-1:
						block = length%block+block
					
					

					sel = []
					sel = set(sel)
	
					plt.plot(x_line,y_line_down,markersize=1,color="orange")
		
			
	plt.scatter(x,y,s=2,alpha=0.7)
	plt.show()
	plt.cla()




	return 0

	

if __name__ == "__main__":
	main()
	print()







