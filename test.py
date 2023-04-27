import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import libs.getpara as gp

set = gp.loadJson()
rate,samples,ch = set["rate"],set["samples"],set["channel"]
os.chdir(set["path"])

print(set["path"])
time = gp.data_time(rate,samples)
data = np.loadtxt(f"CH{ch}_pulse/output/average_pulse.txt")

plt.plot(time,data)
plt.title("average pulse")
plt.xlabel("Time (s)")
plt.ylabel("Volt (V)")
plt.savefig(f'CH0_pulse/output/average_pulse.png')
plt.show()

