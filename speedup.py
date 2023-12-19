import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
threads = [1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72]
times1 = [32.4139,22.3492,12.4596,7.30688,5.4687,4.62374,4.02916,3.95705,3.36906,3.30055,3.89056,3.61223,3.60353,3.61485,3.51522,3.54514,3.41112,3.37954,3.46492,4.75774]
times2 = [34.1437,22.303,12.3551,7.3812,5.50276,4.55978,4.04818,3.56039,3.84876,4.08984,3.29253,3.78863,3.59134,3.63438,3.73626,3.31294,3.42426,3.27519,3.52007,3.7335]
times = [(time1+time2)/2 for time1,time2 in zip(times1,times2)]
Speedup = [times[0]/time for time in times]
Efficiency = [speedup/thread for speedup,thread in zip(Speedup,threads)]

plt.plot(threads, Speedup, 'o-')
plt.title('Speedup')
plt.xlabel('Number of threads')
plt.ylabel('Sp')
ax = plt.gca()
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0, xmax=72)
plt.savefig('speedup.png')
plt.clf()
plt.plot(threads, Efficiency, 'o-')
plt.title('Efficiency')
plt.xlabel('Number of threads')
plt.ylabel('Sp/p')
ax = plt.gca()
ax.set_ylim(ymin=0,ymax=1)
ax.set_xlim(xmin=0,xmax=72)
plt.savefig('efficiency.png')

