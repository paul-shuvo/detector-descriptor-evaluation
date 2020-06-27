import matplotlib.pyplot as plt
import numpy as np
from random import*

str_t = r'''5/1/2019 35.1 55.9
5/2/2019 33.1 64
5/3/2019 37.9 71.1
5/4/2019 42.1 75.9
5/5/2019 46 77
5/6/2019 44.1 75.9
5/7/2019 46 64.9
5/8/2019 44.1 64.9
5/9/2019 42.1 62.1'''
date = []
low = []
high = []
for el in str_t.split('\n'):
    val = el.split(' ')
    date.append(val[0])
    low.append(float(val[1]))
    high.append(float(val[2]))
# print(str_t)

avg, sTemp = ([] for i in range(2))
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
i, avgsum, beg, end = 0, 0, 0, 9
sDate = ["June", "June", "July", "August", "September", "October", "November", "December", "January", "Febuary", "March", "April"]


for x, c in zip(low,high):
   if (x=='M'):
       low [i] = low [i-1]
   if (c =='M'):
       high [i] = high[i-1]
   avg.append((low[i]+high[i])/2)
   i+=1
sampMean = np.mean(avg)
sampVar = np.var(avg)
for x in range(12):
   i = randrange(beg,end)
   sTemp.append(avg[i])
   # if (x%3 ==0):
   #     beg+=19
   #     end+=19
   # else:
   #     beg+=18
   #     end+=18
sVar = np.var(sTemp)
sM = np.mean(sTemp)
# p=plt.hist(sTemp, bins = 10, density = True)
# plt.show()
fig, axs = plt.subplots(1, 1, figsize=(12, 5))
# axs[0].hist(sTemp, bins = 10, density = True)
axs.plot(sDate, sTemp)
axs.set_ylabel("Temp in Farenheit")
axs.set_xlabel("Date")
axs.set_xticks(sDate)
plt.show()

