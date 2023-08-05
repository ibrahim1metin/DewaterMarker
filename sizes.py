import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

xsizes=[]
ysizes=[]
for file in os.listdir("memes"):
    img=Image.open(fr"memes\{file}")
    xsizes.append(img.size[0])
    ysizes.append(img.size[1])

fig,(ax0,ax1)=plt.subplots(1,2)
ax0.boxplot(xsizes, positions=[0], notch=True, patch_artist=True, boxprops=dict(facecolor="red"))
ax0.boxplot(ysizes, positions=[0.4], notch=True, patch_artist=True, boxprops=dict(facecolor="violet"))
ax0.text(0, min(xsizes) - 0.5, 'X', fontsize=12, color='white', ha='center', va='center', fontweight='bold')
ax0.text(.4, min(ysizes) - 0.5, 'Y', fontsize=12, color='white', ha='center', va='center', fontweight='bold')
ax0.set_xticks((0,0.4), ['X', 'Y'])
ax0.axis(xmin=-.2,xmax=.6)

ax1.violinplot([xsizes,ysizes])
ax1.text(0, min(xsizes) - 0.5, 'X', fontsize=12, color='white', ha='center', va='center', fontweight='bold')
ax1.text(.4, min(ysizes) - 0.5, 'Y', fontsize=12, color='white', ha='center', va='center', fontweight='bold')
ax1.set_xticks((1,2), ['X', 'Y'])
ax1.axis(xmin=0,xmax=3)
ax1.set_yticks(range(0,9000,500))
plt.show()

dataFrame=pd.DataFrame(zip(xsizes,ysizes),columns=["X","Y"])
print(dataFrame.describe())



mse=lambda x,y:(x-y)**2
mae=lambda x,y: abs(x-y)
maes=[]
mses=[]
for tu in list(zip(xsizes,ysizes)):
    maes.append(mae(tu[0],tu[1]))
    mses.append(mse(tu[0],tu[1]))
errorFrame=pd.DataFrame(zip(maes,mses),columns=["mae","mse"])
print(errorFrame.describe())

xborder=892+2*570
yborder=895+661*2

print(len(list(zip(xsizes,ysizes))))
filtre=lambda x: x[0]<xborder  and x[1]<yborder
filteredList=list(filter(filtre,zip(xsizes,ysizes)))
print(len(filteredList))

meanX=892
medianX=731
meanY=895
medianY=720
length=3124

distanceX={"mean":0,"median":0}
distanceY={"mean":0,"median":0}
for sizex,sizey in filteredList:
    distanceX['mean']+=abs(meanX-sizex)/length
    distanceX["median"]+=abs(medianX-sizex)/length
    distanceY['mean']+=abs(meanY-sizey)/length
    distanceY['median']+=abs(medianY-sizey)/length
print(distanceX)
print(distanceY)
