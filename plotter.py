import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
plt.figure(figsize=(7.2, 4.8))
plt.subplot(1, 2, 1)
a1 = 0.9707
a2 = 0.9761
a3 = 0.9597
a4 = 0.9865

a = np.reshape(np.array([a1,a2,a3,a4]),[4,])
# a = np.mean(a,1)
# num_list = a
# name_list = ['Huber\nuntrainable\nBN','Huber\ntrainable\nBN','log-cosh\nuntrainable\nBN','log-cosh\ntrainable\nBN'
#              ,'quantile\nuntrainable\nBN','quantile\ntrainable\nBN']
#
# name_list = ['CNN_1','CNN_2','CNN_3','CNN_4'
#              ,'CNN_5','CNN_2\n Fine-tuning']
name_list = ['ViT_1','ViT_2','ViT_3','ViT_2\n Fine-tuning']
plt.barh(name_list,a,ec='r',ls='--')

for x, y in zip(name_list, a):
    plt.text(y-2.2e-2, x, round(y,4), ha='center', va='bottom')


plt.xlim(0.8,1)
plt.xlabel('TR')


plt.subplot(1, 2, 2)

b1 = 0.9067
b2 = 0.9135
b3 = 0.8631
b4 = 0.8853


b = np.reshape(np.array([b1,b2,b3,b4]),[4,])
# a = np.mean(a,1)
# num_list = a
# name_list = ['FMHS\nuntrainable\nBN','FMHS\ntrainable\nBN','FMVS\nuntrainable\nBN','FMVS\ntrainable\nBN'
#              ,'SM\nuntrainable\nBN','SM\ntrainable\nBN']

name_list = ['1','2','3','4']
plt.barh(name_list,b,color='green',ec='r',ls='--')

for x, y in zip(name_list, b):
    plt.text(y-2.2e-2, x, round(y,4), ha='center', va='bottom')

plt.yticks([])
plt.xlim(0.8,1)
plt.xlabel('Mean R2')
plt.show()
plt.show()

#%%
# 并列柱状图
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']#设置字体以便支持中文
import numpy as np
plt.figure(figsize=(9, 4.8))
plt.subplot(1, 2, 1)

x=np.arange(2)#柱状图在横坐标上的位置
#列出你要显示的数据，数据的列表长度与x长度相同
y1=[0.986,0.9793]
y2=[0.8891,0.8863]

bar_width=0.3#设置柱状图的宽度
# tick_label=['FMHS','FMVS','SM']
tick_label=['CNN_2','ViT_2']

#绘制并列柱状图
plt.bar(x,y1,bar_width,color='salmon',label='TR')

plt.bar(x+bar_width,y2,bar_width,color='orchid',label='Mean R2')
plt.ylim(0.8,1.02)
plt.legend()#显示图例，即label
plt.grid()
plt.xticks(x+bar_width/2,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置



plt.subplot(1, 2, 2)
plt.xlabel('R2')
plt.ylabel('Pass Rate')
plt.plot([0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50],
         [0.0493,0.5773,0.814,0.9193,0.9627,0.986,0.992,0.9973,0.9993,1.0], label='CNN_2')
plt.plot([0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50],
         [0.0573,0.5247,0.81,0.9227,0.9607,0.9793,0.988,0.9953,0.9986,1.0], label='ViT_2')
plt.grid()
# plt.ylim(0.8,1.02)
plt.legend(loc='lower left')
plt.show()