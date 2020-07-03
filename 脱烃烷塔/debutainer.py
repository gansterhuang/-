import  numpy as np

import matplotlib.pyplot as plt



filename = 'H:\我的文件夹\python_code\脱烃烷塔\debutanizer_data.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
line_list=[]
file=open(filename, 'r')
file_read=file.readlines()

line_number=0

for line in file_read:
    line_list.append(line)  # 整行读取数据
    line_number=line_number+1

#定义分割函数，将每一行的数切出来并转换成float形式
def transfer(input):
    temp=np.zeros((8,))
    for i in range(8):
        temp[i]=float(input[i*16:(i+1)*16])

    return temp

data=np.zeros((2394,8))
for i in range(2394):
    data[i]=transfer(line_list[i+5])

def data_generator():
    return data

plt.plot(range(2394),data_generator()[:,7])
plt.plot(range(2394),data_generator()[:,6])
plt.plot(range(2394),data_generator()[:,5])
plt.plot(range(2394),data_generator()[:,4])

plt.show()
print(np.shape(data_generator()))