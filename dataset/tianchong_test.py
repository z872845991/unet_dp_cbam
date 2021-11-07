import matplotlib.image as Image
import numpy as np
# a = Image.imread('000_HC_Annotation.png')
# new_a = np.array(a)
all_index=[]
a = np.array([[1 ,2,0],[0,0,1],[1,0,1]])
# a = np.array([1 ,2,0])
print(a)

for i in range(3):
    # 取到每一行的数据
    b = a[i,:]
    # 找到不为0的下标，放在一个元祖里
    index = np.where(b!=0)
    new_index = []
    for j in range(len(index[0])):
        if j!=len(index[0])-1:
            if index[0][j]!=(index[0][j+1]-1):
                m = index[0][j]
                new_index.append(index[0][j])
                new_index.append(index[0][j+1])
        # else:
        #     if
    all_index.append(new_index)
all_index_new = np.array(all_index)
# tmp = np.array(all_index_new[2])
# print(tmp[1])
m = all_index_new.shape[0]
# print(all_index_new[0]!=[])


for n in range(m):
    if all_index_new[n] != []:
        tmp_index = np.array(all_index_new[2])
        left = tmp_index[0]
        right = tmp_index[1]
        mid = left
        for l in range(3):
            if mid<=right:
                a[n,mid]=1
                mid+=1
print(a)
