from PIL import Image
import numpy as np
import os

# 把头围分割的label填充

def tianchong():
    img = Image.open('H:\\python_workspace\\pytorch\\U_net\\simple_u_net_v2\\test\\000_HC_Annotation.png')
    new_img = np.array(img)
    all_index=[]
    # a = np.array([[1 ,2,0],[0,0,1],[1,0,1]])
    # a = np.array([1 ,2,0])
    # print(a)

    for i in range(new_img.shape[0]):
        # 取到每一行的数据
        b = new_img[i,:]
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
    # print(all_index_new[100])
    m = all_index_new.shape[0]
    # print(all_index_new[0]!=[])


    for n in range(m):
        if all_index_new[n] != []:
            tmp_index = np.array(all_index_new[n])
            # print(tmp_index)
            left = tmp_index[0]
            right = tmp_index[1]
            mid = left
            for l in range(new_img.shape[0]):
                if mid<=right:
                    new_img[n,mid]=1
                    mid+=1
    # print(a)
    final_img = Image.fromarray(np.uint8(new_img*255))
    final_img.save('new.png')


if __name__ == '__main__':
    path = os.listdir()