import os
import shutil

# 把备份的训练数据中的label抽出来后续处理
"""
path = "F:\\dataset\\2019_seg\\"
files = os.listdir(path)
target = "F:\\dataset\\old_label\\"

for file in files:
    if "Annotation" in file:
        shutil.move(os.path.join(path,file),os.path.join(target,file))


"""


# 把标注文件生成的单个文件夹中的img和label统一放到一个文件夹中，并重新命名

path = "F:\\dataset\\2019_seg\\"
target = "F:\\dataset\\old_label\\"

files = os.listdir(path)

num=0
# 3中的文件
for file in files:
    # F:\\dataset\\3\\文件夹
    inpath = os.path.join(path, file)
    # 如果是文件夹，就把文件夹中的文件名改掉
    if os.path.isdir(inpath):
        infiles = os.listdir(inpath)
        num+=1
        for item in infiles:
            if item=='img.png':
                os.rename(os.path.join(inpath,item),os.path.join(inpath,str(num))+'.png')
                shutil.move(os.path.join(inpath,str(num))+'.png',target)
            if item == 'label.png':
                os.rename(os.path.join(inpath, item), os.path.join(inpath, str(num)) + '_Annotation.png')
                shutil.move(os.path.join(inpath, str(num)) + '_Annotation.png', target)

