import shutil
import os

# 医院老师给的数据集中img和json文件是混合在一起的
# 本代码的功能是把json文件单独拿出来用批量脚本生成文件

path = 'F:\\dataset\\with_data_on\\'
target = 'F:\\dataset\\json\\'
files = os.listdir(path)
for file in files:
    if file.split('.')[-1]=='json':
        shutil.move(os.path.join(path,file),target)