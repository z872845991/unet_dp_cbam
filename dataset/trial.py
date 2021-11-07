import os
import re

# files = make_root('./data')
# print(files)
#
files =['000_HC.png', '000_HC_Annotation.png', '778_HC.png', '778_HC_Annotation.png','778_2HC.png', '778_2HC_Annotation.png','776_HC.png', '776_HC_Annotation.png']
new_files = []
new_labels = []
for file in files:
    if not re.search('\Snnotation', file):
        file_chip=file.split('.')[0]+'_Annotation.png'
        new_files.append((file,file_chip))

print(new_files)


"""
这是第一种思路
new_labels_1 = []
for file in new_labels:
    tmp = file.split('H')
    # if len(tmp)==3:
    #     tmp = tmp[0]+'_'+tmp[1]
    # else:
    #     tmp = tmp[0]+'_'+tmp[1]+'_'+tmp[2]
    new_labels_1.append(tmp)

# new_labels_2 = new_labels_1.sort()
print(new_labels_1)
# print(new_files)
"""
