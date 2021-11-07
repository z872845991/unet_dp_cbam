import os
import shutil
import random

def move(file1, path1, target):
    global  no
    for file in file1:
        if "Annotation" in file:
            continue
        name = file.split('.')[0]
        label = name + "_Annotation.png"

        source_img = os.path.join(path1, file)
        source_label = os.path.join(path1, label)
        new_name = target + str(no) + ".png"
        new_label = target + str(no) + "_Annotation.png"

        shutil.move(source_img, new_name)
        shutil.move(source_label, new_label)

        no += 1

    print(no)

def fuse():
    path1 = "F:\\dataset\\JSFH-Cer\\all\\"
    path2 = "F:\\dataset\\JSFH-Cer\\cerv_3\\"
    target = "F:\\dataset\\JSFH-Cer\\new_all\\"

    file1 = os.listdir(path1)
    file2 = os.listdir(path2)


    move(file1,path1,target)
    move(file2,path2,target)


def split_train():
    path = "F:\\dataset\\JSFH-Cer\\123-all\\"
    files = os.listdir(path)
    target = "F:\\dataset\\JSFH-Cer\\train\\"

    nums = int(len(files)/2)
    train = int(nums*0.6)
    train_list = random.sample(range(1,nums),train)
    print(train_list)
    for num in train_list:
        target_img = target + str(num) + ".png"
        target_label = target + str(num) + "_Annotation.png"

        source_img = path + str(num) + ".png"
        source_label = path + str(num) + "_Annotation.png"

        shutil.move(source_img,target_img)
        shutil.move(source_label,target_label)

def split_val_test():
    path = "F:\\dataset\\JSFH-Cer\\123-all\\"
    files = os.listdir(path)
    target = "F:\\dataset\\JSFH-Cer\\val\\"
    nums = len(files)
    val = int(nums * 0.5)
    val_list = files[:val]
    for num in val_list:
        shutil.move(os.path.join(path,num),target)





no = 1

# fuse()
# split_train()
split_val_test()
