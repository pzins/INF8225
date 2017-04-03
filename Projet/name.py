import os
import sys
from django.http import request
from os import rename
import requests
import shutil

__author__ = 'Philip Masek'
from genderize import Genderize

def main():
    fileList = []
    fileSize = 0
    folderCount = 0
    rootdir = "lfw_funneled"
    maleFolder = "result/male"
    femaleFolder = "result/female"
    count = 0
    tmp = ""

    names = []
    genders = []
    for root, subFolders, files in os.walk("./"):
        folderCount += len(subFolders)
        for file in files:
            f = os.path.join(root,file)
            fileSize = fileSize + os.path.getsize(f)
            fileSplit = file.split("_")
            fileList.append(f)
            count += 1
            names.append(fileSplit[0])
    print("done")   
    begin = 0
    print(names)
    print(len(names))
    exit()

    for j in range(10):
        res = Genderize().get(names[begin:begin+10])
        begin = begin+10

        for i in res:
            if i['gender']=='female':
                genders.append(1)
            else:
                genders.append(0)
        print(j)  
    f = open('gen.txt','w')
    for i in genders:
        f.write("%d\n" % i)          
    print(genders)
            # if count == 1:
            #     result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
            #     print(type(result))
            #     result = result.json()
            #     tmp = fileSplit[0]
            # elif tmp != fileSplit[0]:
            #     result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
            #     result = result.json()
            #     tmp = fileSplit[0]
            # else:
            #     tmp = fileSplit[0]

            # try:
            #     if float(result['probability']) > 0.9:
            #         if result['gender'] == 'male':
            #             shutil.copyfile(f,"%s/%s" % (maleFolder,file))
            #         elif result['gender'] == 'female':
            #             shutil.copyfile(f,"%s/%s" % (femaleFolder,file))
            # except Exception as e:
            #     print(result['name'])

            # print(count)



if __name__ == "__main__":
    main()
