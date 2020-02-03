import numpy as np
import os
from scipy.io import loadmat
from scipy.io import savemat
#人之间应该分开

DEPTH=15
WIDTH = 384
HEIGHT = 384
def package(listname, path, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    f = open(os.path.join(path, listname)).readlines()
    a = np.zeros((DEPTH,WIDTH,HEIGHT))
    idx=0
    cnt = 0
    personID = 0
    for obj in f:
        obj = obj.strip("\n")
        personID_buf = obj.split("_")[3]
        if personID_buf != personID:
            if personID != 0:
                #a = a.transpose((1,2,0))
                np.save(os.path.join(outpath, "person_"+str(personID)+"_"+str(cnt)), a)
                idx = 0
                a = np.zeros((DEPTH,WIDTH,HEIGHT))
            personID = personID_buf
            print("new person!")
        mat = loadmat(os.path.join(path, obj))['data']
        mat = np.where(mat < 0.5, 0., 1.)
        a[idx] = mat
        idx += 1
        if idx==DEPTH:
            idx=0
            #a = a.transpose((1,2,0))
            np.save(os.path.join(outpath, "person_"+str(personID)+"_"+str(cnt)), a)
            a = np.zeros((DEPTH,WIDTH,HEIGHT))
            cnt+=1
            print("save ",outpath, personID, cnt)

def extract_fea(list_name):
    f = open(list_name).readlines()
    for obj in f:
        obj = obj.strip("\n")
        mat = loadmat(obj)['data']
        path, name = obj.split("/")
        savemat(os.path.join("./feature-ground/", name), {'data':mat[0]})
        savemat(os.path.join("./feature-arotia/", name), {'data':mat[1]})
        print("saved !", name)

if __name__=="__main__":
    package("img_list.txt" ,"/home/idolphint/lttWorkSpace/FRB/data20200111", "/home/idolphint/lttWorkSpace/FRB/5Ddata/label"+str(DEPTH))
    package("label_list.txt" ,"/home/idolphint/lttWorkSpace/FRB/data20200111", "/home/idolphint/lttWorkSpace/FRB/5Ddata/img"+str(DEPTH))
    #extract_fea("fea_list.txt")
    package("feature_ground_list.txt" ,"/home/idolphint/lttWorkSpace/FRB/data20200111", "/home/idolphint/lttWorkSpace/FRB/5Ddata/fea_ground"+str(DEPTH))
    package("feature_arotia_list.txt" ,"/home/idolphint/lttWorkSpace/FRB/data20200111", "/home/idolphint/lttWorkSpace/FRB/5Ddata/fea_arotia"+str(DEPTH))
