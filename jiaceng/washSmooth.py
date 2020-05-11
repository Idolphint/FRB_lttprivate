# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:50:02 2020

@author: litia
"""

import csv
import os
import pandas as pd
path = 'E:/FRB/jiaceng/'
INFILE = "dicomJiaCeng.csv"
OUTFILE = "dicomSmooth.csv"
WASHLEN = 15
def washCsv(name):
    mat_name= pd.read_csv(filepath_or_buffer = name, 
                          sep = ',')["fileName"].values
                        
    hasJiaCeng= pd.read_csv(filepath_or_buffer = name,
                          sep = ',')["hasJiaCeng"].values
    cnt = 0
    newDF = []
    res = 0
    for i in range(len(hasJiaCeng) - WASHLEN):
        res = 0
        for j in range(WASHLEN):
            if hasJiaCeng[i+j]:
                res+=1
        res = int((2.* res+1)/(WASHLEN+1))
        cnt+=res
        print(res)
        newDF.append({"fileName": mat_name[i], "hasJC": res})
    for i in range(len(hasJiaCeng) - WASHLEN -1, len(hasJiaCeng)):
        newDF.append({"fileName": mat_name[i], "hasJC": res})
    pd.DataFrame(newDF).to_csv(OUTFILE)
    print(cnt / len(hasJiaCeng))

if __name__ == "__main__":
    washCsv(INFILE)    