#!/bin/bash
#author:idolphint
#use to genarate list of npy_file_dirctory;
#exp. ls img > img_list.txt

echo please input DEPTH
ls img$1 > img$1_list.txt
ls label$1 > label$1_list.txt
ls fea_arotia$1 > fea_arotia$1_list.txt
ls fea_ground$1 > fea_ground$1_list.txt

sed s#person#img$1/person#g img$1_list.txt > buf.txt
cat buf.txt > img$1_list.txt

sed s#person#label$1/person#g label$1_list.txt > buf.txt
cat buf.txt > label$1_list.txt

sed s#person#fea_arotia$1/person#g fea_arotia$1_list.txt > buf.txt
cat buf.txt > fea_arotia$1_list.txt

sed s#person#fea_ground$1/person#g fea_ground$1_list.txt > buf.txt
cat buf.txt > fea_ground$1_list.txt

echo finish bulid list !
