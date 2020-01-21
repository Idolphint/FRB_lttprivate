deeplab3D使用方法

1. 创造数据集，假设图片在img下，需要这个路径下的img_list.txt那么就修改 packageimg.py(大概是这个名字)的路径，运行即可得到数据集
2. 修改deeplab3D/modeling/mydataset.py 只需要修改路径
3. `python train3d.py`