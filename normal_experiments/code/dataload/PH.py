
import numpy as np
import sys,os
import random
import math
from skimage import io
from dataload.Dataset import Dataset

class PH(object):
    # http://www-i6.informatik.rwth-aachen.de/~koller/1miohands/
    def __init__(self,id,folderpath):
        self.id=id
        self.folderpath=folderpath

        metadata_path = os.path.join(folderpath,"3359-ph2014-MS-handshape-annotations.txt")
        print("Loading ph dataset from %s" % metadata_path)
        with open(metadata_path) as f:
          lines = f.readlines()
        lines = [x.strip().split(" ") for x in lines]
        for l in lines:
          assert(len(l)==2)
        images_paths=[x[0] for x in lines]
        images_class_names=[x[1] for x in lines]

        self.classes= sorted(list(set(images_class_names)))
        self.y= np.array([self.classes.index(name) for name in images_class_names])
        # print(self.y)
        # print(self.classes)
        # print(images_paths)

        print("Reading images")
        paths = [os.path.join(folderpath,path) for path in images_paths]
        self.x=[]
        for filepath in paths:
          im=io.imread(filepath)
          if len(im.shape)==2:
              im=im[np.newaxis,:,:]
          self.x.append(im)
    def as_dataset(self):
        meta={"class_names":self.classes}
        return Dataset(self.id,self.x,self.y,meta)

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from BatchedDataset import BatchedDataset
    sys.path.insert(0, os.getcwd())
    #from matplotlib import pyplot as plt
    input_folderpath='../datasets/ph32'
    ph=PH("ph32",input_folderpath)
    d=ph.as_dataset()
    batch_size=11
    classes=max(d.y)+1
    assert(classes==45)
    image_size=d.x[0].shape
    print("Examples:"+str(len(d.x)))
    print("Image size"+str(d.x[0].shape))

    d.remove_classes_with_few_examples(5)
