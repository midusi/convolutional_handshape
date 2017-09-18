import numpy as np
import sys,os
import random
import math
from skimage import io

#from Dataset import Dataset
from dataload.Dataset import Dataset

class LSA16h(object):
    def __init__(self,id,folderpath):
        self.id=id
        self.folderpath=folderpath
        self.x=[]
        self.subjects=[]
        self.repetitions=[]
        self.rotations=[]
        self.y=[]
        for filename in sorted(os.listdir(folderpath)):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                (klass,subject,repetition,rotation)=self.info_from_filename(filename)
                self.subjects.append(subject)
                self.y.append(klass)
                self.repetitions.append(repetition)
                self.rotations.append(rotation)
                im=io.imread(os.path.join(folderpath,filename))
                if len(im.shape)==2:
                    im=im[np.newaxis,:,:]
                self.x.append(im)

    def info_from_filename(self,filename):
        name,ext=filename.split(".")
        parts=name.split("_")
        if len(parts)<3:
            raise ValueError("Invalid image name %s" % filename)
        klass=int(parts[0])-1
        subject=int(parts[1])
        repetition=int(parts[2])
        if len(parts)==4:
            rotation=int(parts[3])
        else:
            rotation=0
        return (klass,subject,repetition,rotation)
    def as_dataset(self):
        meta={'subjects':self.subjects,
              'rotations':self.rotations,
              'repetitions':self.repetitions}
        return Dataset(self.id,self.x,self.y,meta)


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from BatchedDataset import BatchedDataset
    sys.path.insert(0, os.getcwd())
    #from matplotlib import pyplot as plt
    input_folderpath='./datasets/lsa16h_mr/rgb_black_background/'
    lsa=LSA16h(input_folderpath)
    d=lsa.as_dataset()
    batch_size=11

    bd=BatchedDataset(d.x,d.y,batch_size)
    for i in range(100):
      xb,yb=bd.next_batch()
      #print(bd.batched_indices.index)
      #print(len(xb))
    # image_size=(64,64,3)
    # classes=16
    classes=max(d.y)+1
    image_size=d.x[0].shape

    train,test=d.split_stratified(0.5)
    print(len(d.x))
    print(d.x[0].shape)
