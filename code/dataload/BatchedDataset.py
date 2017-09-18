import numpy as np
import random

class BatchedIndices(object):
    def __init__(self,n,batch_size,shuffle=False):
        self.batch_size=batch_size
        self.n=n
        if (self.n<self.batch_size):
          raise ValueError("Batch size (%d) is greater than dataset size (%d)" % (self.batch_size,self.n))
        self.indices=range(self.n)

        self.shuffle=shuffle
        if self.shuffle:
          random.shuffle(self.indices)

        self.index=0

    def next_batch(self):
      start=self.index
      to=start+self.batch_size

      if to<self.n:
        batch_indices=list(range(start,to))
      else:
        extra=(to-self.n)
        batch_indices=list(range(start,self.n))
        batch_indices.extend(list(range(extra)))
      self.index= to % self.n
      return batch_indices

class BatchedDataset(object):
    def __init__(self,x,y,batch_size,shuffle=False):
        self.x=x
        self.y=y
        self.batched_indices=BatchedIndices(len(y),batch_size,shuffle)

    def next_batch(self):
        batch_indices=self.batched_indices.next_batch()
        batch_x=[self.x[i] for i in batch_indices]
        batch_y=[self.y[i] for i in batch_indices]
        return (batch_x,batch_y)

    def next_batch_numpy(self):
        batch_x,batch_y = self.next_batch()
        return (np.array(batch_x),np.array(batch_y))
