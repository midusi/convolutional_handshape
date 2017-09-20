import tensorflow as tf
import tflearn

def model(x):
 y=x
 y= tflearn.layers.core.fully_connected(y,32,activation='relu')
 #y= tflearn.batch_normalization(y)
 y= tflearn.layers.core.fully_connected(y,10,activation='softmax')

 return y
