import tensorflow as tf
import numpy as np
import os,sys
from sklearn.metrics import confusion_matrix
from time import gmtime, strftime

sys.path.insert(0, os.getcwd())

from tflearn.data_utils import shuffle, to_categorical
import utils
from dataload.LSA16h import LSA16h
from dataload.PH import PH
from dataload.Dataset import Dataset
from dataload.BatchedDataset import BatchedDataset

import models.common as cm
import tflearn

class ExperimentResult:
  @staticmethod
  def mean_results(results):
      train_accuracy=np.mean([result.train_accuracy for result in results])
      test_accuracy=np.mean([result.test_accuracy for result in results])
      return ExperimentResult(train_accuracy,test_accuracy)

  def __init__(self,train_accuracy,test_accuracy):
    self.train_accuracy=train_accuracy
    self.test_accuracy=test_accuracy

def write_dict(d, filename):
  with open(filename, "a") as input_file:
    sorted_keys=sorted(d.keys())
    for k in sorted_keys:
      v=d[k]
      line = '{}={}'.format(k, v)
      print(line, file=input_file)



def get_experiment_folder(base_folder):
  def make_path(index):
    name= base_name + str(index).zfill(digits)
    return os.path.join(base_folder,name)
  digits=4
  base_name="exp"
  files = [f for f in os.listdir(base_folder) if f.startswith(base_name) and (not os.path.isfile(f))]
  files.sort()
  if files:
    last_experiment=files[-1]
    index=int(last_experiment[-digits:])
    return make_path(index+1)
  else:
    return make_path(0)

def save_results(experiment_folder):
  pass

DATASETS_FOLDER='../datasets/'

def get_lsadataset(dataset):
  input_folderpath=os.path.join(DATASETS_FOLDER, dataset)
  return LSA16h(dataset,input_folderpath).as_dataset()

def get_phdataset(dataset):
  input_folderpath=os.path.join(DATASETS_FOLDER, dataset)
  ph= PH(dataset,input_folderpath)
  d= ph.as_dataset()
  min_examples=8
  d.remove_classes_with_few_examples(min_examples)
  return d

def get_dataset(dataset):
  if dataset.startswith("ph"):
    return get_phdataset(dataset)
  elif dataset.startswith("lsa"):
    return get_lsadataset(dataset)
  else:
    raise ValueError('Unknown dataset %s' % dataset)

def evaluate(model,train_dataset,test_dataset,experiment_folder,parameters,classes):
  train_y= np.argmax(model.predict(train_dataset.x),axis=1)
  train_accuracy=np.mean(train_y==train_dataset.y)
  test_y= np.argmax(model.predict(test_dataset.x),axis=1)
  test_accuracy=np.mean(test_y==test_dataset.y)
  print("Train accuracy: %f" % train_accuracy)
  print("Test accuracy: %f" % test_accuracy)
  parameters["accuracy_train"]=train_accuracy
  parameters["accuracy_test"]=test_accuracy
  write_dict(parameters,os.path.join(experiment_folder,"results.txt"))
  test_confusion_matrix=confusion_matrix(test_dataset.y, test_y)
  utils.save_confusion_matrix(os.path.join(experiment_folder,"test_confusion.png"),test_confusion_matrix,range(classes))
  train_confusion_matrix=confusion_matrix(train_dataset.y, train_y)
  utils.save_confusion_matrix(os.path.join(experiment_folder,"train_confusion.png"),train_confusion_matrix,range(classes))
  # print("Saving model..")
  # model.save('trained_models/vgg16')
  return ExperimentResult(train_accuracy,test_accuracy)

def train(m,train_dataset,parameters,experiment_folder):
  regression = tflearn.regression(m.graph, optimizer='adam', loss='categorical_crossentropy',learning_rate=parameters['learning_rate'])
  checkpoint_path=os.path.join(experiment_folder, 'checkpoint')
  tensorboard_dir= os.path.join(experiment_folder, 'tensorboard')
  model = tflearn.DNN(regression)#, checkpoint_path=checkpoint_path, max_checkpoints=3, tensorboard_verbose=2,tensorboard_dir=tensorboard_dir)
  model.fit(train_dataset.x, train_dataset.y_one_hot, n_epoch=parameters['epochs'], validation_set=0.1, shuffle=True,show_metric=True, batch_size=parameters['batch_size'], snapshot_step=200,snapshot_epoch=False,  run_id='tflearn_snapshots/model_finetuning')
  return model

def save_weights(model,m,experiment_folder):
  for l in m.layers:
    variables=tflearn.get_layer_variables_by_name(l)
    for v in variables:
      w=model.get_weights(v)



def prepare_dataset(dataset):
  classes=max(dataset.y)+1
  dataset.x=list(map(lambda i: i.astype(float)/255.0,dataset.x))
  image_size=dataset.x[0].shape

  print("Dataset %s" % dataset.id)
  print("Image size: %s" % str(image_size))
  print("Classes: %d" % classes)
  print("Class labels: %s" % str(np.unique(dataset.y)))
  print("Samples in dataset: %d" % len(dataset.y))
  train_dataset,test_dataset=dataset.split_stratified(parameters['split'])
  print("Samples in train dataset: %d" % len(train_dataset.y))
  print("Samples in test dataset: %d" % len(test_dataset.y))
  # print(train_dataset.y)
  # print(classes)
  train_dataset.y_one_hot=to_categorical(train_dataset.y,classes)
  test_dataset.y_one_hot=to_categorical(test_dataset.y,classes)

  return (train_dataset,test_dataset,classes,image_size)

def experiment(dataset,parameters,model_generator):
  train_dataset,test_dataset,classes,image_size=prepare_dataset(dataset)

  input_size=np.prod(image_size)
  print("Preparing model...")
  with tf.Graph().as_default():
    #s =  tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[ None,image_size[0],image_size[1],image_size[2]])
    y = tf.placeholder(tf.int32, shape=[None])
    graph_model=model_generator(classes,x)

    #m = model.all_convolutional(classes,x)

    # print("Loading weights..")
    # model.load("/home/facuq/dev/models/vgg16.tflearn")
    parameters["model"]=graph_model.id

    experiments_folder='tmp/'
    experiment_folder=get_experiment_folder(experiments_folder)
    os.mkdir(experiment_folder)
    write_dict(parameters,os.path.join(experiment_folder,"parameters.txt"))
    print("Fitting model "+graph_model.id+"...")
    model=train(graph_model,train_dataset,parameters,experiment_folder)
    print("Finished training.")
    print("Evaluating...")
    experiment_result=evaluate(model,train_dataset,test_dataset,experiment_folder,parameters,classes)
    save_weights(model,graph_model,experiment_folder)
    return (experiment_result,graph_model)

def experiment_set(model,dataset,parameters,repetitions):
  print("Executing %d repetitions of experiment " % repetitions)
  results=[]
  for repetition in range(repetitions):
    print("\n *** Executing repetition %d / %d... *** " % (repetition+1,repetitions))
    result,graph_model=experiment(dataset,parameters,model)
    results.append(result)

  print("\n *** Repetitions finished (%d) *** " % repetitions)
  mean_results=ExperimentResult.mean_results(results)
  print("Mean training accuracy: %f" % mean_results.train_accuracy)
  print("Mean test accuracy: %f" % mean_results.test_accuracy)

  parameters["repetitions"]= repetitions
  parameters["train_accuracy"]= mean_results.train_accuracy
  parameters["test_accuracy"]= mean_results.test_accuracy
  parameters["model_id"] = graph_model.id
  write_dict(parameters,"tmp/results_%s.txt" % timestamp)


#print("Loading data...")
#dataset_id='lsa32x32/nr/rgb_black_background'
dataset_id='ph32'
dataset=get_dataset(dataset_id)
timestamp=strftime("%Y%m%d_%H:%M:%S", gmtime())
parameters={
            'batch_size':16,
            'epochs':2,
            'learning_rate':0.0007,
            'split':0.8,
            'timestamp':timestamp,
            'dataset':dataset.id,
            }
image_size=32
#model =cm.simple_feed
#model = cm.two_layers_feed
model = cm.conv_simple
#model = lambda classes,x: cm.all_convolutional(classes,x,image_size)
#model = cm.vgg16
#model = cm.resnet
#model = cm.inceptionv3
repetitions=1
# experiment_set(model,get_dataset('lsa32x32/original/rgb_resized'),parameters,repetitions)
# experiment_set(model,get_dataset('lsa32x32/original/rgb_black_background'),parameters,repetitions)
# experiment_set(model,get_dataset('lsa32x32/original/bw_contour'),parameters,repetitions)
# experiment_set(model,get_dataset('lsa32x32/original/gray'),parameters,repetitions)
# experiment_set(model,get_dataset('lsa32x32/original/bw'),parameters,repetitions)
experiment_set(model,dataset,parameters,repetitions)
