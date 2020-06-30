import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# TODO: trim SMILES data files to have exactly 50% toxicity
# initialize from SMILES file
def init_with(training_file):
  qs=[]
  rs=[]
  max_len=0
  S='.'
  chars = set()
  vals = set()
  with open(training_file,'r') as f:
    for k,l in enumerate(f.readlines()) :
      #if k>4 : break
      l=l[:-1]
      l=l.replace(':',' ')
      l=l.replace('\t',' ')
      t,x=l.split(' ')
      x=int(x)

      max_len=max(max_len,len(t))
      qs.append(t)
      rs.append(x)
      for c in t : chars.add(c)
      vals.add(x)

  chars=[S]+sorted(chars)
  cdict=dict()
  for i,c in enumerate(chars) :
    cdict[c]=i
  for i,cs in enumerate(qs) :
    ds=[]
    for c in cs :
      ds.append(cdict[c])
    qs[i]=pad_with_int(0,max_len,ds)

  mid = round(len(qs) * 0.80)

  x_all = np.array(qs,dtype=np.int16)
  y_all = np.array(rs,np.bool)

  indices = np.random.permutation(x_all.shape[0])

  training_idx, test_idx = indices[:mid], indices[mid:]

  # split data into training and testing data
  x_train, x_test = x_all[training_idx,:], x_all[test_idx,:]
  y_train, y_test = y_all[training_idx], y_all[test_idx]

  x_train = tf.convert_to_tensor(x_train)
  x_test = tf.convert_to_tensor(x_test)

  y_train = tf.convert_to_tensor(y_train)
  y_test = tf.convert_to_tensor(y_test)

  return (x_train, y_train), (x_test, y_test)

# pad at the end, to make all max_length
def pad_with_int(S,max,xs) :
  l=len(xs)
  m=max-l
  return xs + ([S]*m)

# run everything
def run() :
  # get data
  train,test=init_with('data/smiles.txt')
  x_train,y_train = train
  x_test, y_test  = test

  # build model
  # room to refine and improve this baseline
  model = keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
  ])

  # compile/optimize it
  model.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

  # apply model to data and collect history
  history = model.fit(x_train,y_train,
                      epochs=100,
                      validation_data=(x_test,y_test),
                      validation_steps=5)

  model.summary()

  # visualize history of training and validation
  plot_graphs(history, 'accuracy')
  plot_graphs(history, 'loss')

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

# run it, inless imported as a module
if __name__=='__main__':
  run()
