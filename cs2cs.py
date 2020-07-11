# -*- coding: utf-8 -*-

# derived from keras, addition example at:
# https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

'''
# A generic implementation of seq2seq one char at a time.

Use cases:

DATA is of the FORM:

theorem    : proof term
0A00AB00BCC:1A1B1C0C0BA
00AB0A00BCC:1A1B1C0C0AB
or
0AAB0A00BCC:?

encoding: 0 on the left = -o (lollipop)
          0 on the right = application
          1 on the right: lambda
          A,B,C,... variables, on both sides
          ? non-theorem: failure to find a proof term

'''

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
np.random.seed(0)

# All the symbols, .' for padding

# reads a theorem - proof pair for implicational linear logic
# or a tree-based natural number and ist successor, as  a pair

def init_with(cfg) :
  training_file = cfg['TRAINING_FILE']
  qs=[]
  rs=[]
  max_len=0
  S='.'
  chars = set()
  with open(training_file,'r') as f:
    for l in f.readlines() :
      l=l[:-1]
      t,x=l.split(':')

      max_len=max(max_len,len(t))
      qs.append(t)
      rs.append(x)
      for c in t : chars.add(c)
      for c in x : chars.add(c)

  chars=[S]+sorted(chars)
  #print(max_len, chars)
  for i,x in enumerate(qs) :
    qs[i]=pad_to_str(S,max_len,x)
  for i,x in enumerate(rs) :
    rs[i]=pad_to_str(S,max_len,x)
    #print(rs[i])

  return qs,rs,chars,max_len

def pad_to_str(S,max,xs) :
  l=len(xs)
  m=max-l
  return xs + (S*m)



class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)



def vectorize(ctable,questions,expected,chars,MAXLEN) :
  print('Vectorization...', (len(questions), MAXLEN, len(chars)))
  x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
  y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)

  for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
  for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, MAXLEN)

  # Shuffle (x, y) in unison
  indices = np.arange(len(y))
  np.random.seed(0)
  np.random.shuffle(indices)
  # ensure same shuffling for all runs
  print(indices[0:10])
  # for cats() only
  #assert indices[0:10].tolist() == \
  #  [55458,26836,34921,41001,42101,53975,10906,49837,21398,63510]

  #[162387 149843  60785 154195 129744 173381 118270  31251 149870  89622]
  x = x[indices]
  y = y[indices]

  # Explicitly set apart 10% for validation data that we never train over.
  split_at = len(x) - len(x) // 5
  (x_train, x_val_test) = x[:split_at], x[split_at:]
  (y_train, y_val_test) = y[:split_at], y[split_at:]
  split_at = len(x_val_test) - len(x_val_test) // 2
  (x_val,x_test) = x_val_test[:split_at], x_val_test[split_at:]
  (y_val, y_test) = y_val_test[:split_at], y_val_test[split_at:]

  print('Training Data:')
  print(x_train.shape)
  print(y_train.shape)

  print('Validation Data:')
  print(x_val.shape)
  print(y_val.shape)

  print('Test Data:')
  print(x_test.shape)
  print(y_test.shape)

  return x_train,y_train,x_val,y_val,x_test,y_test

def build_model(cfg,chars,MAXLEN) :
  print('Build model...')
  model = Sequential()
  # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
  # Note: In a situation where your input sequences have a variable length,
  # use input_shape=(None, num_feature).
  model.add(cfg['RNN'](cfg['HIDDEN_SIZE'], input_shape=(MAXLEN, len(chars))))
  # As the decoder RNN's input, repeatedly provide with the last output of
  # RNN for each time step. Repeat MAXLEN times as that's the maximum
  # length of output
  model.add(layers.RepeatVector(MAXLEN))
  # The decoder cfg['RNN'] could be multiple layers stacked or a single layer.
  for _ in range(cfg['LAYERS']):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(cfg['RNN'](cfg['HIDDEN_SIZE'], return_sequences=True))

  # Apply a dense layer to the every temporal slice of an input. For each of step
  # of the output sequence, decide which character should be chosen.
  model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  model.summary()
  return model

# Train the model each generation and show predictions against the validation
# dataset.

def ITERATIONS(cfg) :
  return min(100, cfg['TRAINING_SIZE'] // cfg['BATCH_SIZE'])

def test_with(cfg,ctable,model,x_test,y_test) :
  # Select GUESSES samples from the validation set at random so we can visualize
  # errors.
  print('TESTING')
  good = 0
  guesses=cfg['GUESSES']
  for i in range(guesses):
    ind = np.random.randint(0, len(x_test))
    rowx, rowy = x_test[np.array([ind])], y_test[np.array([ind])]
    preds = model.predict_classes(rowx, verbose=0)
    q = ctable.decode(rowx[0])
    q=q.replace('.','')
    correct = ctable.decode(rowy[0], calc_argmax=True)
    correct=correct.replace('.','')
    guess = ctable.decode(preds[0], calc_argmax=False)
    guess=guess.replace('.','')
    if correct == guess:
      good+=1
      print('+', end=' ')
    else:
      print('-', end=' ')
    print('Q =', q, end=' ')
    print('T =', correct, end=' ')
    print('A =',guess)
  print('Evaluating model.')
  eval_results = model.evaluate(x_test,
                                y_test
                               )
  print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(*eval_results))
  print('good:',good,'/',guesses,'=',good/guesses)
  print('-'*40,'\n')


def learn(cfg,ctable,model,x_train, y_train,x_val, y_val) :
  its=ITERATIONS(cfg)
  for iteration in range(0, its//100):
    print('ITERATIONS:',its)
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history=model.fit(x_train, y_train,
              batch_size=cfg['BATCH_SIZE'],
              epochs=100,
              validation_data=(x_val, y_val))
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    test_with(cfg,ctable,model,x_val,y_val)
  model.save(cfg['MODEL_FILE']) #, save_format='tf')


def run_with(cfg,test_only=True) :
  questions, expected, chars, MAXLEN = init_with(cfg)
  ctable = CharacterTable(chars)
  x_train, y_train, x_val, y_val, x_test, y_test = \
    vectorize(ctable,questions,expected,chars,MAXLEN)
  if test_only :
    model_file=cfg['MODEL_FILE']
    model = keras.models.load_model(model_file)
    test_with(cfg,ctable,model,x_test,y_test)
  else :
    model=build_model(cfg,chars,MAXLEN)
    learn(cfg,ctable, model, x_train, y_train, x_val, y_val)


def tlin(test_only=False) :
  cfg = dict(
    TRAINING_FILE='data/tlin.txt',
    MODEL_FILE='models/tlin_cs2cs',
    # Parameters for the model and dataset.
    TRAINING_SIZE=2 ** 18,
    # Try replacing LSTM, GRU, or SimpleRNN.
    RNN=layers.LSTM,
    HIDDEN_SIZE=128,
    BATCH_SIZE=32,
    LAYERS=1,
    GUESSES=30
  )
  run_with(cfg,test_only=test_only)

def full_tlin(test_only=False) :
  cfg = dict(
    TRAINING_FILE='data/full_tlin.txt',
    MODEL_FILE='models/full_tlin_cs2cs',
    # Parameters for the model and dataset.
    TRAINING_SIZE=2 ** 18,
    # Try replacing LSTM, GRU, or SimpleRNN.
    RNN=layers.LSTM,
    HIDDEN_SIZE=128,
    BATCH_SIZE=32,
    LAYERS=1,
    GUESSES=30
  )
  run_with(cfg,test_only=test_only)

def cats(test_only=False) :
  cfg = dict(
    TRAINING_FILE='data/cats.txt',
    MODEL_FILE='models/cats_cs2cs',
    # Parameters for the model and dataset.
    TRAINING_SIZE=2 ** 18,
    # Try replacing LSTM, GRU, or SimpleRNN.
    RNN=layers.LSTM,
    HIDDEN_SIZE=128,
    BATCH_SIZE=32,
    LAYERS=1,
    GUESSES=30
  )
  run_with(cfg,test_only=test_only)


def smiles(test_only=False) :
  cfg = dict(
    TRAINING_FILE='data/smiles_trimmed.txt',
    MODEL_FILE='models/smiles_trimmed_cs2cs',
    # Parameters for the model and dataset.
    TRAINING_SIZE=2 ** 18,
    # Try replacing LSTM, GRU, or SimpleRNN.
    RNN=layers.LSTM,
    HIDDEN_SIZE=128,
    BATCH_SIZE=32,
    LAYERS=1,
    GUESSES=30
  )
  run_with(cfg,test_only=test_only)

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

# runs everything, assuming models have been created
def test() :
  tlin(test_only=True)
  full_tlin(test_only=True)
  cats(test_only=True)

# runs everything, builds models
def run() :
  tlin()
  full_tlin()
  cats()
