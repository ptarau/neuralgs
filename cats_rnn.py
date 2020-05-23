# -*- coding: utf-8 -*-

# derived from keras, addition example at:
# https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

'''
# An implementation of sequence to sequence learning for 
performing successor on tree based numbers

it seemed most likely unlearnable by neural networks
but with fairly short training it made it close to 100%

Input: "112120000"
Output: "112112200"
Padding is handled by using a repeated sentinel character ('0')

or, with a different representation:

Input ="[[][][[]][][[[]][]]]  "
Output="[[][[][]][][[[]][]]]  "

with padding handled by white space
'''

# tree based arithmetic operations
# derived from:
# https://github.com/ptarau/Tree-based-Arithmetic-Systems-in-Python

from catnums import *

from keras.models import Sequential
from keras import layers
import numpy as np

# Parameters for the model and dataset.
TRAINING_SIZE = 2**16
# All the symbols, 0 for padding

questions,expected=nats_from_str(t(0),TRAINING_SIZE)
MAXLEN = len(questions[0])

# Try replacing LSTM, GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

ITERATIONS=TRAINING_SIZE // BATCH_SIZE
GUESSES=20


class CharacterTable(object):
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

    def encode(self, seq, num_rows):
        """One-hot encode given string seq.

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(seq):
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

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

ctable = CharacterTable(chars)

print('Vectorization...',(len(questions), MAXLEN, len(chars)))
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)

for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
    #print('!!!!',x[i])
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, MAXLEN)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'MAXLEN' times as that's the maximum
# length of output
model.add(layers.RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
def run() :
  for iteration in range(0, ITERATIONS):
    print('ITERATIONS:',ITERATIONS)
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

    # Select GUESSES samples from the validation set at random so we can visualize
    # errors.
    for i in range(GUESSES):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '+' + colors.close, end=' ')
        else:
            print(colors.fail + '-' + colors.close, end=' ')
        print(guess)
  model.save('cats_model.tf') #, save_format='tf')
