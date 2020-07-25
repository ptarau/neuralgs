"""
DERIVED from:

Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2020/04/26
Description: Character-level recurrent sequence-to-sequence model.

Made it indepandent of the specific data files.

"""

"""
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. 

**Summary of the algorithm**

- We start with input sequences from a domain 
    and corresponding target sequences from another domain.
    
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

"""
## Prepare the data
"""

# Vectorize the data.

class Data() :
  def __init__(self,data_path,sep,cfg):
    self.input_texts = []
    self.target_texts = []
    self.io_map=dict()
    self.input_characters = {" "} #set()
    self.target_characters = {" "} #set()
    with open(data_path, "r", encoding="utf-8") as f:
      lines = f.read().split("\n")
    for line in lines[: min(cfg['num_samples'], len(lines) - 1)]:
      parts=line.split(sep)
      input_text, target_text = parts[0:2]
      self.io_map[input_text]=target_text # for testing accuracy
      # We use "tab" as the "start sequence" character
      # for the targets, and "\n" as "end sequence" character.
      target_text = "\t" + target_text + "\n"
      self.input_texts.append(input_text)
      self.target_texts.append(target_text)
      for char in input_text:
        if char not in self.input_characters:
          self.input_characters.add(char)
      for char in target_text:
        if char not in self.target_characters:
          self.target_characters.add(char)

    self.input_characters = sorted(list(self.input_characters))
    self.target_characters = sorted(list(self.target_characters))
    self.input_token_index = dict([(char, i) for i, char in enumerate(self.input_characters)])
    self.target_token_index = dict([(char, i) for i, char in enumerate(self.target_characters)])

    self.num_encoder_tokens = len(self.input_characters)
    self.num_decoder_tokens = len(self.target_characters)
    self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
    self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])



def build_model(data,cfg) :

  encoder_input_data = np.zeros(
    (len(data.input_texts), data.max_encoder_seq_length, data.num_encoder_tokens), dtype="float32"
  )
  decoder_input_data = np.zeros(
    (len(data.input_texts), data.max_decoder_seq_length, data.num_decoder_tokens), dtype="float32"
  )
  decoder_target_data = np.zeros(
    (len(data.input_texts), data.max_decoder_seq_length, data.num_decoder_tokens), dtype="float32"
  )

  for i, (input_text, target_text) in enumerate(zip(data.input_texts, data.target_texts)):
    for t, char in enumerate(input_text):
      encoder_input_data[i, t, data.input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:, data.input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
      # decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_input_data[i, t, data.target_token_index[char]] = 1.0
      if t > 0:
        # decoder_target_data will be ahead by one timestep
        # and will not include the start character.
        decoder_target_data[i, t - 1, data.target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1:, data.target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, data.target_token_index[" "]] = 1.0

  """
  ## Build the model
  """

  # Define an input sequence and process it.
  encoder_inputs = keras.Input(shape=(None, data.num_encoder_tokens))
  encoder = keras.layers.LSTM(cfg['latent_dim'], return_state=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)

  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = keras.Input(shape=(None, data.num_decoder_tokens))

  # We set up our decoder to return full output sequences,
  # and to return internal states as well. We don't use the
  # return states in the training model, but we will use them in inference.
  decoder_lstm = keras.layers.LSTM(cfg['latent_dim'], return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
  decoder_dense = keras.layers.Dense(data.num_decoder_tokens, activation="softmax")
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

  model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

  return model,encoder_input_data, decoder_input_data, decoder_target_data

def learn(model,cfg,encoder_input_data, decoder_input_data, decoder_target_data):
    """
    ## Train the model
    """

    #model.summary()

    history=model.fit(
      [encoder_input_data, decoder_input_data],
      decoder_target_data,
      batch_size=cfg['batch_size'],
      epochs=cfg['epochs'],
      validation_split=0.2,
    )

    return history


def decode_sequence(input_seq,encoder_model,decoder_model,data,reverse_target_char_index):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, data.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, data.target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
      output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

      # Sample a token
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_char = reverse_target_char_index[sampled_token_index]
      decoded_sentence += sampled_char

      # Exit condition: either hit max length
      # or find stop character.
      if sampled_char == "\n" or len(decoded_sentence) > data.max_decoder_seq_length:
        stop_condition = True

      # Update the target sequence (of length 1).
      target_seq = np.zeros((1, 1, data.num_decoder_tokens))
      target_seq[0, 0, sampled_token_index] = 1.0

      # Update states
      states_value = [h, c]
    return decoded_sentence

def infer(data,cfg,encoder_input_data, decoder_input_data, decoder_target_data, model_file) :
  """
  ## Run inference (sampling)

  1. encode input and retrieve initial decoder state
  2. run one step of decoder with this initial state
  and a "start of sequence" token as target.
  Output will be the next target token.
  3. Repeat with the current target token and current states
  """
  model = keras.models.load_model(model_file)

  # Define sampling models
  # Restore the model and construct the encoder and decoder.

  encoder_inputs = model.input[0]  # input_1
  encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
  encoder_states = [state_h_enc, state_c_enc]
  encoder_model = keras.Model(encoder_inputs, encoder_states)

  decoder_inputs = model.input[1]  # input_2
  decoder_state_input_h = keras.Input(shape=(cfg['latent_dim'],), name="input_3")
  decoder_state_input_c = keras.Input(shape=(cfg['latent_dim'],), name="input_4")
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  decoder_lstm = model.layers[3]
  decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
  )
  decoder_states = [state_h_dec, state_c_dec]
  decoder_dense = model.layers[4]
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
  )

  # Reverse-lookup token index to decode sequences back to
  # something readable.

  reverse_target_char_index = dict((i, char) for char, i in data.target_token_index.items())

  for _ in range(50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index=random.randrange(len(data.io_map))
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq,encoder_model,decoder_model,data,reverse_target_char_index)
    decoded_sentence = decoded_sentence.strip()
    #print("-")
    #print("Input sentence:", data.input_texts[seq_index])
    #print("Decoded sentence:", decoded_sentence)
    input_text=data.input_texts[seq_index]
    target_text=data.io_map[input_text]
    ok=target_text==decoded_sentence
    if ok : r="+"
    else : r="-"
    #print(r, len(input_text), len(target_text), '==', len(decoded_sentence))
    print(r,':',input_text,'->',target_text,'==',decoded_sentence)

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

def run_with(data_file,model_file,infer_only=False,cfg =
    {'sep':':','batch_size': 64, 'epochs': 100, 'latent_dim': 256, 'num_samples': 10000,'iterations' : 1}):
  sep=cfg['sep']
  data = Data(data_file, sep, cfg)

  model,encoder_input_data, decoder_input_data, decoder_target_data=build_model(data,cfg)
  model.summary()
  print(cfg)
  history=None
  if not infer_only :
    data_size=len(data.io_map)
    model.compile(
      #optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
      optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"]
    )
    for i in range(cfg['iterations']) :
       print("ITERATION:", i, '/', cfg['iterations'], 'on data_size:',data_size,'file:',data_file)
       history=learn(model,cfg, encoder_input_data, decoder_input_data, decoder_target_data)
    model.save(model_file)
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

  infer(data,cfg,encoder_input_data, decoder_input_data, decoder_target_data,model_file)

def theo(infer_only=False) :
  run_with('data/tlin.txt','models/tlin_s2s',infer_only=infer_only,cfg =
    {'sep':':','batch_size': 64*4, 'epochs': 1, 'latent_dim': 256, 'num_samples': 200000, 'iterations':94})

def full_theo(infer_only=False) :
  run_with('data/full_tlin.txt','models/full_tlin_s2s',infer_only=infer_only,cfg =
    {'sep':':','batch_size': 64*4, 'epochs': 1, 'latent_dim': 256, 'num_samples': 200000, 'iterations':94})

def cats(infer_only=False) :
  run_with('data/cats.txt','models/cats_s2s',infer_only=infer_only,cfg =
    {'sep':':','batch_size': 64, 'epochs': 1, 'latent_dim': 256, 'num_samples': 200000, 'iterations':100})

def test() :
  theo(infer_only=True)
  full_theo(infer_only=True)
  cats(infer_only=True)

def run() :
  theo()
  full_theo()
  cats()
