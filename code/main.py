# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:25:13 2020

@author: Daniel Brunnsåker
"""

import numpy as np
from tensorflow import keras
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import normalize
import random

def load_text(txt_path):
    '''
    Input: path to raw txt-file
    Output: loaded & lowercased file
    '''
    
    lowercase_speeches = open(txt_path,encoding="utf8").read().lower()
    return lowercase_speeches

def define_sequences(raw_string,seq_length):
    '''
    Codes input-string into readable format for RNN (converts letters to corresponding integers)
    
    Input: 
        raw_string = data in form of string
        seq_length = integer, amount of pre-sequential "letters" RNN will use to make next prediction 
    Output: 
        X = input-data for RNN, format - (N, seq_length, 1)
        y = labels for RNN (in form of overlay-mask), format - (N, #uniquesymbols)
        c_indices: dictionary of available symbols in input-text
    '''
    tokenizer = Tokenizer(char_level = True)
    tokenizer.fit_on_texts(raw_string)
    c_indices = tokenizer.word_index
    
    X = []
    y = []
    
    for i in range(len(raw_string)-seq_length):

        inputseq = raw_string[i:i+seq_length]
        outputseq = raw_string[i+seq_length]
    
        X.append([c_indices[char] for char in inputseq])
        y.append(c_indices[outputseq])
        
    return X,y, c_indices

def convert_format(X,y,c_indices,seq_length):
    '''
    Converts data and label arrays into format useable for keras
    
    Input:
        X: input-data for RNN
        y: input-labels for RNN
        c_indices: dictionary of available symbols in input-text
        seq_length: integer, amount of pre-sequential "letters" RNN will use to make next prediction
    Output:
        X: formatted, reshaped X (for use in sequential model)
        y: label-vector converted to categorical (0,1)
    '''
    X = np.reshape(X, (len(X),seq_length, 1))
    X = X/len(c_indices)
    
    y = keras.utils.to_categorical(y)
    
    return X,y

def create_samplevector(prediction, toppicks):
    '''
    creates a sampling vector from n top predictions, done to ensure some variability in predictions,
    possibly avoiding loop
    
    Input:
        prediction: predictions made by RNN, format - ()
        toppicks: integer, picks out n most probable predictions and samples
    Output:
        norm_vec: normalized vector of probabilities, for use in sampling function (such as randomchoice)
    '''
    vec = []
    for k in range(len(prediction[0])):

        indices = np.argpartition(prediction, -toppicks, axis=1)[:, -toppicks:]
    
        if k in indices:
            vec.append(prediction[0][k])
        else:
            vec.append(0)
    
    norm_vec = vec/sum(vec)
    return norm_vec


def generate_sample(model, lngth):
    '''
    generates lngth-length text using pre-trained RNN-model
    
    Input:
        model: trained text-generating model
        lngth: length of generated sample (integer - symbols)
    Output: 
        sample: generated text, string
    '''
    
    txt = []
    start = np.random.randint(1,10000) #random seed
    pattern = X[start]

    ichar = dict(enumerate(c_indices.keys()))
    #print(pattern)
    for i in range(lngth):

        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / len(ichar)
        prediction = model.predict(x, verbose=0)
    
        #create sampling function
    
        norm_vec = create_samplevector(prediction, 2)
        index = np.random.choice(len(c_indices)+1, 1, p=norm_vec)

        result = ichar[index[0]-1].rstrip('\n\r')
        
        txt.append(result)
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]
    
    sample = ''.join(txt)
    return sample

class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        '''
        Prints a generated sample after each epoch, allows for quick and easy performance surveys
        '''
        
        print('\n \n Generating sample: ')
        sample = generate_sample(model, 100)
        print(sample)
        print('\n ')

def rnn_model(nodes, drop):
    '''
    Model architecture
    
    Input:
        nodes: amount of trainable nodes in LSTM-layer
        drop: dropout-rate (0-0.5)
    Output:
        model: sequential model
    '''
    
    model = keras.Sequential(
    [
     keras.layers.LSTM(nodes, return_sequences=False,input_shape = (X.shape[1], X.shape[2])),
     keras.layers.Dropout(drop),
     keras.layers.Dense(y.shape[1],activation = 'softmax')
     ])
    
    return model


def stacked_rnn_model(node_vector, drop):
    '''
    Model architecture
    
    Input:
        nodes: amount of trainable nodes in LSTM-layer
        drop: dropout-rate (0-0.5)
    Output:
        model: sequential model
    '''
    
    model = keras.Sequential(
    [
     keras.layers.LSTM(node_vector[0], return_sequences=True,input_shape = (X.shape[1], X.shape[2])),
     keras.layers.Dropout(drop),
     keras.layers.LSTM(node_vector[1], return_sequences=False),
     keras.layers.Dropout(drop),
     keras.layers.Dense(y.shape[1],activation = 'softmax')
     ])
    
    return model

txt_path = r'C:\Users\Daniel Brunnsåker\Desktop\trump\data\speech_addition.txt'
raw_string = load_text(txt_path)

seq_length = 5
X, y, c_indices = define_sequences(raw_string,seq_length)
X, y = convert_format(X,y,c_indices, seq_length)
nodes = [128,64]
model = stacked_rnn_model(nodes, 0.25)

model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

model.fit(X, y, epochs=250, batch_size=128, verbose = 1,callbacks=[CustomCallback()])


























 