#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Dark-Sied/Intent_Classification/blob/master/Intent_classification_final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
# from sklearn.preprocessing import OneHotEncoder
# import matplotlib.pyplot as plt
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
# from keras.callbacks import ModelCheckpoint
import os
from finbot import settings
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('default')
cleaned_words = ''

# In[2]:


def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  # print(df.head())
  intent = df["Intent"]
  global unique_intent
  global sentences
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"])
  return (intent, unique_intent, sentences)

def train_and_save_model():
    global cleaned_words
    global word_tokenizer
    # intent, unique_intent, sentences = load_dataset(os.path.join(settings.STATIC_DIR,"Dataset.csv"))
    nltk.download("stopwords")
    nltk.download("punkt")
    stemmer = LancasterStemmer()
    cleaned_words = cleaning(sentences)
    word_tokenizer = create_tokenizer(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    max_length = max_length(cleaned_words)
    encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
    padded_doc = padding_doc(encoded_doc, max_length)
    output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
    encoded_output = encoding_doc(output_tokenizer, intent)
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    output_one_hot = one_hot(encoded_output)

    train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)

    model = create_model(vocab_size, max_length)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])




def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])

  return words

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

def max_length(words):
  return(len(max(words, key = len)))

def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


from sklearn.model_selection import train_test_split

def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = True))
  model.add(Bidirectional(LSTM(128)))
#   model.add(LSTM(128))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(23, activation = "softmax"))

  return model



def predictions(text):
  intent, unique_intent, sentences = load_dataset(os.path.join(settings.STATIC_DIR,"Dataset.csv"))
  model = load_model("model.h5")
  print("{}".format(model.summary()))
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  # cleaned_words = cleaning(text)
  # print(cleaned_words)
  # word_tokenizer = create_tokenizer(cleaned_words)
  # print("test word = {}".format(test_word))
  # filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
  # token = Tokenizer(filters = filters)
  # token.fit_on_texts(clean)
  # print("sentences = {}".format(sentences))
  cleaned_words = cleaning(sentences)
  max_lengt = max_length(cleaned_words)
  word_tokenizer = create_tokenizer(cleaned_words)
  # print(clean)
  # print("Token = {}".format(token))
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print("Test LC with text to sequence {}".format(test_ls))
  # print("execiting work_tokenizer")
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  # x = padding_doc(test_ls, 28)
  # print("before predict function")
  # print(x)
  print(test_ls)
  print("MAx Length = {}".format(max_lengt))
  x = padding_doc(test_ls, max_lengt)
  print(x)
  pred = model.predict(x)
  # pred = ""
  return pred


def get_final_output(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))
  return classes[0]

def get_pred(text):
    # text ="test"
    pred = predictions(text)
    return get_final_output(pred, unique_intent)
