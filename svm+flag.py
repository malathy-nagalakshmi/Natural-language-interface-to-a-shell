import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
df = pd.read_csv('base_class_flag1.csv',error_bad_lines=False)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)


corr=[]
index=0
for i in range(len(df['command'])):
    if("ifconfig" in df['command'][i]):
        sentence="Show all interactive network interface in a short list."
        sent=df['Description'][i]
        tf.logging.set_verbosity(tf.logging.ERROR)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(embed([sentence]))
            sent_emb=session.run(embed([sent]))
            #sentence = sentence.astype(np.float)
            corr.append(np.inner(sent_emb, message_embeddings))
        max=corr[0]
        for j in range(1,len(corr)):
            if(corr[j]>max):
                max=corr[j]
                index=j
        print(df['command'][i+index])
