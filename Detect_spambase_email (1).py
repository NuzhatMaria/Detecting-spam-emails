#!/usr/bin/env python
# coding: utf-8

# # DETECT AN EMAIL

# In[1]:


import pandas as pd
import string
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# In[2]:


data = pd.read_csv("spam_ham_dataset.csv",encoding = "'latin'")


# In[3]:


data.head()


# In[4]:


data['text'] =data.text
data['spam'] =data.label


# # Splitting the data

# In[5]:


from sklearn.model_selection import train_test_split
emails_train, emails_test, target_train, target_test = train_test_split(data.text,data.spam,test_size = 0.2) 
data.info


# In[6]:


emails_train.shape


# In[7]:


import string
import re

def to_lower(word):
    result = word.lower()
    return result

def remove_special_character(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def remove_hyperlink(word):
    return re.sub(r"http\s+", "", word)

def replace_newline(word):
    return word.replace('\n', '')

def clean_up_pipeline(sentence):
    print(f"Input the sentence: {sentence}")
    cleaning_util = [
        to_lower, remove_special_character,
        remove_number, remove_whitespace,
        remove_hyperlink, replace_newline
    ]
    for o in cleaning_util:
        sentence = o(sentence)
    return sentence

# Assuming you have emails_train and emails_test defined somewhere
x_train = [clean_up_pipeline(o) for o in emails_train]
x_test = [clean_up_pipeline(o) for o in emails_test]


# In[8]:



x_train[0]
   


# # Label Encoding

# In[9]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder() 
train_y= le.fit_transform(target_train.values)
test_y =le.transform(target_test.values)


# In[10]:


train_y


# In[11]:


pip install tensorflow


# # Tokenize

# In[12]:


embed_size = 100 #how big is the vector
max_feature = 5000  #number of rows in embedding vector i.e., max number of unique words
max_len = 2000 #max number of words allowed in a question to use


# In[13]:


import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words= max_feature)
tokenizer.fit_on_texts(x_train)
x_train_features = tokenizer.texts_to_sequences(x_train)
x_test_features =tokenizer.texts_to_sequences(x_test)
x_train_features= pad_sequences(x_train_features , maxlen =max_len)
x_test_features =pad_sequences(x_test_features ,maxlen = max_len)
x_train_features[0]


# # Model 

# In[14]:


from keras.layers import Dense ,Input ,LSTM , Embedding , Dropout, Activation 
from keras.layers import Bidirectional 
from keras.models import Model , Sequential 
from keras.metrics import Accuracy 
import tensorflow as tf


# In[15]:


embed_layer = 32
model= tf.keras.Sequential()
model.add(Embedding(max_feature , embed_layer ,input_length= max_len))
model.add(Bidirectional(tf.keras.layers.LSTM(64)))

model.add(Dense(16, activation ='relu'))# this layer introduces non linearity 
model.add(Dropout(0.1))  #this layer prevents overfitting
model.add(Dense(1 ,activation ='sigmoid')) # this layer has 1 unit for output with binary classification 
model.compile(loss ='binary_crossentropy', optimizer ='adam' ,metrics =['accuracy'])
print(model.summary())


# In[16]:


history = model.fit(x_train_features, train_y, batch_size=512, epochs=20, validation_data=(x_test_features, test_y))


# In[17]:


from matplotlib import pyplot as plt 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train' ,'test'] ,loc ='upper left')
plt.grid()
plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix ,f1_score , precision_score , recall_score


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt 
ax =plt.subplot()
y_predict = [1 if o > 0.5 else 0 for o in model.predict(x_test_features)]

cf_matrix = confusion_matrix(test_y , y_predict)
sns.heatmap(cf_matrix ,annot = True , ax= ax, cmap= 'Blues',fmt ='')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True label');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Not Spam', 'Spam']);
ax.yaxis.set_ticklabels(['Not Spam','Spam']);


# In[24]:


tn ,fp , fn ,tp =confusion_matrix(test_y , y_predict).ravel()


# In[25]:


print("Precision: {:.2f}%".format(100*precision_score(test_y ,y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(test_y, y_predict)))
print("F1 Score: {:.2f}%".format(100 *f1_score(test_y ,y_predict)))



# In[27]:


f1_score(test_y, y_predict)


# In[ ]:




