
from flask import Flask,render_template,request
from flask_mail import Mail,Message
from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS, cross_origin
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_mail import Mail
from flask_mail import Message
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)

from random import sample
from tabnanny import verbose
import pandas as pd # Used for reading the csv data
from nltk.corpus import stopwords
import string # For punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.utils import pad_sequences
from keras.models import load_model
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from nltk.stem.wordnet import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import re
tokenizer = Tokenizer(num_words = 1000) 
app=Flask(__name__)


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT']=465
app.config['MAIL_USERNAME']='pythonfabhost2021@gmail.com'
app.config['MAIL_PASSWORD']='dfxluaswydffbbxz'
app.config['MAIL_USE_TLS']=False
app.config['MAIL_USE_SSL']=True

mail=Mail(app)

@app.route('/')
def index():

    return render_template("index1.html")
def data_preparation(message):
    """Removes stopwords and punctuations

    Args:
        message (string): message

    Returns:
        string: new cleaned message
    """
    only_letters = re.sub("[^a-zA-Z]", " ", str(message))
    only_letters = only_letters.lower()
    only_letters = only_letters.split()
    filtered_result = [word for word in only_letters if word not in stopwords.words('english')]
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    lemmas = ' '.join(lemmas)
    return lemmas

@app.route('/base',methods=['GET','POST'])
def base():
    if request.method=="POST":

        email=request.form["email"]
        subject=request.form["subject"]
        msg=request.form["message"]
        print(msg)
        dict = {"a": "#", "b": "#","c": "#", "d": "#","e": "#","f":"#","g": "#", "h": "#","i": "#", "j": "#","k": "#","l":"#","m": "#", "n": "#","o": "#", "p": "#","q": "#","r":"#","s": "#", "t": "#","u": "#", "v": "#","w": "#","x":"#","y": "#", "z": "#","1": "#", "2": "#","3": "#","4":"#","5": "#", "6": "#","7": "#","8":"#","9": "#", "0": "#"}
        num = msg[::-1]
        for i in dict:
            num = num.replace(i, dict[i])
        encode=f"{num}"
        
        #encode=cipher_suite.encrypt(msg.encode())
        #msg.encode('utf-16', 'surrogatepass')
        # print(encode,"hello")
        messag=Message(subject,sender="HARI",recipients=[email])
        spam=Message(subject,sender="hari",recipients=[email])
        spam.body=encode
        messag.body=msg
        message = request.form.get('message')
        # print(type(message))
        message=[message]
        # msg=request.form["message"]
    
    
    
    df = pd.read_csv("data/spam.csv", encoding = 'latin-1')
    df = shuffle(df)
    encoder=LabelEncoder()
    df['label']=encoder.fit_transform(df['label'])
    df["label"].value_counts()
    df['text']= df.text.apply(data_preparation)
    targets = df.label
    messages = df.text  
    messages_train, messages_test, targets_train, targets_test = train_test_split(messages, targets, test_size=0.2, random_state=20)
    tokenizer.fit_on_texts(messages_train)
    # Tokenize and paddin for train dataset
    mx = len(max(messages, key=len))

    # Tokenize and padding

    num_words = 50000 # The maximum number of words to keep, based on word frequency. 
    max_len = 91
    messages_train_features = tokenizer.texts_to_sequences(messages_train) # Updates internal vocabulary based on a list of sequences.
    # print(len(max(messages_train_features, key=len))) 79
    messages_train_features = pad_sequences(messages_train_features, maxlen = max_len)

    # Tokenize and paddin for test dataset

    messages_test_features = tokenizer.texts_to_sequences(messages_test)
    # print(len(max(messages_test_features, key=len))) #91
    messages_test_features = pad_sequences(messages_test_features, maxlen = max_len)

    # print(len(messages_train_features), len(messages_train_features[0]))
    # print(len(messages_test_features), len(messages_test_features[0]))

    
    
    from keras.models import load_model
    model1=load_model("model/gru_model.h5")
    # message=list(messag)
    sample_texts = [data_preparation(sentence) for sentence in message ]
    print(sample_texts)
   
    txts = tokenizer.texts_to_sequences(sample_texts)
    txts = pad_sequences(txts, maxlen=91)
    # print(txts)
    preds = model1.predict(txts, verbose=0)
    pred=np.around(preds)
    print(pred)
    
    if pred:
        a='THIS IS A BULLYING COMMENT '
        print(a)
        mail.send(spam)
        return render_template('index1.html',a=a)
        
    else:
        mail.send(messag)
        b='NON BULLYING COMMENT'
        print(b)
        
        return render_template('index1.html',b=b)
         
        

if __name__ == "__main__":

    app.run(debug=True)