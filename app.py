from flask import Flask, render_template, flash, request, url_for, redirect, session
from flask_bootstrap import Bootstrap
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model


Static_Images = os.path.join('static', 'images')

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['Upload_Images'] = Static_Images
max_review_length = 500

model_path = 'sentiment_analysis.h5'  
model = load_model(model_path)





@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("index.html")

@app.route('/sent_pred', methods = ['GET', 'POST'])
def sent_pred():
    if request.method=='POST':
        text = request.form['text']
        sentiment = ''
        
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split() 
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500) 
        vector = np.array([x_test.flatten()])
        
        probability = model.predict(array([vector][0]))[0][0]
        sent = model.predict_classes(array([vector][0]))[0][0]
        
        if sent == 0:
            sentiment = 'Negative'
            thumbsnail = os.path.join(app.config['Upload_Images'], 'thumbs_down.png')
        else:
            sentiment = 'Positive'
            thumbsnail = os.path.join(app.config['Upload_Images'], 'thumbs_up.png')
            
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability, image=thumbsnail)
        
      
       

if __name__ == "__main__":
    
    app.run(debug=True)