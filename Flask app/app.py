import sys
import os
import glob
import re
import cv2
import numpy as np
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
model = keras.models.load_model('CNN+SVM_FOR_CYCLONE_DETECTION.h5')


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


def prepare(filepath):
    image = cv2.imread(filepath , cv2.IMREAD_GRAYSCALE)
    ret,image = cv2.threshold(image, 180, 220, cv2.THRESH_BINARY)

    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel)

    image = cv2.medianBlur(image, 1)
    image = cv2.resize(image, (300,300)) 

    image = cv2.resize(image, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA) 
    image = image/255.0
    #image = np.expand_dims(image, axis=0)
    return image.reshape(-1,300,300,1)

@app.route('/prediction', methods=["POST"])
def prediction():    
	img = request.files['img']
	img.save('img.jpg')
	prediction = model.predict([prepare('img.jpg')])

	if(prediction <= 0):
	    pred = "cyclone"
	else:
	    pred = "No_cyclone"

	return render_template("prediction.html", data=pred)




if __name__ =="__main__":
    app.run(debug=True)