from flask import Flask, render_template,request
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from googletrans import Translator
import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm.notebook import tqdm # how much data is process till now
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input # extract features from image data.
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input , Dense , LSTM , Embedding , Dropout , add




def idx_to_word(integer, tokenizer):
      for word, index in tokenizer.word_index.items():
            if index == integer:
                  print(word, "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                  return word
      print("rrrrrrrrr")
      return None


def predict_caption(model, image, tokenizer, max_length):
      in_text = 'startseq'
      for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length)
            yhat = model.predict([image, sequence], verbose=0)
            yhat = np.argmax(yhat)
            print(yhat,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            word = idx_to_word(yhat, tokenizer)
            if word is None:
                  break
            in_text += " " + word
            if word == 'endseq':
                  break

      in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
      translator = Translator()
      in_text = translator.translate(in_text, src='en', dest='ne').text
      
      print("Translated Caption:", in_text)

      print(in_text)
      return in_text

#-----------------------------------------------------------flask---------------------------------------


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def Home():
      print("/////////////////////////////////////////////////////////////")
      return render_template('index.html')

@app.route('/caption',methods=['POST'])
def caption(): 

      image_folder = './static'
      image_path = os.path.abspath(image_folder)
      
      #uploaded image name and saved in destined folder
      if request.method == 'POST':
            file = request.files['file1']
            filename = file.filename
            image_path = image_folder + '/' + filename 
            file.save(image_path)
            print(image_path, "*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*")
      print("****************************************************************")
      # Load model
      saved_model_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\imgmodel.h5'  # Adjust the file name as needed
      model = load_model(saved_model_path)     
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      # Load model
      saved_model_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\vgg16_model.h5'  # Adjust the file name as needed
      vgg16_model = load_model(saved_model_path)
      # load Features and tokenizers
      tokenizer_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\tokenizer.pkl'
      with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
      max_length_path = 'C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\max_length.txt'
      with open(max_length_path, 'r') as f:
            max_length = int(f.read())
      with open(os.path.join("C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb", 'imgfeatures.pkl'), 'rb') as f:
            feature = pickle.load(f)
      
      
      # load image
      image = load_img(image_path, target_size=(224, 224))
      # convert image pixels to numpy array
      image = img_to_array(image)
      # reshape data for model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # preprocess image for vgg
      image = preprocess_input(image)
      # extract features
      feature = vgg16_model.predict(image)
      print(feature)
      # predict from the trained model
      caption = predict_caption(model, feature, tokenizer, max_length)
      image = Image.open(image_path)
            
      if request.method == 'POST':
            file = request.files['file1']
            if file and file.filename:
                  file_data = file.read()
                  if file_data:
                        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
                        print("Image shape:", image.shape)
                  else:
                        print("Error: Empty file data.")
            else:
                  print("Error: No file selected.")
      image_folder = "C:\\Users\\LEVEL51PC\\Desktop\\imageCaptionWeb\\try"
      print(image_folder)
      print(filename, "lllllllllllllllllllllllllllllllllllllllllllllllllllllll")
      image_path = os.path.join(image_folder, filename)
      print(image_path)
      return render_template('caption.html', caption=caption, filename=filename)

if __name__ == "__main__":
      app.run(debug = True)

