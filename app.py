import os
import numpy as np
import pickle
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained CNN model
model = load_model('CNNMODELMAIN.h5')

# model1 = load_model('AlexNet_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Class names in correct order
class_names = [ 'Arborio','Basmati', 'Ipsala' ,'Jasmine','Karacadag']

IMG_SIZE = 224

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)  # shape: (1, 58, 58, 3)

@app.route('/')
def index():
    return render_template('index.html', prediction="", img_path="")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded", img_path="")

    img_file = request.files['file']
    if img_file.filename == '':
        return render_template('index.html', prediction="No selected file", img_path="")

    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_names[predicted_class_index]

    # predictions1 = model1.predict(processed_img)
    # predicted_class_index1 = np.argmax(predictions1)
    # predicted_label1 = "<br>"+"ALEXNET PREDICTED CLASS"+class_names[predicted_class_index1]

    return render_template('index.html', prediction=predicted_label, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
