# Rice_Classification

This project implements Convolutional Neural Networks (CNN) and AlexNet to classify rice grain images into their respective varieties.

# Dataset
The dataset used is a labeled image dataset of different rice grain varieties:

Basmati

Jasmine

Arborio

Karacadag

Ipsala

#Project Structure

<pre> . ├── MODELTRAINING.ipynb # Jupyter notebook for training CNN and AlexNet ├── app.py # Flask app to classify uploaded images ├── templates/ │ └── index.html # HTML form for uploading image ├── static/ # Stores uploaded test images ├── AlexNet_model.h5 # Saved AlexNet model (Keras format) ├── label_encoder.pkl # LabelEncoder to map labels to class names ├── requirements.txt # Python dependencies └── README.md # This file </pre>

## 🚀 How to Run

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🏃 Train the Model

```bash
# Option 1: Open the notebook and run all cells
MODELTRAINING.ipynb

# Option 2: Convert notebook to .py and run
jupyter nbconvert --to script MODELTRAINING.ipynb
python MODELTRAINING.py
```

### 🌐 Run the Flask App

```bash
python app.py
```

Then open your browser and visit:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)
