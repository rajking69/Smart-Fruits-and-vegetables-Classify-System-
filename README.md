# Fruits & Vegetables Recognition System

A machine learning project that uses a Convolutional Neural Network (CNN) model to classify various fruits and vegetables.

## Project Structure
- `main.py`: Streamlit web application
- `model.h5`: Trained model file
- `labels.txt`: Class labels for predictions
- Jupyter Notebooks:
  - Training notebook: For training the model
  - Testing notebook: For testing the model

## Setup Instructions

1. **Create and Activate Virtual Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Run Streamlit Web App**:
   ```bash
   streamlit run main.py
   ```
   This will open the web application in your default browser.

2. **Using the Web Interface**:
   - Select a page from the sidebar: Home, About Project, or Contact
   - On the Home page, upload an image of a fruit or vegetable
   - Click "Predict" to get the classification result

## Training and Testing

1. **Training the Model**:
   - Open `Smart Fruits and vegetables Classify System (Training fruit vegetable).ipynb` in Jupyter Notebook
   - Follow the instructions in the notebook to train the model
   - Requirements:
     - Training dataset in the correct directory structure
     - GPU recommended for faster training

2. **Testing the Model**:
   - Open `Smart Fruits and vegetables Classify System (Testing fruit and vegetable recognition).ipynb` in Jupyter Notebook
   - Follow the instructions to test the model with new images

## Dataset Structure
```
dataset/
├── train/      # 100 images per class
├── test/       # 10 images per class
└── validation/ # 10 images per class
```

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow (PIL)

## Developers
- Sheikh Mohammad Rajking (rajking4457@gmail.com)
- Adri Shikar Barua (adrishikharbarua77452@gmail.com)

## Note
Make sure you have all the required image files in the correct locations:
- `Cover Photo.jpg`
- `rajking.JPG`
- `adri.png`
- `model.h5`
- `labels.txt` 