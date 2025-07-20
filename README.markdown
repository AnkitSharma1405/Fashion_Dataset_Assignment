# Fashion Product Classification Streamlit App

This repository contains a Streamlit web application that classifies fashion product images into attributes such as **color**, **type**, **season**, and **gender** using a deep learning model trained on the Fashion Product Images Dataset. The model is a multi-task ResNet50-based neural network trained with 5-fold cross-validation, and the app allows users to upload an image and view predicted attributes.

## Table of Contents

- Project Overview
- Requirements
- Directory Structure
- Setup and Installation
- Running the Streamlit App
- Training Details
- Demo video link


## Project Overview

The app uses a pre-trained PyTorch model (`best_model.pth`) to predict four attributes of a fashion product image:

- **Color** (\~46 classes, e.g., Blue, Red)
- **Type** (\~143 classes, e.g., T-shirt, Jeans)
- **Season** (\~4 classes, e.g., Summer, Winter)
- **Gender** (\~5 classes, e.g., Men, Women)

The model was trained using 5-fold cross-validation on the Fashion Product Images Dataset, achieving macro F1-scores of approximately 0.3-0.6 for color/type and 0.5-0.8 for season/gender. The app provides a user-friendly interface to upload images and view predictions.

## Requirements

To run the app, install the following dependencies listed in `requirements.txt`:

```
streamlit==1.36.0
torch==2.3.1
torchvision==0.18.1
Pillow==10.4.0
numpy==1.26.4
scikit-learn==1.5.2
```

**Note**: The app requires Python 3.8 or higher. PyTorch supports both CPU and GPU, but GPU (CUDA) is recommended for faster inference.

## Directory Structure

```
Streamlit_app/
├── streamlit_app.py          # Main Streamlit app script
├── best_model.pth            # Trained model weights
├── requirements.txt          # Python dependencies
├── color_encoder.pkl
├── session_encoder.pkl         
├── gender_encoder.pkl
├── type_encoder.pkl
```

- `streamlit_app.py`: The Streamlit script that loads the model and handles image uploads/predictions.
- `best_model.pth`: The trained model weights from the best fold (based on average F1-score).
- `requirements.txt`: Lists dependencies for easy installation.
- `color_encoder.pkl, session_encoder.pkl, gender_encoder.pkl, type_encoder.pkl:  all these files are used to decode the encoded classes.`

## Setup and Installation

Follow these steps to set up and run the app on your local system:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/AnkitSharma1405/Fashion_Dataset_Assignment.git
   ```

2. **Set Up a Virtual Environment** (recommended):

   ```bash
   python -m venv venv  # create a virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Trained Model**:

   - The `best_model.pth` file is included in the repository.
   - Place `best_model.pth` in the root directory.

5. **Verify Setup**:

   - Ensure  `streamlit_app.py`, `best_model.pth`, encoder files and `requirements.txt` are in the root directory.

## Running the Streamlit App

1. **Activate the Virtual Environment** (if used):

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the Streamlit App**:

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the App**:

   - Open your browser and go to `http://localhost:8501` (default Streamlit port).
   - Upload a fashion product image (PNG/JPG).
   - The app will display predicted attributes (color, type, season, gender).

## Training Details

The model was trained using a Kaggle notebook with the following setup:

- **Dataset**: Fashion Product Images Dataset
- **Model**: ResNet50-based multi-task model (4 heads for color, type, season, gender)

To reproduce training:

1. Use the Kaggle notebook linked here (replace with your notebook URL).
2. Attach the dataset to the notebook.
3. Run cells 1-8 (Cell 8 contains the training loop).
4. Download `best_model.pth` and checkpoints from `/kaggle/working/` (Kaggle Output tab).

## Demo video
[Click here](https://www.loom.com/share/851e91b71f644bfea3bd2af1d7df6de4?sid=525bf4d2-bfc2-40a5-a5ba-2eac63161c49) to watch demo video.

## Thank You
