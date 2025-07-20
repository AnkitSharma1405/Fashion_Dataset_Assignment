import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Path to your CSV file (update this to your actual CSV path)
csv_path = "E:\kaggle_dataset\styles.csv"  

# Load the CSV file
data = pd.read_csv(csv_path)

# Initialize LabelEncoders
color_encoder = LabelEncoder()
type_encoder = LabelEncoder()
season_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Fit encoders on unique values from the CSV
color_encoder.fit(data['baseColour'].dropna().unique())
type_encoder.fit(data['articleType'].dropna().unique())
season_encoder.fit(data['season'].dropna().unique())
gender_encoder.fit(data['gender'].dropna().unique())

# Save encoders to pickle files
with open('color_encoder.pkl', 'wb') as f:
    pickle.dump(color_encoder, f)
with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(type_encoder, f)
with open('season_encoder.pkl', 'wb') as f:
    pickle.dump(season_encoder, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)

# Optional: Print the classes to verify
print("Color classes:", color_encoder.classes_)
print("Type classes:", type_encoder.classes_)
print("Season classes:", season_encoder.classes_)
print("Gender classes:", gender_encoder.classes_)