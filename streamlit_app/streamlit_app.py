import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import pickle
from sklearn.preprocessing import LabelEncoder

# Model architecture
class MultiTaskModel(nn.Module):
    def __init__(self, num_colors, num_types, num_seasons, num_genders):
        super(MultiTaskModel, self).__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze backbone
        self.color_head = nn.Linear(2048, num_colors)
        self.type_head = nn.Linear(2048, num_types)
        self.season_head = nn.Linear(2048, num_seasons)
        self.gender_head = nn.Linear(2048, num_genders)

    def forward(self, x):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)
        color_output = self.color_head(features)
        type_output = self.type_head(features)
        season_output = self.season_head(features)
        gender_output = self.gender_head(features)
        return color_output, type_output, season_output, gender_output

# Function to enhance image resolution/display quality
def enhance_image(image):
    # Resize with high-quality interpolation
    image = image.resize((512, 512), Image.LANCZOS)  # High-quality resizing
    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Increase sharpness (adjustable)
    return image

# Streamlit app
st.title("Fashion Image Classifier")
st.write("Upload an image and click 'Predict' to classify its gender, season, type, and color.")

# Initialize session state
if 'page_state' not in st.session_state:
    st.session_state.page_state = "main"  # Tracks main or predictions page
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'upload_counter' not in st.session_state:
    st.session_state.upload_counter = 0  # To create unique file uploader keys

# Main page: Image upload and Predict button
if st.session_state.page_state == "main":
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key=f"file_uploader_{st.session_state.upload_counter}"
    )

    if uploaded_file is not None:
        # Show the enhanced image
        image = Image.open(uploaded_file).convert('RGB')
        enhanced_image = enhance_image(image)
        st.image(enhanced_image, caption="Uploaded Image (Enhanced)", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            try:
                # Load model weights
                state_dict = torch.load("best_model.pth", map_location="cpu")

                # Detect number of classes from state_dict
                num_colors = state_dict["color_head.weight"].shape[0]
                num_types = state_dict["type_head.weight"].shape[0]
                num_seasons = state_dict["season_head.weight"].shape[0]
                num_genders = state_dict["gender_head.weight"].shape[0]

                # Initialize model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = MultiTaskModel(num_colors, num_types, num_seasons, num_genders).to(device)
                model.load_state_dict(state_dict)
                model.eval()

                # Load LabelEncoders
                try:
                    with open("color_encoder.pkl", 'rb') as f:
                        color_encoder = pickle.load(f)
                    with open("type_encoder.pkl", 'rb') as f:
                        type_encoder = pickle.load(f)
                    with open("season_encoder.pkl", 'rb') as f:
                        season_encoder = pickle.load(f)
                    with open("gender_encoder.pkl", 'rb') as f:
                        gender_encoder = pickle.load(f)
                except FileNotFoundError:
                    st.error("LabelEncoder files (color_encoder.pkl, type_encoder.pkl, season_encoder.pkl, gender_encoder.pkl) not found. Ensure they are in 'E:\\kaggle_dataset\\models\\'.")
                    st.stop()

                # Image preprocessing (for model, unchanged)
                preprocess = transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # Preprocess image for model
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                # Make predictions
                with torch.no_grad():
                    color_output, type_output, season_output, gender_output = model(image_tensor)
                    color_pred = torch.argmax(color_output, dim=1).item()
                    type_pred = torch.argmax(type_output, dim=1).item()
                    season_pred = torch.argmax(season_output, dim=1).item()
                    gender_pred = torch.argmax(gender_output, dim=1).item()

                    # Decode predictions using LabelEncoders
                    color = color_encoder.inverse_transform([color_pred])[0]
                    type_ = type_encoder.inverse_transform([type_pred])[0]
                    season = season_encoder.inverse_transform([season_pred])[0]
                    gender = gender_encoder.inverse_transform([gender_pred])[0]

                # Store predictions and switch to predictions page
                st.session_state.predictions = {
                    "Color": color,
                    "Type": type_,
                    "Season": season,
                    "Gender": gender
                }
                st.session_state.page_state = "predictions"
                st.session_state.upload_counter += 1  # Increment for new uploader key
                st.rerun()

            except FileNotFoundError:
                st.error("Model file 'best_model.pt' not found. Ensure it is in same folder")
            except Exception as e:
                st.error(f"Error during processing or prediction: {e}")

# Predictions page
if st.session_state.page_state == "predictions" and st.session_state.predictions:
    st.subheader("Predictions")
    for key, value in st.session_state.predictions.items():
        st.write(f"**{key}:** {value}")
    
    # Back button to return to main page
    if st.button("Back"):
        st.session_state.predictions = None
        st.session_state.page_state = "main"
        st.session_state.upload_counter += 1  # Ensure new uploader key
        st.rerun()
else:
    if st.session_state.page_state == "main" and uploaded_file is None:
        st.write("Please upload an image to get predictions.")