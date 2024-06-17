import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# # Function to load the model
# model_path = 'best_model.pth'
# model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model
# model.eval()  # Set model to evaluation mode
def load_yolov5_model(weights_path):
    # Load model
    model = torch.hub.load('ultralytics/yolov8', 'yolov8', pretrained=False)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(checkpoint.state_dict())
    
    # Set model in evaluation mode
    model.eval()
    
    return model
model = load_yolov5_model('best_model.pth')

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),                    # Resize to 256x256
        transforms.CenterCrop(224),                # Crop the center 224x224 region
        transforms.ToTensor(),                     # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = preprocess(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load the model
  # Replace with your model path
# Streamlit app
st.title('Image Classification with PyTorch')

# Upload image through Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image and perform inference
    with st.spinner('Predicting...'):
        img_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(img_tensor)

        # Get predicted class
        predicted_class = torch.argmax(output).item()

    # Display prediction
    st.write(f'Prediction: {predicted_class}')
