import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import requests
import io

@st.cache(allow_output_mutation=True)
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


file_id = '1MPkwGD6Jx0b63mvdCKdIBubtCDHMaZry'
destination = 'best_model.pth'
download_file_from_google_drive(file_id, destination) 
# def load_model(model_url):
#     try:
#         response = requests.get(model_url)
#         response.raise_for_status()
#         model = torch.load(response.content, map_location=torch.device('cpu'))
#         model.eval()
#         return model
#     except Exception as e:
#         st.error(f"Failed to load model: {e}")
#         return None

def preprocess_image(image):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def main():
    st.title("Kidney Stone Annotation Tool")
    st.write("Upload an image to get started.")
    
    model_url ="https://drive.google.com/uc?export=download&id=1MPkwGD6Jx0b63mvdCKdIBubtCDHMaZry"
    # model = load_model(model_url)
    MODEL_PATH = "best_model.pth"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        response = requests.get(model_url)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded!") 
    # Load your pre-trained model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
                
            st.write("Annotating...")
            annotated_image = model.predict(preprocess_image(image),conf=0.35)
                
            st.image(annotated_image.show(), caption='Annotated Image', use_column_width=True)

if __name__ == "__main__":
    main()
