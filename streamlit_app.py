import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load('path/to/your/model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

# Function to make prediction and annotate image
def predict_and_annotate(image, model):
    # Define the transformations for the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Move the input to the same device as the model
    device = torch.device('cpu')
    input_batch = input_batch.to(device)
    model.to(device)
    
    # Get the model's prediction
    with torch.no_grad():
        output = model(input_batch)
    
    # Post-process the output and annotate the image (example for segmentation task)
    output = output.squeeze().cpu().numpy()
    mask = (output > 0.5).astype(np.uint8)  # Example thresholding
    mask = Image.fromarray(mask * 255)
    
    # Combine the mask with the original image
    annotated_image = Image.blend(image.convert('RGBA'), mask.convert('RGBA'), alpha=0.5)
    
    return annotated_image

# Streamlit app
def main():
    st.title("Kidney Stone Annotation Tool")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Annotating...")
        annotated_image = predict_and_annotate(image, model)
        
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)

if __name__ == "__main__":
    main()
