import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load the pre-trained model
model_path = 'best_model.pth'  # Replace with the actual path
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image):
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted

def main():
    st.title("Image Classification with PyTorch")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        label = predict_image(image)
        st.write(f"Prediction: {label.item()}")  # Assuming the label is a single integer

if __name__ == "__main__":
    main()
