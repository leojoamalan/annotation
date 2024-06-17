# import streamlit as st
# import torch
# from torchvision import transforms
# from PIL import Image

# # # Function to load the model
# # model_path = 'best_model.pth'
# # model = torch.load(model_path, map_location=torch.device('cpu'))  # Load model
# # model.eval()  # Set model to evaluation mode
# def load_yolov5_model(weights_path):
#     # Load model
#     model = torch.hub.load('ultralytics/yolov8', 'yolov8', pretrained=False)
    
#     # Load weights
#     checkpoint = torch.load(weights_path, map_location='cpu')['model']
#     model.load_state_dict(checkpoint.state_dict())
    
#     # Set model in evaluation mode
#     model.eval()
    
#     return model
# model = load_yolov5_model('best_model.pth')

# # Function to preprocess the image
# def preprocess_image(image):
#     preprocess = transforms.Compose([
#         transforms.Resize(256),                    # Resize to 256x256
#         transforms.CenterCrop(224),                # Crop the center 224x224 region
#         transforms.ToTensor(),                     # Convert PIL Image to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
#     ])
#     image = preprocess(image)
#     return image.unsqueeze(0)  # Add batch dimension

# # Load the model
#   # Replace with your model path
# # Streamlit app
# st.title('Image Classification with PyTorch')

# # Upload image through Streamlit file uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)

#     # Preprocess the image and perform inference
#     with st.spinner('Predicting...'):
#         img_tensor = preprocess_image(image)
#         with torch.no_grad():
#             output = model(img_tensor)

#         # Get predicted class
#         predicted_class = torch.argmax(output).item()

#     # Display prediction
#     st.write(f'Prediction: {predicted_class}')
import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import requests

# Function to load YOLO NAS model
def load_yolo_nas_model(weights_path):
    # Define and load your YOLO NAS model here
    # Example:
    class YOLONAS(torch.nn.Module):
        def __init__(self):
            super(YOLONAS, self).__init__()
            # Define your model layers

        def forward(self, x):
            # Implement your forward pass logic
            return x

    model = YOLONAS()
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def detect_objects(image, model):
    # Perform object detection here
    # Example: process image and get detections
    # Replace this with your actual detection code
    detections = model(image)
    return detections

def main():
    st.title('YOLO NAS Object Detection')
    st.write('Upload an image for object detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Load YOLO NAS model (replace with your model path)
        model_path = 'https://github.com/leojoamalan/annotation/blob/main/best_model.pth'  # Update with your GitHub URL
        model = load_yolo_nas_model(model_path)

        # Perform object detection
        detections = detect_objects(image, model)

        # Display detection results
        st.subheader('Detection Results:')
        st.write(detections)  # Replace with your display logic

if __name__ == "__main__":
    main()

