import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os
from facenet_pytorch import InceptionResnetV1

# LOAD MODEL ARCHITECTURE
model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Linear(512, 150))
model.load_state_dict(torch.load('pokemodel.pt'))

model.eval()


# Define the same transformations used during training
transform = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomRotation(degrees = 15),
              transforms.Resize((224,224)),
              transforms.ToTensor(), 
              transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

st.set_page_config("Ali's Pokedex")

st.title("Ali's Pokedex")

img = st.file_uploader('Upload image of Pokemon: ', type = ['jpeg', 'jpg', 'png'])

if img is not None:


    img = Image.open(img).convert('RGB')
    st.image(img, caption = "Uploaded Pokemon", use_column_width= True)

    input_tensor = transform(img).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class (you need to map class indices to Pokemon names)
    predicted_class = output.argmax().item()
    # Assuming 'output' is your model's output tensor
    class_probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(class_probabilities, dim=1).item()


    # Define the path to your dataset directory
    dataset_dir = 'PokemonData'  # Replace with your dataset path

    # Get a list of class names from the subfolder names
    class_names = sorted(os.listdir(dataset_dir))

    # Create a dictionary to map class indices to Pokemon names
    class_to_name = {i: class_name for i, class_name in enumerate(class_names)}

    # Now, when you have a predicted class index (predicted_class), you can get the Pokemon name
    predicted_pokemon = class_to_name[predicted_class]

    st.write(f'Predicted Pokemon: {predicted_pokemon}')
    # Display the image with the predicted classification
