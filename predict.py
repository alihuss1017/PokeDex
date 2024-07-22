import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os

# LOAD MODEL ARCHITECTURE

def predict_page():
    model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                            nn.ReLU(),
                            nn.Linear(512, 150))
    model.load_state_dict(torch.load('pokemodel.pt'), map_location = torch.device('cpu'))
    model.eval()
    # Define the same transformations used during training
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees = 15),
                transforms.Resize((224,224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    img = st.file_uploader('Upload image of Pokemon: ', type = ['jpeg', 'jpg', 'png'])
    poke_df = pd.read_csv('gen1_pokemon_stats.csv').sort_values(by = "Name")

    if img is not None:

        img = Image.open(img).convert('RGB')

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

        pokemon = poke_df.iloc[predicted_class]
        st.image(f'sprites/{pokemon["Name"].lower()}.gif', width = 200)
        st.write(f'Pokédex Entry: {pokemon["#"]}')
        st.write(f'Predicted Pokémon: {pokemon["Name"]}')

        data = {
            'Category': ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp Def', 'Speed'],
            'Values' : [pokemon["HP"], pokemon["Attack"], pokemon["Defense"],
                        pokemon["Sp. Atk"], pokemon["Sp. Def"], pokemon["Speed"]]
        }

        stats_df = pd.DataFrame(data)
        chart = alt.Chart(stats_df).mark_bar().encode(y = alt.Y('Category:N', sort = None, axis = alt.Axis(labelFontSize = 16)),
                                                      x = alt.X("Values:Q", scale = alt.Scale(domain = [0, 125]), axis = None),
                                                      color = alt.Color('Values:Q', scale = alt.Scale(domain = [20, 125], range = ['red', 'green']), legend = None)
        ).properties(width = 800, height = 350)
        st.title("Stats")
        st.altair_chart(chart, use_container_width = True)

