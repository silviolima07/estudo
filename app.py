
import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
 


import json

import torch

from torchvision import transforms

from tensorflow import keras

#import tensorflow as tf

#tf.compat.v1.disable_eager_execution()




def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    draw_app()    
    PAGES = {
        "Draw numbers from 0 to 9": draw_app,
    }
    page = st.sidebar.selectbox("Opções: ", options=list(PAGES.keys()))
    PAGES[page]()





def draw_app():
    st.sidebar.header("Configuration")
    
    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = "freedraw"
        #st.sidebar.selectbox(
        #    "Drawing tool:",
        #    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        #)
        #stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        stroke_width = 10
        #if drawing_mode == 'point':
        #    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        #stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        #bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        #bg_image = 'png'
        realtime_update = False # st.sidebar.checkbox("Update in realtime", False)
        st.subheader("Imagem 0")
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(0,0, 0)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="rgba(255, 255, 255)",
            background_color="rgba(0, 0, 0)",
            background_image=None,
            update_streamlit=True,
            height=450,
            drawing_mode=   drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar2= st.sidebar.checkbox("Display toolbar", True),
            key="draw_app2",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
            
            img_data = canvas_result.image_data
            im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
            
            #button_id = st.session_state["button_id"]
            #file_path = f"tmp/{button_id}.png"
            
            #im.save(file_path, "PNG")
            #img = Image.open(file_path)
            img_28_28 = im.resize([50,50], Image.Resampling.NEAREST)
            st.subheader("Imagem 28x28")
            st.image(img_28_28)
            
            if st.button("Prever")and data is not None and data.image_data is not None:
           
                    
            #convert_tensor = transforms.ToTensor()
            #file = file_path
            #img = Image.open(file)
            #st.subheader("Imagem1")
            #st.image(img)
            #img_inverted = invert_color(img)
            
            #img = img_inverted
            #img = img.convert('LA')
            #st.image(img)
            #file_tensor = convert_tensor(img)
            #st.write("Imagem na forma de tensor")
            #st.write(file_tensor)
            #st.write(modelo.predict(file_tensor))
            #img_28_28 = img.resize([28,28], Image.Resampling.NEAREST)
            #st.image(img_28_28)
                img_array = np.array(img_28_28)
 
                img_784 = img_array.reshape(-1,28*28)
                img_784 = img_784.astype('float32')
                img_normalizado = img_784/255.0
            
        
                st.title("Previsão")
                 
            #pred = modelo_keras.predict(img_normalizado)
            #st.write(pred)
            #st.title(pred.argmax())
            
        #if canvas_result.json_data is not None:
        #    objects = pd.json_normalize(canvas_result.json_data["objects"])
        #    for col in objects.select_dtypes(include=["object"]).columns:
        #        objects[col] = objects[col].astype("str")
        #    st.dataframe(objects)

            
           
    
        


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit app", page_icon=":pencil2:"
    )
    st.title("Transfer Learning")
    st.sidebar.subheader("Menu")
    
    # Load model
    #PATH= './modelo_normal.pth'
    #modelo = torch.load(PATH)
    
    modelo_keras = keras.models.load_model('./modelo_keras.h5')
    
    #mnist_keras = keras.models.load_model('./mnist_keras.h5')
    
    main()
    

