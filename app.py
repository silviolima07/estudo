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
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

#from tensorflow import keras


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Draw numbers": full_app,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()




def full_app():

        drawing_mode = "freedraw" #st.sidebar.selectbox(
  
        stroke_width = 10 #st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = "rgba(255, 255, 255)" #st.sidebar.color_picker("Stroke color hex: ")
        bg_color = "rgba(0, 0, 0)" # st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = None,#st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = True # st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image= None,# Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=450,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar= True, #st.sidebar.checkbox("Display toolbar", False),
            key="full_app",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
        #if canvas_result.json_data is not None:
        #    objects = pd.json_normalize(canvas_result.json_data["objects"])
        #    for col in objects.select_dtypes(include=["object"]).columns:
        #        objects[col] = objects[col].astype("str")
        #     st.dataframe(objects)
        
        
        if st.button("Prever")and data is not None:
            img = canvas_result.image_data
        
            st.write(type(img))
        
            # Get the numpy array (4-channel RGBA 100,100,4)
            input_numpy_array = np.array(img)
        
            # Get the numpy array (4-channel RGBA 100,100,4)
            input_numpy_array = np.array(canvas_result.image_data)
     
        
            # Get the RGBA PIL image
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            input_image.save('user_input.png')
     
     
            st.write("Input Image")
            st.image(input_image)
        
        #img_pil = Image.fromarray(img)
        
        #im = Image.fromarray(img_pil.reshape(28,28))
        
        #img_28_28 = img_pil.resize((28,28), Image.LANCZOS)
        #img_28_28 = np.array(img_pil.resize((28, 28), Image.LANCZOS))
        #img_28_28 = img_pil.resize(size=(28, 28),Image.LANCZOS)
        
        
        #st.image(im)
        
        #img_784 = img_28_28.reshape(-1,28*28)
        #img_784 = img_784.astype('float32')
        #img_normalizado = img_784/255.0
            
        
        #st.title("Previsão")
                 
        #pred = modelo_keras.predict(img_784)
        
        #st.title(pred.argmax())
        






if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    
    #modelo_keras = keras.models.load_model('./modelo_keras.h5')
    
    main()