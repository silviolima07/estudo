
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

import json

import torch

from torchvision import transforms

from tensorflow import keras




def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Draw numbers from 0 to 9": png_export,
    }
    page = st.sidebar.selectbox("Opções: ", options=list(PAGES.keys()))
    PAGES[page]()





def draw_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Configure canvas in the sidebar
    * In transform mode, double-click an object to remove it
    * In polygon mode, left-click to add a point, right-click to close the polygon, double-click to remove the latest point
    """
    )
    """
    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="draw_app",
        )

        # Do something interesting with the image data and paths
        #if canvas_result.image_data is not None:
        #    st.image(canvas_result.image_data)
        #if canvas_result.json_data is not None:
        #    objects = pd.json_normalize(canvas_result.json_data["objects"])
        #    for col in objects.select_dtypes(include=["object"]).columns:
        #        objects[col] = objects[col].astype("str")
        #    st.dataframe(objects)

        """



def png_export():
    st.markdown(
        """
    ### Desenhe um número de 0 a 9. 
    """
    )
    #st.markdown(
    #    """
    ##### - clique em Send to Streamlit e depois Export PNG.
    #"""
    #)
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        #st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        st.session_state["button_id"] = re.sub(
            "\d+", "", str(uuid.uuid4()).replace("-", "")
        )
    
    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"
    #st.subheader(file_path)
    #temp = file_path
    #temp = temp.replace('/','_')
    #st.subheader(temp)
    #teste = os.rename(file_path, 'image_number.png')
    
    #file_path = teste
    
    
    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255,255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """
    
    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
            
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()
                      
        if st.button("Teste"):
            dl_link = (
                custom_css
                + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
            )
            st.markdown("#### Clique:")
            st.success("Send to Streamlit ---> Export PNG.")
            st.markdown(dl_link, unsafe_allow_html=True)
        
            convert_tensor = transforms.ToTensor()
            file = file_path
            img = Image.open(file)
            #st.image(img)
            file_tensor = convert_tensor(img)
            #st.write("Imagem na forma de tensor")
            #st.write(file_tensor)
            #st.write(modelo.predict(file_tensor))
            img_28_28 = img.resize([28,28], Image.NEAREST)
            #st.image(img_28_28)
            img_array = np.array(img_28_28)
            img_784 = img_array.reshape(-1,28*28)
            img_784 = img_784.astype('float32')
            img_normalizado = img_784 / 255
            #st.write('Predict img_normalizado')
            #st.write(modelo_keras.predict(img_normalizado))
            #st.subheader((modelo_keras.predict(img_normalizado) > 0.5).astype("int32"))
        
            st.title("Previsões")
            st.subheader("valores:")
            predict_x=modelo_keras.predict(img_normalizado) 
            st.write(predict_x)
            
            classes_x=np.argmax(predict_x,axis=1)
            st.subheader("classes:")
            st.write(classes_x)
            pred = modelo_keras.predict(img_normalizado)
            print(pred.argmax())
            
            pred2 = mnist_keras.predict(img_784.reshape(1, 28, 28, 1))
            print(pred2.argmax())
            
    
        


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
    
    mnist_keras = keras.models.load_model('./mnist_keras.h5')
    
    main()
