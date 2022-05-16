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

from scipy.misc import imread,imresize

import cv2

import keras.models
from keras.models import model_from_json

def carregar_modelo(): 
	json_file = open('./model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("./model.h5")
	print("Modelo Carregado do Disco")

	# Compila e Avalia o Modelo Carregado
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#graph = tf.get_default_graph()

	return loaded_model #, graph

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

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
        #if canvas_result.image_data is not None:
            #st.image(canvas_result.image_data)
        #if canvas_result.json_data is not None:
        #    objects = pd.json_normalize(canvas_result.json_data["objects"])
        #    for col in objects.select_dtypes(include=["object"]).columns:
        #        objects[col] = objects[col].astype("str")
        #     st.dataframe(objects)
        
        
        if st.button("Prever") and canvas_result.image_data is not None:
            img = canvas_result.image_data
        
            st.image(img)
            
            # Get the numpy array (4-channel RGBA 100,100,4)
            input_numpy_array = np.array(canvas_result.image_data)
     
     
            # Get the RGBA PIL image
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            input_image.save('user_input.png')
            
            # Convert it to grayscale
            input_image_gs = input_image.convert('L')
            input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
            all_zeros = not np.any(input_image_gs_np)
            if not all_zeros:
                # st.write('### Image as a grayscale Numpy array')
                # st.write(input_image_gs_np)
         
                # Create a temporary image for opencv to read it
                input_image_gs.save('temp_for_cv2.jpg')
                image = cv2.imread('temp_for_cv2.jpg', 0)
                # Start creating a bounding box
                height, width = image.shape
                x,y,w,h = cv2.boundingRect(image)
 
 
                # Create new blank image and shift ROI to new coordinates
                ROI = image[y:y+h, x:x+w]
                mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
                width, height = mask.shape
    #     print(ROI.shape)
    #     print(mask.shape)
                x = width//2 - ROI.shape[0]//2
                y = height//2 - ROI.shape[1]//2
    #           print(x,y)
                mask[y:y+h, x:x+w] = ROI
    #     print(mask)
        # Check if centering/masking was successful
    #     plt.imshow(mask, cmap='viridis') 
                output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
        # Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
        # compressed_output_image = output_image.resize((22,22))
        # Therefore, we use the following:
                compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good
 
                tensor_image = np.array(compressed_output_image.getdata())/255.
                tensor_image = tensor_image.reshape(22,22)
        # Padding
                tensor_image = np.pad(tensor_image, (3,3), "constant", constant_values=(0,0))
        # Normalization should be done after padding i guess
                tensor_image = (tensor_image - 0.1307) / 0.3081
        # st.write(tensor_image.shape) 
        # Shape of tensor image is (1,28,28)
         
 
 
        # st.write('### Processing steps:')
        # st.write('1. Find the bounding box of the digit blob and use that.')
        # st.write('2. Convert it to size 22x22.')
        # st.write('3. Pad the image with 3 pixels on all the sides to get a 28x28 image.')
        # st.write('4. Normalize the image to have pixel values between 0 and 1.')
        # st.write('5. Standardize the image using the mean and standard deviation of the MNIST_plus dataset.')
 
        # The following gives noisy image because the values are from -1 to 1, which is not a proper image format
        # im = Image.fromarray(tensor_image.reshape(28,28), mode='L')
        # im.save("processed_tensor.png", "PNG")
        # So we use matplotlib to save it instead
                plt.imsave('processed_tensor.png',tensor_image.reshape(28,28), cmap='gray')
 
        # st.write('### Processed image')
        # st.image('processed_tensor.png')
        # st.write(tensor_image.detach().cpu().numpy().reshape(28,28))
 
 
        ### Compute the predictions
                output_probabilities = modelo.predict(tensor_image.reshape(1,784).astype(np.float32))
                prediction = np.argmax(output_probabilities)
 
         
 
            
            #im = Image.fromarray(img)
            #im.save("output.png")
            
            #input_image = image.read(img)            
            # Encode em formato que possa ser alimentado no modelo 
            #input_image.save("output.png")
            
            # Grava a imagem na memória
            #x = imread('output.png', mode='L')
            # Calcula uma inversão bit-wise onde preto torna-se branco e vice-versa
            #x = np.invert(x)
            # Redimensiona a imagem para o tamanho que será alimentado no modelo
            #x = imresize(x,(28,28))
            #x = Image.fromarray(x).resize(size=(28, 28))
            # Converte para um tensor 4D e alimenta nosso modelo
            #x = x.reshape(1,28,28,1)
            # Faz a previsão
            #out = model.predict(x)
		    #print(np.argmax(out, axis=1))
		    # Converte a resposta em uma string
		    #response = np.array_str(np.argmax(out,axis=1))
            #st.title("Previsão")
            #pred = modelo_keras.predict(x)
            #st.title(pred.argmax())
            
            
            
        
            # Get the numpy array (4-channel RGBA 100,100,4)
            #input_numpy_array = np.array(img)
        
            # Get the numpy array (4-channel RGBA 100,100,4)
            #input_numpy_array = np.array(canvas_result.image_data)
            
            #img_28_28 = img.resize((28,28), Image.NEAREST)
            
            #img = cv.resize(img , (28,28))
            #resized = cv2.resize(input_numpy_array, (28,28)) 
            #features = resized.reshape(1,-1)
            
            #st.image(img)
            #img_array = np.array(input_numpy_array.resize((28, 28), Image.LANCZOS))
     
            #img_teste = features.astype('float32')
            #img_teste = img_teste / 255
        
            # Get the RGBA PIL image
            #input_image = Image.fromarray(img_teste.astype('uint8'), 'RGBA')
            
     
     
            #st.write("Input Image")
            #st.image(input_image)
        
        
            #img_28_28 = input_image.resize((28,28), Image.NEAREST)
            #img_28_28 = np.array(img_pil.resize((28, 28), Image.LANCZOS))
            #img_28_28 = img_pil.resize(size=(28, 28),Image.LANCZOS)
        
            #st.write("Input Image resized to 28x28")
            #st.image(img_28_28)
            
            
            
            
            #img_teste = img_28_28.reshape(1, 28, 28, 1).astype('float32')
            
            #image2 = input_image.resize((22,22), Image.LANCZOS) 
            
            #st.write("Input Image resized to 22x22")
            #st.image(image2)
        
            #img_teste = img_28_28.resize(784, 784)
            #img_teste = img_teste.astype('float32')
            
            #st.write(img_784)
            
            #img_normalizado = img_teste/255.0
            
        
            #st.title("Previsão")
                 
            #pred = modelo_keras.predict(img_teste)
        
            #st.title(pred.argmax())
        






if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    
    modelo = carregar_modelo()
    
    main()