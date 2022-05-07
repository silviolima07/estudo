import streamlit as st

from bokeh.models.widgets import Div

import pandas as pd


pd.set_option('precision',2)


import sys

import time





def main():


    # Titulo do web app
    #html_page = """
    #<div style="background-color:blue;padding=30px">
    #    <p style='text-align:center;font-size:30px;font-weight:bold;color:white'>Indeed</p>
    #</div>
    #          """
    #st.markdown(html_page, unsafe_allow_html=True)
   
    html_page = """
    <div style="background-color:white;padding=36px">
        <p style='text-align:center;font-size:36px;font-weight:bold;color:red'>Web scrap de Estabelecimentos Comerciais</p>
    </div>
              """
    st.markdown(html_page, unsafe_allow_html=True)
    
    
    
if __name__ == '__main__':
    main()
