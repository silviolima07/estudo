from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool

import streamlit as st


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
    
    p = figure(x_range=(0, 10), y_range=(0, 10), width=400, height=400)

    renderer = p.multi_line([[1, 1]], [[1, 1]], line_width=1, alpha=0.4, color='red')

    draw_tool = FreehandDrawTool(renderers=[renderer], num_objects=99999)
    p.add_tools(draw_tool)
    p.toolbar.active_drag = draw_tool

    st.bokeh_chart(p)  
    
    
    
if __name__ == '__main__':
    main()
