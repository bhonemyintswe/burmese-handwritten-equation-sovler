import numpy as np
from burmese_handwritten_equ_solver import equ_solver
import streamlit as st
from streamlit_drawable_canvas import st_canvas

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 7)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    drawing_mode='freedraw',
    height=350,
    width=2500,
    key="canvas",
)

img = canvas_result.image_data

def solution(img):

    classes = equ_solver(img)
    classes = np.array(classes)

    equ_string = ""
    equ_list = list(map(str, classes))
    arithmic_index = [10, 11, 12, 13]
    arithmic = ['+', '.', '*', '-']

    for i, val in enumerate(arithmic_index):
        if val in classes:
            indices = np.where(classes == val)[0]
            for j in range(len(indices)):
                equ_list[indices[j]] = arithmic[i]

    equ = equ_string.join(equ_list)
    sol = eval(equ)

    return equ, sol


if st.button('Submit'):
    equation, solution = solution(img)
    st.write(equation)
    st.write(solution)

