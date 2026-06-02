import streamlit as st
from PIL import Image
import numpy as np

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    st.image(img)
    img_array = np.array(img)
    st.write(type(img_array))
    st.write(img_array.shape)
