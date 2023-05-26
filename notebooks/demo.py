import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import torch
import torchvision
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

frame_width = 500
r = 5
st.set_page_config(layout="wide")

def show_mask(image, mask, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = mask.reshape(h,w,1)
    image[mask==True] = (36,255,12)

    # redImg = np.zeros(image.shape, image.dtype)
    # redImg[:,:] = (0, 0, 255)
    # redMask = cv2.bitwise_and(redImg, redImg, mask=mask_image)
    # cv2.addWeighted(redMask, 1, image, 1, 0, image)
    frame_height = int(300 * (h/w))
    image = cv2.resize(image, (300, frame_height), interpolation = cv2.INTER_AREA)
    return image

def segmentation(uploaded_file, points, predictor):
    im = Image.open(uploaded_file).convert('RGB')
    w,h = im.size
    frame_height = frame_width*(h/w)
    input_point = np.array([[(w/frame_width)*points[0][0], (h/frame_height)*points[0][1]]])
    im = np.array(im)
    predictor.set_image(im)
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label,
                                multimask_output=True,)

    captions = []
    images = []
    for i,(mask, score) in enumerate(zip(masks, scores)):
        score = round(score, 2)
        captions.append(f'Mask: {i+1} Score : {score}')
        images.append(show_mask(im, mask))

    return images, captions


def load_model():
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    return predictor

def im_resize(im):
    w,h = im.size
    # frame_width = frame_height*(w/h)
    frame_height = frame_width*(h/w)
    im = im.resize((int(frame_width),int(frame_height)))
    return im

def demo():
    def init_state():
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False

        if 'points' not in st.session_state:
            st.session_state['points'] = []

        if 'sam' not in st.session_state:
            st.session_state['sam'] = load_model()

        return


    st.header("Object Segmentation")
    with st.expander("See Instructions"):
        st.caption("Instructions: ")
        st.markdown("1. If the object needs to be warped, select any coordinate on the object and click on **:red[Segment]** button  \n"
                    "2. Select the **:blue[Reset]** button to clear the coordinate selection")
    init_state()
    im = None
    col3, col4 = st.columns([1,2])
    with col3:
        uploaded_file = st.file_uploader("Upload Image")
        col3_1, col3_2 = st.columns([1,1])
        with col3_1:
            st.text("")
            st.text("")
            segment_bt = st.button('Segment', type = 'primary')
        with col3_2:
            st.text("")
            st.text("")
            clear_points = st.button('Reset', type = 'secondary')
            if clear_points:
                st.session_state['points'] = []
                st.experimental_rerun()

    with col4:
        task_data = st.empty()

    if uploaded_file is not None:
        im = Image.open(uploaded_file)
        im = im_resize(im)
        draw = ImageDraw.Draw(im)
        # print(st.session_state['points'])
        for point in st.session_state['points']:
            coords = [point[0]-r, point[1] -r,
                    point[0]+r, point[1]+r]
            draw.ellipse(coords, fill = 'green')

    if im is not None:
        value = streamlit_image_coordinates(im, key = 'pil')
        if value is not None:
            point = value['x'], value['y']
            if point not in st.session_state['points']:
                st.session_state['points'].append(point)
                st.experimental_rerun()
        
    if segment_bt:
        if len(st.session_state['points'])>1:
            message = st.text('You have selected more than 1 point. Please re-select')
        elif len(st.session_state['points'])<1:
            message = st.text('Please select a point on a object you are interested in')
        else:
            if uploaded_file is None:
                message = st.text('No image found')
            else:
                masks, captions = segmentation(uploaded_file, st.session_state['points'], st.session_state['sam'])
                task_data.image(masks, use_column_width='auto', caption=captions)
                # task_data.image(masks[0])

    return

if __name__ == "__main__":
    demo()
