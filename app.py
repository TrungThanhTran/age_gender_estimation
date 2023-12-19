import streamlit as st
import numpy as np
import logging
import json
import time
import cv2
import torch
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
import gc


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

CONFIGURE =  {
    "detector_weights":"models/yolov8x_person_face.pt",
    "checkpoint": "models/model_imdb_cross_person_4.22_99.46.pth.tar",
    "with_persons": False,
    "draw": True,
    "device": "cpu",
    "disable_faces": False
}
CONFIGURE = dotdict(CONFIGURE)
print(CONFIGURE.detector_weights)

def prepare_model():
    return Predictor(CONFIGURE, verbose=True)

def main():
    predictor = prepare_model()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    uploaded_file = st.file_uploader("Upload an image")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        st.image(img)

    detect_age = st.button("Age estimation")
    if detect_age:
        if uploaded_file is not None:
            # Convert the file to an opencv image.                
            start = time.time()

            detected_objects, out_im = predictor.recognize(img)
            if CONFIGURE.draw:
                filename = "out_test.jpg"
                cv2.imwrite(filename, out_im)
            
            st.image(out_im)
            st.markdown(f'running time = {time.time() - start:.2f} seconds')
        else:
            st.warning("Please upload an image")
        gc.collect()
        
if __name__ == "__main__":
    main()
