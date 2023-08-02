import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
# import cv2
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
# import av
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from pycaret.classification import load_model as load_pycaret_model, predict_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model as load_keras_model

# Load your trained model using PyCaret load_model
@st.cache_resource
def set_model():
    loaded_model_efficientnet = load_pycaret_model('tuned_lda_efficientnet2_5folds_0.8513train_0.8194val_512batch_5fold_50epoch_0.2test(LATEST)')
    return loaded_model_efficientnet
# tuned_lda_efficientnet2_5folds_0.8513train_0.8194val_512batch_5fold_50epoch_0.2test(LATEST) / EfficientNet_BatchSize512_25 epochs_test size_0.2encoder (LATEST)
# tuned_et_efficientnet5_extra_0.9129train_0.8191val_256batchsize_10fold_25epoch_0.2test (LATEST) / EfficientNet_BatchSize256_25 epochs_test size_0.2encoder (LATEST)
# tuned_lda_efficientnet_extra_0.82646train_0.8134val_32batch_10fold_50epoch_0.2test (LATEST) / EfficientNet_BatchSize32_50 epochs_test size_0.2encoder (LATEST)

# Load EfficientNetB0 model for feature extraction
base_model_efficientnet = EfficientNetB0(weights='imagenet', include_top=False)

# Use the trained encoder to reduce the dimensionality of the features
@st.cache_resource
def load_encoder():
    h5_encoder = load_keras_model('https://raw.githubusercontent.com/GitUser8888/Predicting-Popularity-in-Korean-Drama-Industry-Based-on-Face-Image/main/streamlit/EfficientNet_BatchSize512_50epochs_testsize_0.2encoder.h5')
    return h5_encoder

encoder = load_encoder()

# Function to preprocess the image for your model
#@st.cache_data
def preprocess_image_efficientnet(image):
    # Resize image
    image = image.resize((224, 224))
    # Convert the image to array
    image = img_to_array(image)
    # Expand dimensions to fit the model's expected input shape
    image = np.expand_dims(image, axis=0)
    # Preprocess the image
    image = preprocess_input(image)
    # Extract features using EfficientNetB0
    features = base_model_efficientnet.predict(image)
    # Flatten features extracted
    efficientnet_feature_flatten = features.reshape(1,-1)
    features_array_efficientnet = np.array(efficientnet_feature_flatten)
    # # Use the trained encoder to reduce the dimensionality of the features
    # encoder = load_keras_model('EfficientNet_BatchSize512_50 epochs_test size_0.2encoder.h5')
    encoded_features = encoder.predict(features_array_efficientnet)
   # Create a DataFrame with the encoded features and labels
    encoded_features_df = pd.DataFrame(encoded_features)    

    return encoded_features_df


# Function to make prediction
def predict(image):
    processed_image = preprocess_image_efficientnet(image)
    prediction = predict_model(set_model(), processed_image, encoded_labels=True)  # use PyCaret's predict_model function
    result = prediction['prediction_label'][0]
    return result


tab1, tab2 = st.tabs(["Main Page", "More Information"])

# Main app
def main():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Photo Upload", "Live Cam"))

    st.sidebar.info('Images and video footage will not be stored. They will only be used for prediction within this app.')
    st.sidebar.info('DISCLAIMER: Contents from this app are for information purposes only, and does not guarantee any outcome or results. This machine learning model is also not 100% accurate. The creator of this app will not be responsible for any damages or losses incurred.')
    st.sidebar.info('Like this app? Reach out to the creator via LinkedIn https://www.linkedin.com/in/jackson-tin/ ')

#    st.title('Potential Popularity Predictor')

    with tab1:
        st.title('Potential Popularity Predictor')
        st.write('I built a machine learning model using face images of South Korean male actors together with their respective popularity level.') 
        st.write('Select from the sidebar to choose if you want to upload an image or use your webcam, then hit the Predict button.')
        st.write('___')

        if add_selectbox == 'Photo Upload':
            st.write('Upload an image of a male face, and this app will predict the potential popularity of the person in the South Korean entertainment (drama) industry based on what the machine learning model has learnt.')
            st.write('\nFor best results, please upload an image with only the **face**, **facing forward**, and **plain background**.')
            st.write('If you get an error, please upload another image.')
            st.divider()

            file = st.file_uploader('Upload an image', type=['jpg', 'png'])

            if file is not None:
                image = Image.open(file)
                st.image(image, caption='Uploaded image.', use_column_width=True)
                
                if st.button('Predict'):
                    with st.spinner('Please wait while the image is being analysed. This may take a few minutes... :timer_clock:'):
                        result1 = predict(image)
                        if result1 == 1:
                            st.write(":tada: That's the face of someone who is or will be super popular! Go for it! :tada:")
                            st.balloons()
                            st.toast('Done!', icon='🎉')
                        elif result1 == 0:
                            st.write('Sorry, based on the image, the machine learning model is not able to recognise the subject as being a potentially popular actor in South Korea. Please upload another image.')
                            st.toast('Done!', icon='🎉')
                        else:
                            st.write('Error! Please try again.')
                        

        if add_selectbox == 'Live Cam':
            picture = st.camera_input("Take a picture")

            if picture:
                image = Image.open(picture)
                st.image(image, caption='Captured image.', use_column_width=True)
                
                if st.button('Predict'):
                    with st.spinner('Please wait while the image is being analysed. This may take a few minutes... :timer_clock:'):
                        result1 = predict(image)
                        if result1 == 1:
                            st.write(":tada: That's the face of someone who is or will be super popular! Go for it! :tada:")
                            st.balloons()
                        elif result1 == 0:
                            st.write('Sorry, based on the image, the machine learning model is not able to recognise the subject as being a potentially popular actor in South Korea. Please upload another image')
                        else:
                            st.write('Error! Please try again.')
                        st.toast('Done!', icon='🎉')

    with tab2:
        st.write("# What's the point of this?")
        st.write('The South Korean entertainment industry is fiercely competitive and burgeoning with potential talents. Current manual scouting methods may not fully capture the breadth of this potential')
        st.write('This project aims to deploy a prototype machine learning model to quickly predict potential success in the South Korean entertainment industry based on face images. The model is trained using face images of a number of very popular Korean male actors and a number of not so popular Korean male actors.')
        st.write('___')
        st.write("# Why do looks matter?")
        st.write('- Looks are not everything, but your visuals make the first impression, more so in the entertainment industry.')
        st.write('- Looks also contribute to the halo effect, where good-looking people are perceived as a good or talented person.')
        st.write('- For other things like acting skills, dancing skills, vocals, these can be trained given a certain amount of time.')
        st.write('___')
        st.write("# Future Work")
        st.write('- This project is done in a limited timeframe of 2 weeks as a prototype, and is **far from perfect**.')
        st.write('- I intend to further train the machine learning model with much more face images and data to improve the prediction. ')
        st.write('- The scope can also be expaned to other regions including Japan. ')

if __name__ == '__main__':
    main()
