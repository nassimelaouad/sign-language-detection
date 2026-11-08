import json
import pickle
import threading
import requests
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr
from streamlit_lottie import st_lottie
from keras.models import load_model  # TensorFlow is required for Keras to work
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Zsign",page_icon='../assets/ZsignLogo-removebg-preview.png',layout="wide")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
st.markdown("<h1 style='text-align: center;margin-bottom:20px;'>Z-sign : Sign Language Detection - English - Darija Morocco </h1>", unsafe_allow_html=True)
app_mode = option_menu(
        menu_title=None,
        options=["About App", "Sign to Text", "Text to sign", "Speech", "write words", "Video"],
        icons=["house", "justify", "chat", "cassette", "file-earmark-word", ""],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#683db7", "font-size": "25px"},
            "nav-link-selected": {"background-color": "#537cd5"},
            "option":{"color":"black"}
        },
    )
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)


my_list = [] 



st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def resiezeImage(img,new_width,new_height) :
    original_image = Image.open(img)
    # Obtenir les dimensions originales de l'image
     # Redimensionner l'image
    resized_image = original_image.resize((new_width, new_height))
    return resized_image
           



logo='../assets/ZsignLogo-removebg-preview.png'
st.sidebar.image(resiezeImage(logo,300,230))
st.sidebar.title('Sign Language Detection - English - Darija Morocco ')



app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','write words using sign language','Text to sign Language','Speech to sign Language']
)
#! About App
if app_mode =='About App':
     
  
    st.markdown('<p style="font-size: 25px;">To facilitate more inclusive and barrier-free communication between hearing and deaf individuals, contributing to a more equitable and understanding society.</p>', unsafe_allow_html=True)


    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    video_path = '../assets/production_id_5211959 (2160p).mp4'
    st.video(video_path)


    lottie_coding = load_lottiefile('../json/aboutApp.json')
    t1,t2=st.columns(2)
    with t1:
        st.markdown('<h1 style="color:#FCAEAE;">About App</h1>', unsafe_allow_html=True)
        st.write("##")
        st.write(
            """
            Welcome to our innovative Sign Language Translation App :
            - our application excels in translating sign language gestures into both English and Moroccan Darija text.
            - dedicated to bridging the communication gap between Moroccan Darija and English! Leveraging the power of Python, TensorFlow, and MediaPipe
            - Our app's versatility extends to translating written text in both languages into corresponding sign language gestures, fostering inclusivity and breaking language barriers
            - Embracing SpeechRecognition technology, we empower vocal translations into sign language, facilitating seamless spoken communication representation visually.
            - Sign into Text and Video into Text

            """
        )
        st.write("[For more information Explore Our GitHup Repository ](https://github.com/Chigaw/Z-Sign---Sign-Language-Darija-/blob/main/Tensorflow%20Setup%20for%20beginners.md)")
    with t2:
        st_lottie(lottie_coding, height=600, key="coding")


    
    
    features=load_lottiefile('../json/key1.json')
    t5,t6=st.columns(2)
    with t5:
        st.markdown('<h1 style="color:#AAC8A7;">Key Features</h1>', unsafe_allow_html=True)
        st.write("##")
        st.write(
            """
            - An intuitive user interface with visual instructions to guide users.
            - Uses a camera to detect gestures and signs made by the user.
            - Incorporates sign recognition algorithms to interpret gestures in  Sign Language.
            - Converts detected signs into displayed text and/or vocal output for bidirectional communication.
            """
        )
    with t6:
        st_lottie(features, height=400, key="features") 

     
    
    features1=load_lottiefile('../json/fetures.json')
    t7, t8 = st.columns(2) 
    with t7:
        st.title("Target Users")
        st.write("##")
        st.write(
            """
            - Deaf or hard-of-hearing individuals.
            - Friends, family members, or colleagues of deaf or hard-of-hearing individuals .
            - Educational Institutions .
            - organizations dedicated to deaf and mute individuals . 
            - Healthcare Professionals, such as doctors, nurses, therapists, and caregivers .
            - Professional Organizations . 
            """
        )
    with t8:
        st_lottie(features1, height=400, key="features1")

#! Sign Language to Text 
elif app_mode == 'Sign Language to Text':


    #st.markdown("<h1 style='text-align: center;'>Z-sign : Sign Language Detection - English - Darija Morocco </h1>", unsafe_allow_html=True)

    usine=load_lottiefile('../assets/convert.json')
    st.header("Text to Sign Language")
    t12,t13=st.columns((1,0.2))
    with t12:   
        st.header('Converting text to sign language involves translating written text into corresponding signs or gestures in a sign language system')
        st.subheader('Choose the language')
    with t13:
        st_lottie(usine, height=200, key="coding")
    
        

    langage_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])
    
   
    def EnglishSign():
    
        
        st.title('English sign language ')

        np.set_printoptions(suppress=True)
        model_dict = pickle.load(open('../model/data_words.pickle','rb'))
        model = model_dict['model']

        cap = cv2.VideoCapture(0)
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240) 

        st.subheader('Click the button below to activate the camera.')
        btn_start_camera = st.button("Activate the camera")

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Dictionary to map class indices to labels
        labels_dict = {0: 'hello', 1: 'our goal', 2: 'is to help', 3: 'deaf people'}

        finalphrase=""
        if btn_start_camera:
            st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")
            current_word = ""
            
            cooldown_start_time = time.time()
            cooldown_duration = 2.0  # Set the cooldown duration in seconds
            placeholder = st.empty()
            finalpred= []
            while btn_start_camera: 
                # Keep running as long as the camera is activated
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data_aux = []
                    x_ = []
                    y_ = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data_aux = data_aux + [0] * (84 - len(data_aux))

                    x1 = int(min(x_) * frame.shape[1]) - 10
                    y1 = int(min(y_) * frame.shape[0]) - 10
                    x2 = int(max(x_) * frame.shape[1]) - 10
                    y2 = int(max(y_) * frame.shape[0]) - 10

                    # Check if cooldown period has passed
                    if time.time() - cooldown_start_time >= cooldown_duration:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]
                        finalpred.append(predicted_character)
                        # Display the prediction on the Streamlit page
                        #st.write(f"""{predicted_character}""")
                       
                        # Update the current word
                        current_word += predicted_character
                        finalphrase += predicted_character+" "
                        
                        if len(finalpred)>4 :
                            st.write(f"{finalphrase} ")
                            break
                        #st.write(f"{finalphrase} ")
                        # Reset cooldown timer
                        cooldown_start_time = time.time()

                    placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    
                    


            
            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        
        if st.button("Stop camera"):
            
            cap.release()

        

    def DarijaSign():
                st.subheader('Darija sign language')
                np.set_printoptions(suppress=True)
                model_dict = pickle.load(open('../model/data_words.pickle','rb'))
                model = model_dict['model']

                cap = cv2.VideoCapture(0)
                cap.set(3, 320)  # Set width to 320
                cap.set(4, 240) 

                st.subheader('Click the button below to activate the camera.')
                btn_start_camera = st.button("Activate the camera")

                # Initialize MediaPipe Hands
                mp_hands = mp.solutions.hands
                mp_drawing = mp.solutions.drawing_utils
                hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

                # Dictionary to map class indices to labels
                labels_dict = {0: 'سلام', 1: 'الهدف ديالنا', 2: 'نعاونو', 3: 'الناس لمكتسمعش'}
              
               
                if btn_start_camera:
                    st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")
                    current_word = ""
                    cooldown_start_time = time.time()
                    cooldown_duration = 2.0  # Set the cooldown duration in seconds
                    placeholder = st.empty()

                    while btn_start_camera:  # Keep running as long as the camera is activated
                        ret, frame = cap.read()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            data_aux = []
                            x_ = []
                            y_ = []
                            for hand_landmarks in results.multi_hand_landmarks:
                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    x_.append(x)
                                    y_.append(y)

                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    data_aux.append(x - min(x_))
                                    data_aux.append(y - min(y_))

                            data_aux = data_aux + [0] * (84 - len(data_aux))

                            x1 = int(min(x_) * frame.shape[1]) - 10
                            y1 = int(min(y_) * frame.shape[0]) - 10
                            x2 = int(max(x_) * frame.shape[1]) - 10
                            y2 = int(max(y_) * frame.shape[0]) - 10

                            # Check if cooldown period has passed
                            if time.time() - cooldown_start_time >= cooldown_duration:
                                prediction = model.predict([np.asarray(data_aux)])
                                predicted_character = labels_dict[int(prediction[0])]

                                # Display the prediction on the Streamlit page
                                st.write(f"{predicted_character}")
                            
                                # Update the current word
                                current_word += predicted_character

                                # Reset cooldown timer
                                cooldown_start_time = time.time()

                            placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                        cv2.LINE_AA)

                    # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
                if st.button("Stop camera"):
                    cap.release()          

    if langage_mode=='Darija Morroco' : 
        DarijaSign()
       
    elif langage_mode=='English':
        EnglishSign()

        
#! Text to sign Language
elif app_mode == 'Text to sign Language':
    #st.markdown("<h1 style='text-align: center;'>Z-sign : Sign Language Detection - English - Darija Morocco </h1>", unsafe_allow_html=True)
    
    st.header('Text to Sign Language ')
    usine=load_lottiefile('convert.json')
    t12,t13=st.columns((1,0.2))
    with t12:   
        st.header('Converting text to sign language involves translating written text into corresponding signs or gestures in a sign language system')
        st.subheader('Choose the language')
    with t13:
        st_lottie(usine, height=200, key="coding")


   
    txt_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])

    # define function to display sign language images
    def EnglishText():
        # get the file path of the images directory
        st.title('English text to English Sign language')
        img_dir = "photo/"
        text = st.text_input("Enter text:")
        text = text.lower()
        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.jpeg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    def DarijaText():
        st.title('Darija text to Darija Sign language')
        # get the file path of the images directory
        img_dir = "photoArab/"
        text = st.text_input("Enter text:")
        text = text.lower()
        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.JPG")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

    
    # convert text to lowercase
    if txt_mode=='English' :
        # display sign language images
        EnglishText()
    else:
        DarijaText()
        
#! Speech to sign Language    
elif app_mode == 'Speech to sign Language':

    st.title('Speech to Sign Language ')
    # initialize the speech recognition engine
    r = sr.Recognizer()
    st.header('Converting a speech into sign language involves translating spoken words and expressions into visual gestures and movements. ')
    st.subheader('Choose the language')
    txt_mode=st.selectbox("Pick one", ["Darija Morroco", "English"])

    # define function to display sign language images
    def display_English_images(text):
        # get the file path of the images directory
        img_dir = "photo/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.jpeg")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

        # define function to display sign language images
    def display_Arabic_images(text):
    # get the file path of the images directory
        img_dir = "photoArab/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display images for Arabic characters
        for char in text:
            if char.isalpha():
                # display image for the Arabic alphabet
                img_path = os.path.join(img_dir, f"{char}.JPG")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(1.5)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()

    # add start button to start recording audio
    if st.button("Start Talking"):
        if txt_mode=='English':
        # record audio for 5 seconds
            with sr.Microphone() as source:
                st.write("Say something!")
                audio = r.listen(source, phrase_time_limit=5)

                try:
                    text = r.recognize_google(audio)
                except sr.UnknownValueError:
                    st.write("Sorry, I did not understand what you said.")
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition service; {e}")
            st.write(f"You said: {text}", font_size=41)
       
            text = text.lower()
            # display sign language images
            display_English_images(text)

        else:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Say something!")
                audio = r.listen(source, phrase_time_limit=5)

                try:
                    # transcribe the speech to text using Google Speech Recognition
                    text = r.recognize_google(audio, language='ar')
                except sr.UnknownValueError:
                    st.write("Sorry, I did not understand what you said.")
                except sr.RequestError as e:
                    st.write(f"Could not request results from Google Speech Recognition service; {e}")

            # convert text to lowercase
            text = text.lower()

            # display the final result
            st.write(f"You said: {text}", font_size=41)

            # display sign language images for Arabic text
            display_Arabic_images(text)

#! Write words using sign language          
else : 
    #st.markdown("<h1 style='text-align: center;'>Z-sign : Sign Language Detection - English - Darija Morocco </h1>", unsafe_allow_html=True)
    st.title("Choose your language")
    txt_mode_alph=st.selectbox("Pick one", ["Darija Morroco", "English"])

    def horofenglishSign():

        st.title('English alphabet ')
        st.subheader('English sign language alphabet')
        st.image('../assets/_American Sign Language ASL Alphabet_ Photographic Print for Sale by SteamerTees.jpeg')
        np.set_printoptions(suppress=True)
        model_dict = pickle.load(open('../model/data_anglais.pickle','rb'))
        model = model_dict['model']

        cap = cv2.VideoCapture(0)
        cap.set(3, 320)  # Set width to 320
        cap.set(4, 240) 

        st.subheader('Click the button below to activate the camera and start facial recognition.')
        btn_start_camera = st.button("Activate camera")

        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Dictionary to map class indices to labels
        labels_dict = {0: 'H', 1: 'E', 2: 'L', 3: 'L', 4: 'O'}

        if btn_start_camera:
            st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")
            current_word = ""
            cooldown_start_time = time.time()
            cooldown_duration = 2.0  # Set the cooldown duration in seconds
            placeholder = st.empty()

            while btn_start_camera:  # Keep running as long as the camera is activated
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data_aux = []
                    x_ = []
                    y_ = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data_aux = data_aux + [0] * (84 - len(data_aux))
                    expected_features = 42
                    data_aux = data_aux[:expected_features] 

                    x1 = int(min(x_) * frame.shape[1]) - 10
                    y1 = int(min(y_) * frame.shape[0]) - 10
                    x2 = int(max(x_) * frame.shape[1]) - 10
                    y2 = int(max(y_) * frame.shape[0]) - 10

                    # Check if cooldown period has passed
                    if time.time() - cooldown_start_time >= cooldown_duration:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        # Display the prediction on the Streamlit page
                        st.write(f"{predicted_character}")
                       
                        # Update the current word
                        current_word += predicted_character

                        # Reset cooldown timer
                        cooldown_start_time = time.time()

                    placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

            # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
        if st.button("Stop camera"):
            cap.release()


    def horofarabicSign():
            st.title('Arabic alphabet')
            st.subheader('arabic sign language alphabet')
            image_path = '../assets/arabicHorof1.jpg'
            resized_image = st.image(image_path, width=400)
            model_dict = pickle.load(open('../model/data_arab.pickle', 'rb'))
            model = model_dict['model']

            cap = cv2.VideoCapture(0)
            cap.set(3, 320)  # Set width to 320
            cap.set(4, 240) 

            st.subheader('Click the button below to activate the camera and start facial recognition.')
            btn_start_camera = st.button("Activate camera")

            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

            # Dictionary to map class indices to labels
            labels_dict = {0: 'ا', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح', 6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر'}

            if btn_start_camera:
                st.warning("The camera is activated. To stop, click on the 'Stop Camera' button.")
                current_word = ""
                cooldown_start_time = time.time()
                cooldown_duration = 2.0  # Set the cooldown duration in seconds
                placeholder = st.empty()

                while btn_start_camera:  # Keep running as long as the camera is activated
                    ret, frame = cap.read()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        data_aux = []
                        x_ = []
                        y_ = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                data_aux.append(x - min(x_))
                                data_aux.append(y - min(y_))

                        data_aux = data_aux + [0] * (84 - len(data_aux))

                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        x2 = int(max(x_) * frame.shape[1]) - 10
                        y2 = int(max(y_) * frame.shape[0]) - 10

                        # Check if cooldown period has passed
                        if time.time() - cooldown_start_time >= cooldown_duration:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict[int(prediction[0])]

                            # Display the prediction on the Streamlit page
                            st.write(f"{predicted_character}")

                            # Update the current word
                            current_word += predicted_character

                            # Reset cooldown timer
                            cooldown_start_time = time.time()

                        placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, current_word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                    cv2.LINE_AA)

                # Arrêter la capture vidéo lorsque l'utilisateur clique sur "Arrêter la caméra"
            if st.button("Stop camera"):
                cap.release()
    
    
    
     # convert text to lowercase
    if txt_mode_alph=='English' :
        # display sign language images
        horofenglishSign()
    else:
        horofarabicSign()

#! Video To Sign Language
