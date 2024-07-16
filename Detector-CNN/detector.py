import face_recognition as frg
import pickle as pkl
import os
import cv2
import numpy as np
import yaml
import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from collections import defaultdict

# Load configurations
information = defaultdict(dict)
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

# Load emotion detection model
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprise'}
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("emotion_model1.h5")

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Database functions
def get_database():
    with open(PKL_PATH, 'rb') as f:
        database = pkl.load(f)
    return database

def recognize(image, TOLERANCE):
    # Memuat database wajah yang dikenal
    database = get_database()
    # Mengambil encoding wajah yang dikenal dari database
    known_encoding = [database[id]['encoding'] for id in database.keys()]
    
    # Menginisialisasi variabel name dan id sebagai 'Unknown'
    name = 'Unknown'
    id = 'Unknown'
    
    # Mendeteksi lokasi wajah dalam gambar
    face_locations = frg.face_locations(image)
    # Menghasilkan encoding wajah dari lokasi wajah yang terdeteksi
    face_encodings = frg.face_encodings(image, face_locations)
    
    # Iterasi melalui setiap pasangan lokasi dan encoding wajah
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Membandingkan encoding wajah yang terdeteksi dengan yang dikenal dari database
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        # Menghitung jarak antara encoding wajah yang terdeteksi dan yang dikenal
        distance = frg.face_distance(known_encoding, face_encoding)
        
        # Menginisialisasi variabel name dan id sebagai 'Unknown'
        name = 'Unknown'
        id = 'Unknown'
        
        # Jika ada kecocokan dengan wajah yang dikenal
        if True in matches:
            # Mendapatkan indeks wajah yang cocok
            match_index = matches.index(True)
            # Mengambil nama dan id dari database berdasarkan indeks kecocokan
            name = database[match_index]['name']
            id = database[match_index]['id']
        
        # Menggambar kotak di sekitar wajah yang terdeteksi pada gambar
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Mengembalikan gambar yang telah diproses serta nama dan id yang dikenali
    return image, name, id



# RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Faceemotion class to perform video transformation
class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        # Mengonversi frame video ke array NumPy dengan format BGR
        img = frame.to_ndarray(format="bgr24")
        
        # Mengonversi gambar berwarna ke gambar grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah dalam gambar grayscale
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
        # Memuat database wajah yang dikenal
        database = get_database()
        known_encoding = [database[id]['encoding'] for id in database.keys()]
        
        # Iterasi melalui semua wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Menggambar kotak di sekitar wajah yang terdeteksi
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Mengambil ROI (Region of Interest) wajah dalam gambar grayscale
            roi_gray = img_gray[y:y + h, x:x + w]
            
            # Mengubah ukuran ROI ke 48x48 untuk input ke model emosi
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # Memastikan ROI tidak kosong
            if np.sum([roi_gray]) != 0:
                # Normalisasi ROI dan mengubahnya menjadi array gambar
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Memprediksi emosi dari ROI menggunakan model klasifikasi emosi
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]

                # Mencari encoding wajah dari gambar asli
                face_encodings = frg.face_encodings(img, [(y, x+w, y+h, x)])
                name = 'Unknown'
                id = 'Unknown'
                
                # Jika ada encoding wajah yang terdeteksi
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    # Membandingkan encoding wajah dengan yang ada di database
                    matches = frg.compare_faces(known_encoding, face_encoding, tolerance=0.6)
                    if True in matches:
                        match_index = matches.index(True)
                        name = database[match_index]['name']
                        id = database[match_index]['id']
                
                # Membuat label yang berisi nama dan emosi yang terdeteksi
                label = f"{name} ({finalout})"
                label_position = (x, y)
                
                # Menambahkan label ke gambar
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mengembalikan gambar yang telah diproses
        return img


# Main function to run Streamlit app
def main():
    st.title("Pengenalan Wajah dan Pendeteksi Ekspresi Wajah")
    st.header("Web-cam Live Feed")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=Faceemotion)
    st.subheader("Step by step")
    st.write("""
            Berikut langkah-langkah yang harus diikuti untuk menggunakan aplikasi: 

            1. Mengizinkan akses kamera.
            2. Klik *SELECT DEVICE* untuk memilih device web-cam apa yang ingin digunakan.
            3. Klik *DONE* setelah memilih device.
            4. Klik *START* untuk mengaktifkan web-cam.
            5. Tunggu beberapa saat sampai web-cam ditampilkan.
            6. Setelah web-cam tampil, siapkan ekspresi terbaikmu dan biarkan aplikasi mendeteksinya!
        """)


if __name__ == "__main__":
    main()
