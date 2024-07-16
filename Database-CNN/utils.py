import face_recognition as frg
import pickle as pkl 
import os 
import cv2 
import numpy as np
import yaml
from collections import defaultdict

# Inisialisasi informasi dan memuat konfigurasi dari file YAML
information = defaultdict(dict)
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

# Memuat database dari file PKL
def get_database():
    with open(PKL_PATH, 'rb') as f:
        database = pkl.load(f)
    return database

# Memeriksa apakah ada wajah dalam gambar
def isFaceExists(image): 
    face_location = frg.face_locations(image)
    if len(face_location) == 0:
        return False
    return True

# Menambahkan wajah baru ke database
def submitNew(name, id, image, old_idx=None):
    database = get_database()
    
    # Membaca gambar jika tidak dalam format np.ndarray
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)

    if not isFaceExists(image):
        return -1
    
    # Menghasilkan encoding wajah
    encoding = frg.face_encodings(image)[0]
    
    # Memeriksa apakah ID sudah ada
    existing_id = [database[i]['id'] for i in database.keys()]
    
    # Mode pembaruan
    if old_idx is not None: 
        new_idx = old_idx
    # Mode penambahan
    else: 
        if id in existing_id:
            return 0
        new_idx = len(database)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    database[new_idx] = {'image': image, 'id': id, 'name': name, 'encoding': encoding}
    
    # Menyimpan database yang diperbarui
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    
    return True

# Mendapatkan informasi berdasarkan ID
def get_info_from_id(id): 
    database = get_database() 
    for idx, person in database.items(): 
        if person['id'] == id: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None

# Menghapus entri dari database berdasarkan ID
def deleteOne(id):
    database = get_database()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    
    # Menyimpan database yang diperbarui
    with open(PKL_PATH, 'wb') as f:
        pkl.dump(database, f)
    
    return True

# Membuat dataset dari direktori gambar
def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR, image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        
        if not image_path.endswith('.jpg'):
            continue
        
        image = frg.load_image_file(image_path)
        information
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    # Menyimpan informasi dataset ke file PKL
    with open(os.path.join(DATASET_DIR, 'database.pkl'), 'wb') as f:
        pkl.dump(information, f)

if __name__ == "__main__": 
    deleteOne(4)