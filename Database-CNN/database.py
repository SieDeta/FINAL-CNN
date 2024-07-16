import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import yaml
import pandas as pd
import cv2
import numpy as np
from utils import submitNew, get_info_from_id, deleteOne

def main():
    st.set_page_config(layout="wide")
    st.title("Database App")

    selected = option_menu(None, ["Updating", "Database", "Download"], 
                           icons=['house', 'camera', "info-circle"], 
                           menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == "Updating":
        st.header("Updating")
        menu = ["Adding", "Deleting", "Adjusting"]
        choice = st.sidebar.selectbox("Options", menu)
        if choice == "Adding":
            name = st.text_input("Name", placeholder='Enter name')
            id = st.text_input("ID", placeholder='Enter id')
            upload = st.radio("Upload image or use webcam", ("Upload", "Webcam"))
            if upload == "Upload":
                uploaded_image = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
                if uploaded_image is not None:
                    st.image(uploaded_image)
                    submit_btn = st.button("Submit", key="submit_btn")
                    if submit_btn:
                        if name == "" or id == "":
                            st.error("Please enter name and ID")
                        else:
                            ret = submitNew(name, id, uploaded_image)
                            if ret == 1:
                                st.success("Student Added")
                            elif ret == 0:
                                st.error("Student ID already exists")
                            elif ret == -1:
                                st.error("There is no face in the picture")
            elif upload == "Webcam":
                img_file_buffer = st.camera_input("Take a picture")
                submit_btn = st.button("Submit", key="submit_btn")
                if img_file_buffer is not None:
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if submit_btn:
                        if name == "" or id == "":
                            st.error("Please enter name and ID")
                        else:
                            ret = submitNew(name, id, cv2_img)
                            if ret == 1:
                                st.success("Student Added")
                            elif ret == 0:
                                st.error("Student ID already exists")
                            elif ret == -1:
                                st.error("There is no face in the picture")
        elif choice == "Deleting":
            def del_btn_callback(id):
                deleteOne(id)
                st.success("Student deleted")
                
            id = st.text_input("ID", placeholder='Enter id')
            submit_btn = st.button("Submit", key="submit_btn")
            if submit_btn:
                name, image, _ = get_info_from_id(id)
                if name == None and image == None:
                    st.error("Student ID does not exist")
                else:
                    st.success(f"Name of student with ID {id} is: {name}")
                    st.warning("Please check the image below to make sure you are deleting the right student")
                    st.image(image)
                    del_btn = st.button("Delete", key="del_btn", on_click=del_btn_callback, args=(id,))
        
        elif choice == "Adjusting":
            def form_callback(old_name, old_id, old_image, old_idx):
                new_name = st.session_state['new_name']
                new_id = st.session_state['new_id']
                new_image = st.session_state['new_image']
                
                name = old_name
                id = old_id
                image = old_image
                
                if new_image is not None:
                    image = cv2.imdecode(np.frombuffer(new_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    
                if new_name != old_name:
                    name = new_name
                    
                if new_id != old_id:
                    id = new_id
                
                ret = submitNew(name, id, image, old_idx=old_idx)
                if ret == 1:
                    st.success("Student Added")
                elif ret == 0:
                    st.error("Student ID already exists")
                elif ret == -1:
                    st.error("There is no face in the picture")
            id = st.text_input("ID", placeholder='Enter id')
            submit_btn = st.button("Submit", key="submit_btn")
            if submit_btn:
                old_name, old_image, old_idx = get_info_from_id(id)
                if old_name == None and old_image == None:
                    st.error("Student ID does not exist")
                else:
                    with st.form(key='my_form'):
                        st.title("Adjusting student info")
                        col1, col2 = st.columns(2)
                        new_name = col1.text_input("Name", key='new_name', value=old_name, placeholder='Enter new name')
                        new_id = col1.text_input("ID", key='new_id', value=id, placeholder='Enter new id')
                        new_image = col1.file_uploader("Upload new image", key='new_image', type=['jpg', 'png', 'jpeg'])
                        col2.image(old_image, caption='Current image', width=400)
                        st.form_submit_button(label='Submit', on_click=form_callback, args=(old_name, id, old_image, old_idx))

    elif selected == "Database":
        st.header("Database")
        cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        PKL_PATH = cfg['PATH']["PKL_PATH"]

        with open(PKL_PATH, 'rb') as file:
            database = pickle.load(file)

        Index, Id, Name, Image = st.columns([0.5, 0.5, 3, 3])

        for idx, person in database.items():
            with Index:
                st.write(idx)
            with Id:
                st.write(person['id'])
            with Name:
                st.write(person['name'])
            with Image:
                st.image(person['image'], width=200)

    elif selected == "Download":
        st.header("Download")
        cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        PKL_PATH = cfg['PATH']["PKL_PATH"]
        
        with open(PKL_PATH, 'rb') as file:
            database = pickle.load(file)
        
        st.download_button(label="Download Database",
                           data=pickle.dumps(database),
                           file_name='database.pkl',
                           mime='application/octet-stream')

if __name__ == "__main__":
    main()