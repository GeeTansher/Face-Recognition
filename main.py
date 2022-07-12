# import dependencies

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

from kivy.clock import Clock  # for real time feed
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import glob
import cv2
import face_recognition
import os
import numpy as np
import time

# Building layout


class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .64))
        nameLabel = Label(text="Enter Name:")
        self.textName = TextInput(text="", multiline=False)
        self.button = Button(
            text="ADD FACE", on_press=self.Add, size_hint=(1, .1))
        self.label = Label(text="Recognizing...", size_hint=(1, .1))
        self.label2 = Label(text="", size_hint=(1, .1))

        # Grid layout
        gridLayout = GridLayout(cols=2, size_hint=(1, .06))

        # Add items to layout
        self.layout = BoxLayout(orientation='vertical')
        self.layout.add_widget(self.web_cam)
        self.layout.add_widget(gridLayout)
        self.layout.add_widget(self.button)
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.label2)

        gridLayout.add_widget(nameLabel)
        gridLayout.add_widget(self.textName)

        self.load_encoding_images("images/")

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return self.layout

    # To show frame as image
    def imgShow(self, frame):
        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Run continuously to get webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        # Detect Faces
        face_locations, face_names = self.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

            # # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35),
            #               (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, name, (left + 6, bottom + 30),
                        font, 1.0, (0, 0, 255), 1)

        self.imgShow(frame)
        self.label.text = "Faces: {}".format(face_names)

    # To add image
    def Add(self, frame):
        ret, frame = self.capture.read()
        face_locations, face_names = self.detect_known_faces(frame)
        if len(face_locations) == 1:
            if self.textName.text != "":
                name = "{}.jpg".format(self.textName.text)
                PATH = os.path.join("images", name)
                cv2.imwrite(PATH, frame)
                self.load_encoding_images("images/")
                self.label2.text = "Ready for recognizing again..."
                self.textName.text = ""
            else:
                self.label2.text = "Please provide a name in the text field."
        else:
            self.label2.text = "Please get only or at least one face to be added."

    # Load Encoding Images
    def load_encoding_images(self, images_path):
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        Logger.info("{} encoding images found.".format(len(images_path)))
        known_face_encodings.clear()
        known_face_names.clear()

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            known_face_encodings.append(img_encoding)
            known_face_names.append(filename)
        Logger.info("Encoding images loaded")

    # Detect faces function
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the image from BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # Get distance between 2 images and get the lowest
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names


if __name__ == '__main__':
    # Initialize
    known_face_names = []
    known_face_encodings = []
    CamApp().run()
