# Importing Libraries
import numpy as np                 # For numerical operations and array handling
import math                        # For mathematical functions like sqrt
import cv2                         # OpenCV library for computer vision tasks

import os, sys                     # Operating system interfaces and system-specific parameters
import traceback                   # For printing exception traceback information
import pyttsx3                     # Text-to-speech conversion library
from keras.models import load_model # For loading the pre-trained neural network model
from cvzone.HandTrackingModule import HandDetector # For hand tracking and detection
from string import ascii_uppercase # To get all uppercase English letters
import enchant                     # For spell checking and word suggestions
ddd = enchant.Dict("en-US")        # Initialize English-US dictionary for spell checking
hd = HandDetector(maxHands=1)      # Initialize first hand detector instance (for main camera)
hd2 = HandDetector(maxHands=1)     # Initialize second hand detector instance (for cropped hand image)
import tkinter as tk               # GUI toolkit for creating the application interface
from PIL import Image, ImageTk     # For image processing in the GUI

offset = 29                        # Pixel offset for hand bounding box padding

# Configure Theano to use GPU if available
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


# Application class definition
class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)          # Initialize webcam (camera index 0)
        self.current_image = None               # Variable to store the current camera frame
        self.model = load_model('cnn8grps_rad1_model.h5')  # Load the trained CNN model
        self.speak_engine = pyttsx3.init()      # Initialize text-to-speech engine
        self.speak_engine.setProperty("rate", 100)  # Set speech rate (words per minute)
        voices = self.speak_engine.getProperty("voices")  # Get available voices
        self.speak_engine.setProperty("voice", voices[0].id)  # Set the first voice (usually male)

# Counters and flags for character detection
        self.ct = {}                # Dictionary to count occurrences of each character
        self.ct['blank'] = 0        # Initialize blank character counter
        self.blank_flag = 0         # Flag for blank detection
        self.space_flag = False     # Flag for space character detection
        self.next_flag = True       # Flag to handle the "next" gesture (confirm character)
        self.prev_char = ""         # Store the previous detected character
        self.count = -1             # Counter for tracking detections
        self.ten_prev_char = []     # List to store the 10 most recent detected characters
        for i in range(10):         # Initialize the list with spaces
            self.ten_prev_char.append(" ")

        # Initialize counters for all uppercase letters
        for i in ascii_uppercase:   # Create counter for each letter A-Z
            self.ct[i] = 0
        print("Loaded model from disk")

# Initialize the GUI window
        self.root = tk.Tk()                      # Create the main Tkinter window
        self.root.title("Sign Language To Speech Conversion")  # Set window title
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)  # Set close window handler
        self.root.geometry("1300x700")           # Set window size
        self.root.configure(bg="#add8e6")  # Light blue background

        self.panel = tk.Label(self.root, bg="#add8e6")  # Main video feed panel
        self.panel.place(x=100, y=3, width=480, height=640)

        self.panel2 = tk.Label(self.root, bg="#add8e6")   # Hand gesture visualization panel
        self.panel2.place(x=700, y=115, width=400, height=400)

        self.T = tk.Label(self.root, text="Sign Language To Speech Conversion", font=("Courier", 30, "bold"), bg="#add8e6")  # Title label
        self.T.place(x=60, y=5)

        self.panel3 = tk.Label(self.root, bg="#add8e6")  # Current detected character display
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"), bg="#add8e6")   # "Character:" label
        self.T1.place(x=10, y=580)

        self.panel5 = tk.Label(self.root, bg="#add8e6")   # Sentence display panel
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"), bg="#add8e6")   # "Sentence:" label
        self.T3.place(x=10, y=632)

        self.T4 = tk.Label(self.root, text="Suggestions :", fg="red", font=("Courier", 30, "bold"), bg="#add8e6")  # "Suggestions:" label
        self.T4.place(x=10, y=700)

    # Create buttons with hover effect
        self.b1 = self.create_button(390, 700) # First suggestion button
        self.b2 = self.create_button(590, 700)  # Second suggestion button
        self.b3 = self.create_button(790, 700) # Third suggestion button
        self.b4 = self.create_button(990, 700)# Fourth suggestion button

        self.clear = self.create_button(1180, 630, text="Clear", command=self.clear_fun)  # Clear button (reset text)
        self.speak = self.create_button(1340, 630, text="Speak", command=self.speak_fun)  # Speak button (text-to-speech)

        # Initialize variables for text and word handling
        self.str = " "              # The current sentence being built
        self.ccc = 0                # General-purpose counter
        self.word = " "             # Current word being built
        self.current_symbol = "C"   # Currently detected symbol/character
        self.photo = "Empty"        # Placeholder for image reference

        # Variables for word suggestions
        self.word1 = " "            # First word suggestion
        self.word2 = " "            # Second word suggestion
        self.word3 = " "            # Third word suggestion
        self.word4 = " "            # Fourth word suggestion

        self.video_loop()           # Start the video processing loop
        
    def create_button(self, x, y, text="", command=None):
        btn = tk.Button(self.root, text=text, font=("Courier", 20), bg="#003366", fg="white",
                        activebackground="#005599", relief="flat", command=command)
        btn.place(x=x, y=y, width=150, height=50)
        btn.bind("<Enter>", lambda e: btn.config(relief="raised", bd=4))  # hover effect
        btn.bind("<Leave>", lambda e: btn.config(relief="flat", bd=0))    # normal
        return btn

    def video_loop(self):
        try:
            ok, frame = self.vs.read()                   # Read frame from webcam
            cv2image = cv2.flip(frame, 1)                # Flip horizontally (mirror image)
            if cv2image.any:                             # If frame has content
                hands = hd.findHands(cv2image, draw=False, flipType=True)  # Detect hands
                cv2image_copy = np.array(cv2image)        # Make a copy of the frame
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
                self.current_image = Image.fromarray(cv2image)  # Convert to PIL Image
                imgtk = ImageTk.PhotoImage(image=self.current_image)  # Convert to Tkinter PhotoImage
                self.panel.imgtk = imgtk                  # Update the panel's image
                self.panel.config(image=imgtk)            # Configure panel to show the new image

                if hands[0]:                             # If hands are detected
                    hand = hands[0]                      # Get the first hand
                    map = hand[0]                        # Get the hand map (landmarks and bounding box)
                    x, y, w, h = map['bbox']             # Extract bounding box coordinates
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop hand region with offset

                    white = cv2.imread("white.jpg")      # Load white background image
                    # img_final=img_final1=img_final2=0   # Commented out code
                    if image.all:                        # If the cropped image exists
                        handz = hd2.findHands(image, draw=False, flipType=True)  # Detect hand in cropped image
                        self.ccc += 1                    # Increment counter
                        if handz[0]:                     # If hand is detected in cropped image
                            hand = handz[0]              # Get the hand
                            handmap = hand[0]            # Get hand map
                            self.pts = handmap['lmList']  # Get list of landmark points
                            
                            

                            # Calculate offsets to center the hand in visualization
                            os = ((400 - w) // 2) - 15   # Horizontal offset
                            os1 = ((400 - h) // 2) - 15  # Vertical offset
                            
                            # Draw lines connecting hand landmarks for thumb
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), 
                                        (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                        (0, 255, 0), 3)
                            # Draw lines for index finger
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), 
                                        (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                        (0, 255, 0), 3)
                            # Draw lines for middle finger
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), 
                                        (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                        (0, 255, 0), 3)
                            # Draw lines for ring finger
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), 
                                        (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                        (0, 255, 0), 3)
                            # Draw lines for pinky finger
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), 
                                        (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                        (0, 255, 0), 3)
                                # Draw palm connections
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                                    3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                                    3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                                    (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                                    3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                                    3)

                            # Draw landmark points as red circles
                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                            res = white                   # Store the resulting image
                            self.predict(res)             # Predict the sign from the hand image

                            self.current_image2 = Image.fromarray(res)  # Convert to PIL Image
                            imgtk = ImageTk.PhotoImage(image=self.current_image2)  # Convert to Tkinter PhotoImage
                            self.panel2.imgtk = imgtk     # Update the hand visualization panel
                            self.panel2.config(image=imgtk)

                            # Update UI elements with detection results
                            self.panel3.config(text=self.current_symbol, font=("Courier", 30))  # Show current symbol
                            
                            # Update word suggestion buttons
                            self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
                            self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
                            self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)

                self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
        except Exception:
            print(Exception.__traceback__)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("C:\\Users\\Mohd khaleelullah\\OneDrive\\Desktop\\sign\\white.jpg")
                # img_final=img_final1=img_final2=0

                handz = hd2.findHands(image, draw=False, flipType=True)
                print(" ", self.ccc)
                self.ccc += 1
                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']
                    # x1,y1,w1,h1=hand['bbox']

                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                            3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                            3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                            (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                            3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                            3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    self.panel3.config(text=self.current_symbol, font=("Courier", 30))

                    #self.panel4.config(text=self.word, font=("Courier", 30))



                    self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                    self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825,  command=self.action2)
                    self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825,  command=self.action3)
                    self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825,  command=self.action4)

            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)  # Schedule the next frame processing after 1ms

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")          # Find the last space in the sentence
        idx_word = self.str.find(self.word, idx_space)  # Find the current word after that space
        last_idx = len(self.str)                 # Get end of string index
        self.str = self.str[:idx_word]           # Remove the current word
        self.str = self.str + self.word1.upper() # Replace with first suggestion (uppercase)


    def action2(self):
        idx_space = self.str.rfind(" ")          # Find the last space in the sentence
        idx_word = self.str.find(self.word, idx_space)  # Find the current word after that space
        last_idx = len(self.str)                 # Get end of string index
        self.str = self.str[:idx_word]           # Remove the current word
        self.str = self.str + self.word2.upper() # Replace with second suggestion (uppercase)


    def action3(self):
        idx_space = self.str.rfind(" ")          # Find the last space in the sentence
        idx_word = self.str.find(self.word, idx_space)  # Find the current word after that space
        last_idx = len(self.str)                 # Get end of string index
        self.str = self.str[:idx_word]           # Remove the current word
        self.str = self.str + self.word3.upper() # Replace with third suggestion (uppercase)



    def action4(self):
        idx_space = self.str.rfind(" ")          # Find the last space in the sentence
        idx_word = self.str.find(self.word, idx_space)  # Find the current word after that space
        last_idx = len(self.str)                 # Get end of string index
        self.str = self.str[:idx_word]           # Remove the current word
        self.str = self.str + self.word4.upper() # Replace with fourth suggestion (uppercase)


    def speak_fun(self):
        self.speak_engine.say(self.str)          # Send current sentence to text-to-speech engine
        self.speak_engine.runAndWait()           # Play the speech and wait for completion


    def clear_fun(self):
        self.str = " "                           # Reset the sentence to a space
        self.word1 = " "                         # Clear all suggestions
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)     # Reshape image for the model
        prob = np.array(self.model.predict(white)[0], dtype='float32')  # Get model predictions
        ch1 = np.argmax(prob, axis=0)             # Get the highest probability prediction
        prob[ch1] = 0                             # Zero out the highest probability
        ch2 = np.argmax(prob, axis=0)             # Get the second highest prediction
        prob[ch2] = 0                             # Zero out the second highest
        ch3 = np.argmax(prob, axis=0)             # Get the third highest prediction
        prob[ch3] = 0                             # Zero out the third highest

        pl = [ch1, ch2]                           # Store top two predictions

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
            [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
            [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
            [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
            [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
            self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1


        # Generate word suggestions
        if len(self.str.strip()) != 0:           # If there's text in the sentence
            st = self.str.rfind(" ")             # Find the last space
            ed = len(self.str)                   # Get end position
            word = self.str[st+1:ed]             # Extract the current word
            self.word = word                     # Store it
            if len(word.strip()) != 0:           # If the word isn't empty
                ddd.check(word)                  # Check if it's a valid word
                lenn = len(ddd.suggest(word))    # Get suggestions
                if lenn >= 4:                    # If there are at least 4 suggestions
                    self.word4 = ddd.suggest(word)[3]  # Store fourth suggestion
                if lenn >= 3:                    # If there are at least 3 suggestions
                    self.word3 = ddd.suggest(word)[2]  # Store third suggestion
                if lenn >= 2:                    # If there are at least 2 suggestions
                    self.word2 = ddd.suggest(word)[1]  # Store second suggestion
                if lenn >= 1:                    # If there is at least 1 suggestion
                    self.word1 = ddd.suggest(word)[0]  # Store first suggestion
            else:
                self.word1 = " "                 # Clear suggestions if no word
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "


    def destructor(self):
        print(self.ten_prev_char)                # Print the last 10 characters detected
        self.root.destroy()                      # Destroy the Tkinter window
        self.vs.release()                        # Release the webcam
        cv2.destroyAllWindows()                  # Close all OpenCV windows


print("Starting Application...")
(Application()).root.mainloop()  # Create Application instance and start the Tkinter event loop
