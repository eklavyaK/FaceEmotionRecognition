# importing the important libraries
import cv2
import numpy as np
from keras.models import model_from_json


################################### LOADING THE FILES ####################################
# loading the face detection classfier
face_recognizer = cv2.CascadeClassifier('D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/haarcascade/haarcascade_frontalface_default.xml')

# Opening and reading the model files
modelfile = open('D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/model/structure.json', "r")
model = modelfile.read()
network = model_from_json(model)
modelfile.close()

# load weights into new model
network.load_weights("D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/model/weights200.h5")

# mapping the emotions
emotions = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]


################################### RUNNING THE INFERENCE ON THE FRAME ####################################
while True:
    # reading the image
    img = cv2.imread("D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/testimages/testimage3.jpg")
    #img = cv2.resize(img,(1216,742))
    #img = cv2.resize(img,(1308,738))
    img = cv2.resize(img,(1314,548))

    # converting the frame to grayscale
    finalframe = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    allfaces = face_recognizer.detectMultiScale(finalframe, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for now in allfaces:
        (x, y, w, h) = now  
        # drawing the rectangle on the region of interest on the frame
        cv2.rectangle(img, (x, y-50), (x+w, y+h+25), (0, 255, 0), 4)

        #getting the region of interest for running the network on it
        region_of_interest = finalframe[y:y + h, x:x + w]
        region = np.expand_dims(np.expand_dims(cv2.resize(region_of_interest, (48, 48)), -1), 0)

        # predicting and highlighting the emotion on the frame
        emotion = network.predict(region)
        pos = int(np.argmax(emotion))
        cv2.putText(img, emotions[pos], (x+5, y-20), 3, 1, (255, 0, 0), 2, 16)

    cv2.imshow('Emotion Detection', img)
    if cv2.waitKey(1000) & 0xFF == ord('s'):
        break
################################### END OF THE PROGRAM ####################################

#available fonts in cv2

#   FONT_HERSHEY_SIMPLEX = 0
#   FONT_HERSHEY_PLAIN = 1
#   FONT_HERSHEY_DUPLEX = 2
#   FONT_HERSHEY_COMPLEX = 3
#   FONT_HERSHEY_TRIPLEX = 4
#   FONT_HERSHEY_COMPLEX_SMALL = 5
#   FONT_HERSHEY_SCRIPT_SIMPLEX = 6
#   FONT_HERSHEY_SCRIPT_COMPLEX = 7


#available linetypes in cv2

#   FILLED = -1,
#   LINE_4 = 4,
#   LINE_8 = 8,
#   LINE_AA = 16