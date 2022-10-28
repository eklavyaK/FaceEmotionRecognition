# importing the important libraries
import cv2
import numpy as np
from keras.models import model_from_json


################################### LOADING THE FILES ####################################
# loading the face detection classfier
face_recognizer = cv2.CascadeClassifier('D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/haarcascade_frontalface_default.xml')

# Opening and reading the model files
modelfile = open('D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/model/structure.json', "r")
model = modelfile.read()
network = model_from_json(model)
modelfile.close()

# load weights into new model
network.load_weights("D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/model/weights200.h5")

# mapping the emotions
emotions = ["Angry","Disgusted","Fearful","Happy","Neutral","Sad","Surprised"]

# capturing the video frame
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:/IIT/Academics/5fth Semester/ECN-343/Project/Project ETE/testvideos/testvideo1.mp4")


################################### RUNNING THE INFERENCE ON THE FRAME ####################################
while True:

    # reading the video and capturing the frame
    frame = cap.read()
    ok, img = frame
    img = cv2.resize(img, (1280, 720))
    # img = cv2.resize(img, (400, 800))

    if not ok:
        break
    
    # converting the frame to grayscale
    finalframe = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    allfaces = face_recognizer.detectMultiScale(finalframe, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for now in allfaces:
        (x, y, w, h) = now  
        # drawing the rectangle on the region of interest on the frame
        cv2.rectangle(img, (x, y-50), (x+w, y+h+5), (0, 255, 0), 4)

        #getting the region of interest for running the network on it
        region_of_interest = finalframe[y:y + h, x:x + w]
        region = np.expand_dims(np.expand_dims(cv2.resize(region_of_interest, (48, 48)), -1), 0)

        # predicting and highlighting the emotion on the frame
        emotion = network.predict(region)
        pos = int(np.argmax(emotion))
        cv2.putText(img, emotions[pos], (x+5, y-20), 3, 1, (255, 0, 0), 2, 16)

    cv2.imshow('Emotion Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
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