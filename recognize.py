import cv2
from os import chdir

chdir("C:/Users/Nick/Documents/Python/FaceRecognition") #working directory

subjects = ["Unknown", "Damien", "Nick", "Rheza", "Frank"]

#LBP Cascade (faster, less accurate)
#faceCascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

#Haar Cascade (slower, more accurate)
faceCascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

def detect_face(img):
    #convert the image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) #LBP
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #Haar

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#function to draw rectangle on image according to given (x, y) coordinates and given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 128, 0), 2)

#function to draw text on give image starting from passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 0), 2)

labels = []
confidences = []
#this function recognizes the person in image passed and draws a rectangle around detected face with name of the subject
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()

    face, rect = detect_face(img)

    if face is not None:
        label, confidence = face_recognizer.predict(face)
        confidence = float("{0:.3f}".format(confidence))
        #p_confidence = 110 - int(round(confidence))

        #labels.append(label)
        #label = max(set(labels[-10:]), key=labels[-10:].count) #find mode of last 10 labels
        #confidence = sum(confidences[-10:])/len(confidences[-10:])
        #print(confidence)

        draw_rectangle(img, rect)

        if label == -1:
            draw_text(img, "Unknown", rect[0], rect[1]-5)
        else:
            label_text = subjects[label]
            draw_text(img, label_text + " " + str(confidence), rect[0], rect[1]-5)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # label = face_recognizer.predict(gray)
    # print(label)
    #confidence = float("{0:.3f}".format(confidence))

    return img

#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.xml")

### Video Capture ###
video_capture = cv2.VideoCapture(0)

#TODO: say 'move closer' when person is too far away
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #frame = cv2.resize(frame, (2160, 2160))

    predicted_img = predict(frame)

    cv2.imshow("Recognizer", predicted_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #print(subjects[4], (sum(average4) + sum(average5)) / (float(len(average4)) + float(len(average5))))
        #print(subjects[5], sum(average5) / float(len(average5)))
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
