import cv2
import os
import numpy as np

os.chdir("C:/Users/Nick/Documents/Python/FaceRecognition") #working directory

#LBP Cascade (faster, less accurate)
#faceCascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
#faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#Haar Cascade (slower, more accurate)
faceCascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):

    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []

    #let's go through each directory and read images within it
    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith(".") or image_name.startswith("_") :
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            # faces.append(image)
            # labels.append(label)

            face, rect = detect_face(image)

            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
                #cv2.imshow(image_name, face)
            else:
                print(label, "invalid")
                cv2.imshow(image_name, image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return faces, labels

faces, labels = prepare_training_data("training-data_new")
print("Total faces: ", len(faces))

#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

face_recognizer.save('face_recognizer.xml')
