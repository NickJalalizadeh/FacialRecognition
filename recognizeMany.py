import cv2
import os

#os.chdir("C:/Users/Nick/Documents/Python/FaceRecognition") #working directory

subjects = ["", "Frank", "Nick"]

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #LBP Cascade (faster, less accurate)
    #faceCascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    #faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #Haar Cascade (slower, more accurate)
    faceCascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return [None], [None]

    rects = []
    #extract the face area
    for face in faces:
        (x, y, w, h) = face
        rects.append(gray[y:y+w, x:x+h])

    #return only the face part of the image
    return rects, faces

#function to draw rectangle on image according to given (x, y) coordinates and given width and height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#function to draw text on give image starting from passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#this function recognizes the person in image passed and draws a rectangle around detected face with name of the subject
def predict(test_img, image_name):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()

    #detect face from the image
    faces, rects = detect_face(img)

    for ind, face in enumerate(faces):
        if face is not None:
            rect = rects[ind]
            #predict the image using our face recognizer
            label, confidence = face_recognizer.predict(face)

            #draw a rectangle around face detected
            draw_rectangle(img, rect)

            if label == -1:
                draw_text(img, "Unknown", rect[0], rect[1]-5)
            else:
                label_text = subjects[label]
                draw_text(img, label_text + " " + str(confidence), rect[0], rect[1]-5)

            #print(image_name, label_text, confidence)
            # cv2.imshow(image_name, face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     label, confidence = face_recognizer.predict(img)
        #     print(image_name, confidence)
        #     # cv2.imshow(image_name, img)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()

    return img

#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create(threshold=37)
face_recognizer.read("face_recognizer.xml")

#load test images
# test_images_names = sorted(os.listdir("test-data"))
# for image_name in test_images_names:
#
#     if image_name.startswith("."):
#         continue
#
#     image_path = "test-data/" + image_name
#     test_img = cv2.imread(image_path)
#     predicted_img = predict(test_img, image_name)

### Video Capture ###
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    predicted_img = predict(frame, "face")

    cv2.imshow("Recognizer", predicted_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
