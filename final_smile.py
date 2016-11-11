import numpy as np
import cv2
from pylab import *
from sklearn.svm import SVC
from scipy.ndimage import zoom
from sklearn.externals import joblib

#loading a classifier model
svc_1 = joblib.load('smile.joblib.pkl')
def detect_face(frame):
            cascPath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascPath)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=6,
                        minSize=(100, 100),
                        flags=cv2.CASCADE_SCALE_IMAGE
            )
            return gray, detected_faces

def extract_face_features(gray, detected_face, offset_coefficients):
            (x, y, w, h) = detected_face
            horizontal_offset = offset_coefficients[0] * w
            vertical_offset = offset_coefficients[1] * h
            extracted_face = gray[y+vertical_offset:y+h,
                                    x+horizontal_offset:x-horizontal_offset+w]
            new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                                64. / extracted_face.shape[1]))
            new_extracted_face = new_extracted_face.astype(float32)
            new_extracted_face /= float(new_extracted_face.max())
            return new_extracted_face

def predict_face_is_smiling(extracted_face):
            return svc_1.predict(extracted_face.ravel())

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#open-webcam
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # detect faces
    gray, detected_faces = detect_face(frame)
    
    face_index = 0
    
    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # draw rectangle around face 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # extract features
            extracted_face = extract_face_features(gray, face, (0.03, 0.05)) #(0.075, 0.05)

            # predict smile
            prediction_result = predict_face_is_smiling(extracted_face)

            # draw extracted face in the top right corner
            frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            if prediction_result == 1:
                cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)
            else:
                cv2.putText(frame, "not smiling",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

            # increment counter
            face_index += 1
             
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
