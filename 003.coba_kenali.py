import face_recognition
import cv2
import numpy as np

def LoadFaces():
    bruno_image = face_recognition.load_image_file("images/sofyan.jpg")
    bruno_face_encoding = face_recognition.face_encodings(bruno_image)[0]
    valentino_image = face_recognition.load_image_file("images/rizal.jpg")
    valentino_face_encoding = face_recognition.face_encodings(valentino_image)[0]
    valentino_imagex = face_recognition.load_image_file("images/boy.jpg")
    valentino_face_encodingx = face_recognition.face_encodings(valentino_imagex)[0]
    valentino_imagexx = face_recognition.load_image_file("images/yusron.jpg")
    valentino_face_encodingxx = face_recognition.face_encodings(valentino_imagexx)[0]
    valentino_image3 = face_recognition.load_image_file("images/tri.jpg")
    valentino_face_encoding3 = face_recognition.face_encodings(valentino_image3)[0]

    known_face_encodings = [
        bruno_face_encoding,
        valentino_face_encoding,
        valentino_face_encodingx,
        valentino_face_encodingxx,
        valentino_face_encoding3
        ]
    known_face_names = [
        "sofyan",
        "rizal",
        "boy",
        "yusron",
        "tri"
    ]

    return known_face_encodings, known_face_names;

video_capture = cv2.VideoCapture(0)
known_face_encodings, known_face_names = LoadFaces()

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    for face_landmarks in face_landmarks_list:

        for facial_feature in face_landmarks.keys():
            pts = np.array([face_landmarks[facial_feature]], np.int32) 
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame, [pts], False, (0,255,0))

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()