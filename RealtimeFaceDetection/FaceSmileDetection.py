import cv2

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
# Trained XML file for detecting smiles
smile_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_smile.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

while True: 
    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 

        # Detect smile
        smile = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor= 1.5,
                minNeighbors=15,
                minSize=(25, 25),
                )
        
        # To draw a rectangle in smile
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
    
    # Display an image in a window 
    cv2.imshow('img',img) 

    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
    
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()