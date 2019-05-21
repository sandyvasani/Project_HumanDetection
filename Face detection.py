# OpenCV program to detect face in real time 
# import libraries of python OpenCV  
# where its functionality resides 
import cv2  
  
# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade_ff = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
face_cascade_sf = cv2.CascadeClassifier('haarcascade_profileface.xml')
face_cascade_up = cv2.CascadeClassifier('haarcascade_upperbody.xml')
face_cascade_lo = cv2.CascadeClassifier('haarcascade_lowerbody.xml') 
# capture frames from a camera 
cap = cv2.VideoCapture(2) 
  
def IOU_1(face1, face2):
	x_min = max (face1[0], face2[0])
	x_max = min (face1[0]+face1[2], face2[0]+face2[2])
	y_min = max (face1[1], face2[1])
	y_max = min (face1[1]+face1[3], face2[1]+face2[3])
	
	if (x_min > x_max or y_min > y_max ):
		return False
	else:
		return True
	   
  
def IOU(faces_ff, faces_sf):
    listToDel = []
    for idx1, face1 in enumerate(faces_ff):
        for idx2, face2 in enumerate(faces_sf):
            result = IOU_1 (face1, face2)
            if result:
                print (len(faces_sf), idx2)
                #faces_sf.tolist().pop(idx2)
                listToDel.append (idx2)
                break
    sideFace = []        
    for idx2, face2 in enumerate(faces_sf):
        if idx2 not in listToDel:
            sideFace.append (face2)
    return faces_ff, sideFace
    		
    			 	       
	 	 
bg = None    
# loop runs if capturing has been initialized.
i = 0 
while i < 10:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = gray
    i += 1

fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()
    
    img = cv2.flip (img, 1)  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    result = abs(bg - gray)
    fgmask = fgbg.apply(gray)
    
    #print (gray.shape)
  
    # Detects faces of different sizes in the input image
    #For Front face 
    faces_ff = face_cascade_ff.detectMultiScale(gray, 1.3, 4) 
    
    #For Side face
    faces_sf = face_cascade_sf.detectMultiScale(gray, 1.05, 4)
    
    #For Upper Body
    faces_up = face_cascade_up.detectMultiScale(gray, 1.05, 4 , minSize=(150,150))
    
    #For Lower Body
    faces_lo = face_cascade_lo.detectMultiScale(gray, 1.05, 4 , minSize=(150,150))
    
    if (len (faces_ff) and len(faces_sf)):
        print ("before : ", len(faces_ff), len(faces_sf))
        faces_ff, faces_sf = IOU(faces_ff, faces_sf)
        print ("After  : ", len(faces_ff), len(faces_sf))
     
  	  
    for (x,y,w,h) in faces_ff: 
        # To draw a rectangle in a face_ff  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
    
    for (x,y,w,h) in faces_sf:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
    
    for (x,y,w,h) in faces_up:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]  
    
    for (x,y,w,h) in faces_lo:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]  
    
         
    # Display an image in a window 
    cv2.imshow('img', img)
    cv2.imshow('fgmask', fgmask)
      
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 