from turtle import color 
import cv2
import numpy as np
import random
from gtts import gTTS
import os

# mytext = 'Welcome to !'
  
# myobj = gTTS(text=mytext, lang='en', slow=False)

# myobj.save("welcome.mp3")

# os.system("mpg321 welcome.mp3")



R=random.randint(0,255)
G=random.randint(0,255)
B=random.randint(0,255)

video = cv2.VideoCapture("test5.mp4")
          

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobile","glasses"]

color=[(R,G,B) for i in CLASSES]
net=cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt','MobileNetSSD_deploy.caffemodel')


count = 0
count2=0

arr = []

st = ""

def addElementToArray(element):
    for i in arr:
        if(i == element):
            return
    arr.append(element)
    # for i in arr:
    #      print(arr)

while True:
    ret,frame=video.read()
    frame=cv2.resize(frame,(800,800))
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)
    net.setInput(blob)
    detections= net.forward()


    for i in np.arange(0, detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.5:

            id=detections[0,0,i,1]

            addElementToArray(str(CLASSES[int(id)]))

            if count%300 == 0:
                
                for f in arr:
                    if count2==0:
                        st= "I can see There" + str(f)
                        count2+=1
                    else:
                        st = st + ", " + str(f)
                mytext = st
                myobj = gTTS(text=mytext, lang='en', slow=False)
                myobj.save("welcome.mp3")
                os.system("welcome.mp3") 
                st = ""
                arr = []

                count = count + 1
                

            count = count + 1


            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            cv2.rectangle(frame, (startX-1,startY-40), (endX+1,startY-3), color[int(id)],-1)
            cv2.rectangle(frame, (startX,startY),(endX,endY),color[int(id)],4)
            cv2.putText(frame,CLASSES[int(id)], (startX+10, startY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),4)
            
    cv2.imshow("Frame",frame)
    k= cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
