import cv2
import numpy as np
from collections import deque
import os


from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model

model = load_model("mnist_model.h5")
model.summary()
model.load_weights("mnist_weights.h5") 


label = ["ZERO","ONE","TWO","TREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE"]

cap = cv2.VideoCapture(0)

lower_blue= np.array([110,75,75])
upper_blue= np.array([140,255,255])

black_points = [deque(maxlen=512)]
black_index = 0

colors = [(0,0,0)]
color_index = 0

paintWindow = np.zeros((210,270,3))+255




#cv2.namedWindow("Paint")

try:
    


    while 1:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(400,400))
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        #draw frame
        frame = cv2.rectangle(frame, (150,80),(250,250),(255,0,0),2)
        paint= frame[83:245,155:245]
        
        #clear button
        frame = cv2.rectangle(frame,(40,1),(140,65),(0,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"CLEAR ALL",(49,33),font,0.5,(0,0,0),2,cv2.LINE_AA)
        #black
        frame = cv2.rectangle(frame,(160,1),(255,65),colors[0],-1)
        cv2.putText(frame,"DRAW",(185,33),font,0.5,(255,255,255),2,cv2.LINE_AA)
        #predict button
        frame = cv2.rectangle(frame,(275,1),(370,65),(255,255,255),-1)
        cv2.putText(frame,"PREDICT",(298,33),font,0.5,(255,0,0),2,cv2.LINE_AA)
        
        #RESULT
        frame = cv2.rectangle(frame,(0,350),(100,400),(255,255,255),2)
        cv2.putText(frame,"RESULT",(0,345),font,0.5,(255,155,0),2,cv2.LINE_AA)
        if ret is False:
            break

        mask = cv2.inRange(hsv,lower_blue,upper_blue)

        mask = cv2.erode(mask,(15,15),iterations =10)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,(5,5))
        mask = cv2.dilate(mask,(10,10),iterations = 1)
    
        contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        center = None
    
    

        if len(contours) > 0:
            max_contours = sorted(contours, key = cv2.contourArea, reverse=True)[0]
            ((x,y),radius) = cv2.minEnclosingCircle(max_contours)
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,0,0),3)
        
            M = cv2.moments(max_contours)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))

            if center[1] <= 65:
                if 40<=center[0]<=140:

                    black_points = [deque(maxlen=512)]
                    black_index=0
            
                    paintWindow[65:,:,:]=255
             
                elif 160<=center[0]<=255:
                    color_index = 0
                
                elif 275<=center[0]<=370:
                    
                   
                    img = load_img("paint.png", grayscale=True, target_size=(28, 28))
                    
                    x= img_to_array(img)
                    x = np.expand_dims(x, axis = 0)
                    x = x.reshape(-1,28,28,1)/255.0


                    
                    predict = model.predict(x)
                    index = np.argmax(predict)
                    cv2.putText(frame,label[index],(15,370),font,0.5,(255,255,255),2,cv2.LINE_AA)                    
                    print(label[index])
           

            else:
                if color_index == 0:
                    black_points[black_index].appendleft(center)
                
        
        else:
            black_points.append(deque(maxlen=512))
            black_index+=1
        

        points = [black_points]
    
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1,len(points[i][j])):
                    if points[i][j][k-1] is None or points[i][j][k] is None:
                        continue
                    
                    cv2.line(frame,points[i][j][k-1], points[i][j][k], colors[i], 10)
                            
                    #cv2.line(paintWindow,points[i][j][k-1], points[i][j][k], colors[i], 5)
    
            

    
    
    
    
        
        try:
            os.remove("paint.png") 
        except FileNotFoundError:
            pass
        cv2.imwrite("paint.png",paint)
        
        
        
        cv2.imshow("Frame",frame)
        cv2.imshow("Paint",paint)
     

        # cv2.imshow("hsv",hsv)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


except ZeroDivisionError:
    print("mavi nesneyi tutunuz kameraya dogru")


except PermissionError:
    print("Erişim engellendi:")



























