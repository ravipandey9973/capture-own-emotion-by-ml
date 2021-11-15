#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


img_array=cv2.imread("training/0/Training_10118481.jpg")


# In[3]:


img_array.shape #rgb


# In[4]:


print(img_array)


# In[5]:


plt.imshow(img_array) ## BGR


# In[6]:


Datadirectory="training/" ## training dataset


# In[7]:


Classes=["0","1","2","3","4","5","6"]   ## list of classes ->exact your folder name


# In[8]:


for category in Classes:
    path=os.path.join(Datadirectory,category)  ##/
    for img in os.listdir(path):
        img_array =cv2.imread(os.path.join(path,img))
        #backtorgb=cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break
    


# In[9]:


img_size=224 ## IMGNET=>224X224
new_array= cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()


# In[10]:


new_array.shape


# # read all the images and convertin them to array

# In[11]:


training_Data =[] ## data

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category) ## 0 1; ## label
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array= cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


# In[ ]:


create_training_Data()


# In[ ]:


print(len(training_Data))


# In[ ]:


temp = np.array(training_Data)


# In[ ]:


temp.shape


# In[ ]:


import random

random.shuffle(training_Data)


# In[ ]:


x = [] ##data /feature
y = [] ##label

for features,label in training_Data:
    x.append(features)
    y.append(label)

    
x = np.array(x).reshape(-1,img_size, img_size, 3)  ## converting it to 4 dimension


# In[ ]:


x.shape


# In[ ]:


# normalize the data
x=x/151.0; ## we are normalizing it


# In[ ]:


type(y)


# In[ ]:


Y=np.array(y)


# In[ ]:


Y.shape


# # deep learning model for training-Transfer Learning

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


model = tf.keras.applications.MobileNetV2() ##Pre-trained Model


# In[ ]:


model.summary()


# # Transfer Learning- Tuning,weight will start from last check point

# In[ ]:


base_input = model.layers[0].input ## input


# In[ ]:


base_input = model.layers[0].input ## input


# In[ ]:


final_output = layers.Dense(128)(base_output)  ##adding new layer,after the output of global pooling layer
final_output = layers.Activation('relu')(final_output) ## activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output)  ## my classes are 07


# In[ ]:


final_output ## output


# In[ ]:


new_model = keras.Model(inputs = base_input, outputs=final_output)


# In[ ]:


new_model.summary()


# In[ ]:


new_model.compile(loss="sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])


# In[ ]:


new_model.fit(x,y, epochs = 25)  ## training


# In[ ]:


new_model.save('Final_model_95p07.h5')


# In[ ]:


new_model = tf.keras.models.load_model('Final_model_95p07.h5')


# In[ ]:


frame = cv2.imread("py.jpg")


# In[ ]:


frame.shape


# In[ ]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[ ]:


# we need face detection algorithm (gray image)


# In[ ]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[ ]:


gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# In[ ]:


gray.shape


# In[ ]:


faces =faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h,x:x+w]
    roi_color =frame[y:y+h,x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #BGR
    facess =faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh,ex:ex+ew]


# In[ ]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[ ]:


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


# In[ ]:


final_image =cv2.resize(face_roi, (224,224))  ##
final_image = np.expand_dims(final_image,axis =0)  ## need fourth dimension
final_image=final_image/255.0  ##normalizing


# In[ ]:


Predictions =new_model.predict(final_image)


# In[ ]:


Predictions[0]


# In[ ]:


np.argmax(Predictions)


# # Realtime Video Demo 

# In[ ]:


import numpy as np
import cv2 ### pip install opencv-python
## pip install opencv-contri-python fullpackage
#from deepface import DeepFace ## pip install deepface
path = "haarcascade_fronatalface_default.xml"
font_scale =1.5
font = cv2.FONT_HERSHEY_PLAIN

#set the rectangle background to white
rectangle_bgr = (255,255,255)
# make a block image
img = np.zeros((500,500))
# set some text
text = "some text in a box!"
#get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
# make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x,text_offset_y),(text_offset_x + text_width +2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
 

cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")
    
while True:
    ret,frame = cap.read()
    #eye_cascade = cv2.CasecadeClassifier(cv2.data.haarcascades +'haarcascades_eye.xml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
            roi_gray =gray[y:y+h, x:x+w]
            roi_color =frame[y:y+h, x:x+w]
            cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("Face not detected")
            else:
                for (ex,ey,ew,eh) in facess:
                     face_roi = roi_color[ey: ey+eh, ex:ex + ew]       ##cropping the face
    
    
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image,axis =0) ## neesd fourth dimension
    final_image=final_image/255.0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    Predictions = new_model.predict(final_image)
    
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    
    if (np.argmax(Predictions)==0):
        status = "Angry"
        
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1),(0,0,0), -1)
        #Add text
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100, 150),font, 3,(0, 0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        
    elif (np.argmax(Predictions)==1):
        status = "Disgust"
        
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1),(x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame,status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100, 150),font, 3,(0, 0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        
    elif (np.argmax(Predictions)== 2):
        status = "Fear"
        
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1+ w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        
        cv2.putText(frame,status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
        
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        
    elif (np.argmax(Predictions)==3):
        status = "Happy"
        
        x1,y1,w1,h1 = 0,0,175,75
        #Draw black background rectangle
        cv2.rectangle(frame, (x1, x1),(x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100,150),font, 3,(0,0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        
    elif (np.argmax(Predictions)==4):
        status = "Sad"
        
        x1,y1,w1,h1 = 0,0,175,75
        #Draw black background rectangle
        cv2.rectangle(frame, (x1, x1),(x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100,150),font, 3,(0,0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
  
    elif (np.argmax(Predictions)==5):
        status = "Surprise"
        
        x1,y1,w1,h1 = 0,0,175,75
        #Draw black background rectangle
        cv2.rectangle(frame, (x1, x1),(x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100,150),font, 3,(0,0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
    
    
    else:
        status = "Neutral"
        
        x1,y1,w1,h1 = 0,0,175,75
        #Draw black background rectangle
        cv2.rectangle(frame, (x1, x1),(x1 + w1, y1 + h1), (0,0,0), -1)
        # Add text
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(frame,status,(100,150),font, 3,(0,0, 255),2,cv2.LINE_4)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
                     
                     
    # gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(faceCascade.empty())
    # faces = faceCascade.detectMultiScale(gray,1.1,4)
                     
    # Draw a rectangle around the facesQQQQQ
    # for(x, y, w, h) in faces
     # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
     
                      
    # Use putText() ethod for
    # inserting text on video

                      
                          
    cv2.imshow('Face Emotion Recognition', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
                    
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




