from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt 

model = load_model("mnist_model.h5")
model.summary()
model.load_weights("mnist_weights.h5") 



img = load_img("paint.png", grayscale=True, target_size=(28, 28))


plt.imshow(img)
plt.axis("off")
plt.show() 
                   
x= img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = x.reshape(-1,28,28,1)/255.0


label = ["ZERO","ONE","TWO","TREE","FOUR","FİVE","SİX","SEVEN","EIGHT","NINE"]
predict = model.predict(x)
index = np.argmax(predict)                    
print(label[index])
