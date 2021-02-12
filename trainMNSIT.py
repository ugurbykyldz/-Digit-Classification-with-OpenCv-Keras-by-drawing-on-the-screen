import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Dropout,Flatten,Activation,BatchNormalization
from keras.utils import to_categorical
from keras.datasets import mnist

import matplotlib.pyplot as plt
import seaborn as sns


(x_train, y_train), (x_test,y_test) = mnist.load_data()


for i in range(10):  
    img = x_train[i,:]
    plt.imshow(img)
    plt.show()
    
sns.countplot(y_train)    



x_train = x_train.reshape(-1,28,28,1).astype("float32")    
x_test = x_test.reshape(-1,28,28,1).astype("float32")
input_shape = (28,28,1)


x_train /= 255
x_test /= 255


num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)



#MODEL
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3),  input_shape = input_shape,padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size = 3,padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())



model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ['accuracy'])

model.summary()

#TRAIN
batch_size = 128  
epochs = 6

hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

#save model
model.save('mnist_model.h5')

#save model weights
model.save_weights("mnist_weights.h5")


# model evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

print(hist.history.keys())
plt.plot(hist.history["loss"], label = "TRAIN LOSS")
plt.plot(hist.history["val_loss"], label = "VALIDATION LOSS")
plt.legend()
plt.show()

plt.figure()
print(hist.history.keys())
plt.plot(hist.history["accuracy"], label = "TRAIN ACC")
plt.plot(hist.history["val_accuracy"], label = "VALIDATION ACC")
plt.legend()
plt.show()











