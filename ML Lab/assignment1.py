import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
import pickle
import cv2

model_save_path = 'saved models/{}.h5'

class models:
    def train_mnist_dense(self):
        (trainX,trainY),(testX,testY) = tf.keras.datasets.mnist.load_data()
        trainX,testX = trainX/255.0,testX/255.0
        mnist_model = Sequential([
            layers.Flatten(),
            layers.Dense(64,activation='relu'),
            layers.Dense(32,activation='relu'),
            layers.Dense(10,activation='softmax'),
        ])

        mnist_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = mnist_model.fit(trainX,trainY,epochs=3,batch_size=32)
        self.save_model(mnist_model,'mnist')
        self.eval_model(mnist_model,testX,testY)
        self.save_plot(history,'mnist')

    def train_cifar10_dense(self):
        (trainX,trainY),(testX,testY) = tf.keras.datasets.cifar10.load_data()
        trainX,testX = trainX/255.0,testX/255.0
        cifar10_model = Sequential([
            layers.Flatten(),
            layers.Dense(64,activation='relu'),
            layers.Dense(32,activation='relu'),
            layers.Dense(10,activation='softmax'),
        ])

        cifar10_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = cifar10_model.fit(trainX,trainY,epochs=3,batch_size=32)
        self.save_model(cifar10_model,'cifar10_dense')
        self.eval_model(cifar10_model,testX,testY)
        self.save_plot(history,'cifar10_dense')

    def train_cifar10_cnn(self):
        (trainX,trainY),(testX,testY) = tf.keras.datasets.cifar10.load_data()
        trainX,testX = trainX/255.0,testX/255.0
        # trainX,testX = np.expand_dims(np.array(trainX),-1),np.expand_dims(np.array(testX),-1)
        cifar10_model = Sequential()
        cifar10_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        cifar10_model.add(layers.MaxPooling2D((2, 2)))
        cifar10_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cifar10_model.add(layers.MaxPooling2D((2, 2)))
        cifar10_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cifar10_model.add(layers.Flatten())
        cifar10_model.add(layers.Dense(64, activation='relu'))
        cifar10_model.add(layers.Dense(10,activation='softmax'))

        cifar10_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        history = cifar10_model.fit(trainX,trainY,epochs=3,batch_size=32)
        self.save_model(cifar10_model,'cifar10_cnn')
        self.eval_model(cifar10_model,testX,testY)
        self.save_plot(history,'cifar10_cnn')

    def save_model(self,model,name):
        model.save(model_save_path.format(f'assignment1_{name}'))

    def eval_model(self,model,x,y):
        eval = model.evaluate(x,y,verbose=0)
        print(f'accuracy: {eval[1]}\tloss: {eval[0]}')
    
    def save_plot(self,hist,name):
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['loss'],color='r')
        plt.savefig(f'saved plots/{name}_eval_curve.png')
        plt.clf()

    def save_pickle(self,model,model_name,x,y):
        pred = model.predict(x)
        pickle.dump(pred,open(f'prediction_{model_name}.pickle','wb+'))
        pickle.dump(y,open(f'actual_{model_name}.pickle','wb+'))


    def create_model_report(self,model_name):
        y = pickle.load(open(f'saved_files/assignment 1/actual_{model_name}.pickle','rb'))
        yhat = pickle.load(open(f'saved_files/assignment 1/prediction_{model_name}.pickle','rb'))
        y_pred = [np.argmax(i) for i in yhat]
        
        return (precision_score(y,y_pred,average='weighted'),recall_score(y,y_pred,average='weighted'),
                f1_score(y,y_pred,average='weighted'))
        
    def get_image(self,f):
        img = cv2.imread(f)
        return img

    def predict_mnist(self,f):
        img = self.get_image(f)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(28,28))
        model = tf.keras.models.load_model('saved models/assignment 1/assignment1_mnist.h5')
        pred = model.predict(np.expand_dims(img,axis=0))
        return pred[0]

    def predict_cifar10(self,f):
        img = self.get_image(f)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(32,32))
        model = tf.keras.models.load_model('saved models/assignment 1/assignment1_cifar10_dense.h5')
        pred = model.predict(np.expand_dims(img,axis=0))
        return pred[0]

    def predict_cifar10_cnn(self,f):
        img = self.get_image(f)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(32,32))
        model = tf.keras.models.load_model('saved models/assignment 1/assignment1_cifar10_cnn.h5')
        pred = model.predict(np.expand_dims(img,axis=0))
        return pred[0]