{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "objective-child",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "pickle_in = open(\"X_train.pickle\",\"rb\")\n",
    "X_train = pickle.load(pickle_in)\n",
    "\n",
    "pickle_in = open(\"y_train.pickle\",\"rb\")\n",
    "y_train = pickle.load(pickle_in)\n",
    "\n",
    "X_train = np.array(X_train/255.0)\n",
    "y_train = np.array(y_train)\n",
    "\n",
    "pickle_in = open(\"X_test.pickle\",\"rb\") \n",
    "X_test = pickle.load(pickle_in)\n",
    "\n",
    "pickle_in = open(\"y_test.pickle\",\"rb\") \n",
    "y_test = pickle.load(pickle_in)\n",
    "\n",
    "X_test = np.array(X_test/255.0) \n",
    "y_test = np.array(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "nearby-leeds",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from tensorflow.keras.layers import Conv2D, MaxPool2D"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "exact-sunrise",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(input_shape=X_train.shape[1:],filters=16,kernel_size=(3,3),padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=16,kernel_size=(3,3),padding=\"same\", activation=\"relu\"))\n",
    "model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))\n",
    "\n",
    "model.add(Conv2D(filters=32, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=32, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))\n",
    "\n",
    "model.add(Conv2D(filters=64, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=64, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=64, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))\n",
    "\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))\n",
    "\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(Conv2D(filters=128, kernel_size=(3,3), padding=\"same\", activation=\"relu\"))\n",
    "model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))\n",
    "\n",
    "model.add(Flatten())\n",
    "\n",
    "model.add(Dense(units=256,activation=\"relu\"))\n",
    "model.add(Dense(units=128,activation=\"relu\"))\n",
    "model.add(Dense(units=1, activation=\"sigmoid\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "floral-stage",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "36/36 [==============================] - 192s 5s/step - loss: 0.5709 - accuracy: 0.7298 - val_loss: 0.4533 - val_accuracy: 0.9126\n",
      "Epoch 2/5\n",
      "36/36 [==============================] - 194s 5s/step - loss: 0.3585 - accuracy: 0.8662 - val_loss: 0.2451 - val_accuracy: 0.9191\n",
      "Epoch 3/5\n",
      "36/36 [==============================] - 197s 5s/step - loss: 0.2790 - accuracy: 0.8897 - val_loss: 0.5029 - val_accuracy: 0.8414\n",
      "Epoch 4/5\n",
      "36/36 [==============================] - 193s 5s/step - loss: 0.1987 - accuracy: 0.9279 - val_loss: 0.2193 - val_accuracy: 0.9288\n",
      "Epoch 5/5\n",
      "36/36 [==============================] - 197s 5s/step - loss: 0.1612 - accuracy: 0.9444 - val_loss: 0.2033 - val_accuracy: 0.9320\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x2a5a3dc9bc8>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile( loss=\"binary_crossentropy\" , \n",
    "              optimizer = \"adam\" , \n",
    "              metrics=[\"accuracy\"] ) \n",
    "\n",
    "model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "durable-syndication",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('VGG16_FOR_CYCLONE_DETECTION.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7c59045b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "model = load_model('VGG16_FOR_CYCLONE_DETECTION.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "04211fa4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[174  14]\n",
      " [  7 114]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      0.93      0.94       188\n",
      "           1       0.89      0.94      0.92       121\n",
      "\n",
      "    accuracy                           0.93       309\n",
      "   macro avg       0.93      0.93      0.93       309\n",
      "weighted avg       0.93      0.93      0.93       309\n",
      "\n"
     ]
    }
   ],
   "source": [
    "mythreshold = 0.5\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "y_pred = (model.predict(X_test)>= mythreshold).astype(int)\n",
    "cm=confusion_matrix(y_test, y_pred)  \n",
    "print(cm)\n",
    "\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dfdca693",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9478c5ec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tensorflow_gpu",
   "language": "python",
   "name": "tensorflow_gpu"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
