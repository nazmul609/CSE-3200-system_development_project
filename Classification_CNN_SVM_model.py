{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "united-configuration",
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
   "execution_count": 2,
   "id": "supported-first",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.layers import Conv2D\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.regularizers import l2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fiscal-plane",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.models.Sequential()\n",
    "\n",
    "model.add( tf.keras.layers.Conv2D(filters=16,padding=\"same\",kernel_size=3, activation='relu', strides=2, input_shape=X_train.shape[1:] ))\n",
    "model.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "\n",
    "model.add( tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu' ))\n",
    "model.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "model.add( tf.keras.layers.Conv2D(filters=64,padding='same',kernel_size=3, activation='relu' ))\n",
    "model.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "model.add( tf.keras.layers.Conv2D(filters=128,padding='same',kernel_size=3, activation='relu' ))\n",
    "model.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "model.add( tf.keras.layers.Flatten())\n",
    "model.add( tf.keras.layers.Dense(units=256, activation='relu' ))\n",
    "model.add( tf.keras.layers.Dense(units=128, activation='relu' ))\n",
    "#model.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # for CNN_classification output\n",
    "\n",
    "model.add( Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear')) #for_svm_classifier\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "quantitative-piano",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "36/36 [==============================] - 16s 446ms/step - loss: 0.6458 - accuracy: 0.5639 - val_loss: 0.3754 - val_accuracy: 0.6117\n",
      "Epoch 2/50\n",
      "36/36 [==============================] - 15s 421ms/step - loss: 0.2749 - accuracy: 0.8801 - val_loss: 0.2529 - val_accuracy: 0.9094\n",
      "Epoch 3/50\n",
      "36/36 [==============================] - 15s 425ms/step - loss: 0.2118 - accuracy: 0.9157 - val_loss: 0.2119 - val_accuracy: 0.9320\n",
      "Epoch 4/50\n",
      "36/36 [==============================] - 15s 416ms/step - loss: 0.1531 - accuracy: 0.9331 - val_loss: 0.1541 - val_accuracy: 0.9320\n",
      "Epoch 5/50\n",
      "36/36 [==============================] - 15s 422ms/step - loss: 0.1171 - accuracy: 0.9461 - val_loss: 0.3134 - val_accuracy: 0.8900\n",
      "Epoch 6/50\n",
      "36/36 [==============================] - 15s 410ms/step - loss: 0.0779 - accuracy: 0.9679 - val_loss: 0.1673 - val_accuracy: 0.9515\n",
      "Epoch 7/50\n",
      "36/36 [==============================] - 15s 419ms/step - loss: 0.0809 - accuracy: 0.9626 - val_loss: 0.1481 - val_accuracy: 0.9482\n",
      "Epoch 8/50\n",
      "36/36 [==============================] - 16s 436ms/step - loss: 0.0491 - accuracy: 0.9844 - val_loss: 0.1204 - val_accuracy: 0.9612\n",
      "Epoch 9/50\n",
      "36/36 [==============================] - 15s 404ms/step - loss: 0.0383 - accuracy: 0.9878 - val_loss: 0.1219 - val_accuracy: 0.9612\n",
      "Epoch 10/50\n",
      "36/36 [==============================] - 15s 415ms/step - loss: 0.0280 - accuracy: 0.9896 - val_loss: 0.1144 - val_accuracy: 0.9579\n",
      "Epoch 11/50\n",
      "36/36 [==============================] - 15s 413ms/step - loss: 0.0250 - accuracy: 0.9913 - val_loss: 0.1100 - val_accuracy: 0.9612\n",
      "Epoch 12/50\n",
      "36/36 [==============================] - 15s 405ms/step - loss: 0.0259 - accuracy: 0.9904 - val_loss: 0.1263 - val_accuracy: 0.9579\n",
      "Epoch 13/50\n",
      "36/36 [==============================] - 15s 413ms/step - loss: 0.0219 - accuracy: 0.9930 - val_loss: 0.1170 - val_accuracy: 0.9612\n",
      "Epoch 14/50\n",
      "36/36 [==============================] - 15s 421ms/step - loss: 0.0322 - accuracy: 0.9887 - val_loss: 0.1355 - val_accuracy: 0.9385\n",
      "Epoch 15/50\n",
      "36/36 [==============================] - 15s 407ms/step - loss: 0.0242 - accuracy: 0.9930 - val_loss: 0.1162 - val_accuracy: 0.9579\n",
      "Epoch 16/50\n",
      "36/36 [==============================] - 16s 454ms/step - loss: 0.0185 - accuracy: 0.9930 - val_loss: 0.0992 - val_accuracy: 0.9644\n",
      "Epoch 17/50\n",
      "36/36 [==============================] - 16s 440ms/step - loss: 0.0165 - accuracy: 0.9957 - val_loss: 0.0898 - val_accuracy: 0.9644\n",
      "Epoch 18/50\n",
      "36/36 [==============================] - 16s 433ms/step - loss: 0.0190 - accuracy: 0.9948 - val_loss: 0.2381 - val_accuracy: 0.9223\n",
      "Epoch 19/50\n",
      "36/36 [==============================] - 15s 425ms/step - loss: 0.0189 - accuracy: 0.9939 - val_loss: 0.1312 - val_accuracy: 0.9547\n",
      "Epoch 20/50\n",
      "36/36 [==============================] - 15s 422ms/step - loss: 0.0149 - accuracy: 0.9957 - val_loss: 0.1099 - val_accuracy: 0.9644\n",
      "Epoch 21/50\n",
      "36/36 [==============================] - 15s 427ms/step - loss: 0.0160 - accuracy: 0.9948 - val_loss: 0.1196 - val_accuracy: 0.9612\n",
      "Epoch 22/50\n",
      "36/36 [==============================] - 15s 422ms/step - loss: 0.0138 - accuracy: 0.9957 - val_loss: 0.0970 - val_accuracy: 0.9644\n",
      "Epoch 23/50\n",
      "36/36 [==============================] - 15s 430ms/step - loss: 0.0159 - accuracy: 0.9939 - val_loss: 0.0883 - val_accuracy: 0.9612\n",
      "Epoch 24/50\n",
      "36/36 [==============================] - 15s 426ms/step - loss: 0.0130 - accuracy: 0.9957 - val_loss: 0.0988 - val_accuracy: 0.9644\n",
      "Epoch 25/50\n",
      "36/36 [==============================] - 15s 430ms/step - loss: 0.0121 - accuracy: 0.9965 - val_loss: 0.1058 - val_accuracy: 0.9644\n",
      "Epoch 26/50\n",
      "36/36 [==============================] - 15s 425ms/step - loss: 0.0121 - accuracy: 0.9957 - val_loss: 0.1093 - val_accuracy: 0.9644\n",
      "Epoch 27/50\n",
      "36/36 [==============================] - 15s 416ms/step - loss: 0.0109 - accuracy: 0.9965 - val_loss: 0.1043 - val_accuracy: 0.9644\n",
      "Epoch 28/50\n",
      "36/36 [==============================] - 14s 402ms/step - loss: 0.0098 - accuracy: 0.9974 - val_loss: 0.1006 - val_accuracy: 0.9612\n",
      "Epoch 29/50\n",
      "36/36 [==============================] - 15s 409ms/step - loss: 0.0091 - accuracy: 0.9974 - val_loss: 0.1047 - val_accuracy: 0.9579\n",
      "Epoch 30/50\n",
      "36/36 [==============================] - 15s 410ms/step - loss: 0.0089 - accuracy: 0.9974 - val_loss: 0.1050 - val_accuracy: 0.9676\n",
      "Epoch 31/50\n",
      "36/36 [==============================] - 15s 404ms/step - loss: 0.0164 - accuracy: 0.9930 - val_loss: 0.1154 - val_accuracy: 0.9579\n",
      "Epoch 32/50\n",
      "36/36 [==============================] - 14s 402ms/step - loss: 0.0206 - accuracy: 0.9904 - val_loss: 0.1741 - val_accuracy: 0.9450\n",
      "Epoch 33/50\n",
      "36/36 [==============================] - 16s 435ms/step - loss: 0.0179 - accuracy: 0.9948 - val_loss: 0.1236 - val_accuracy: 0.9482\n",
      "Epoch 34/50\n",
      "36/36 [==============================] - 15s 415ms/step - loss: 0.0109 - accuracy: 0.9965 - val_loss: 0.0970 - val_accuracy: 0.9644\n",
      "Epoch 35/50\n",
      "36/36 [==============================] - 15s 408ms/step - loss: 0.0104 - accuracy: 0.9957 - val_loss: 0.0995 - val_accuracy: 0.9612\n",
      "Epoch 36/50\n",
      "36/36 [==============================] - 16s 432ms/step - loss: 0.0108 - accuracy: 0.9965 - val_loss: 0.1122 - val_accuracy: 0.9515\n",
      "Epoch 37/50\n",
      "36/36 [==============================] - 15s 417ms/step - loss: 0.0079 - accuracy: 0.9974 - val_loss: 0.1257 - val_accuracy: 0.9515\n",
      "Epoch 38/50\n",
      "36/36 [==============================] - 15s 427ms/step - loss: 0.0121 - accuracy: 0.9948 - val_loss: 0.1480 - val_accuracy: 0.9547\n",
      "Epoch 39/50\n",
      "36/36 [==============================] - 15s 429ms/step - loss: 0.0533 - accuracy: 0.9800 - val_loss: 0.2020 - val_accuracy: 0.9288\n",
      "Epoch 40/50\n",
      "36/36 [==============================] - 15s 404ms/step - loss: 0.0287 - accuracy: 0.9896 - val_loss: 0.1391 - val_accuracy: 0.9515\n",
      "Epoch 41/50\n",
      "36/36 [==============================] - 16s 447ms/step - loss: 0.0166 - accuracy: 0.9922 - val_loss: 0.1197 - val_accuracy: 0.9579\n",
      "Epoch 42/50\n",
      "36/36 [==============================] - 17s 468ms/step - loss: 0.0102 - accuracy: 0.9965 - val_loss: 0.1784 - val_accuracy: 0.9385\n",
      "Epoch 43/50\n",
      "36/36 [==============================] - 16s 432ms/step - loss: 0.0078 - accuracy: 0.9974 - val_loss: 0.1549 - val_accuracy: 0.9450\n",
      "Epoch 44/50\n",
      "36/36 [==============================] - 15s 418ms/step - loss: 0.0111 - accuracy: 0.9948 - val_loss: 0.1459 - val_accuracy: 0.9547\n",
      "Epoch 45/50\n",
      "36/36 [==============================] - 17s 460ms/step - loss: 0.0073 - accuracy: 0.9974 - val_loss: 0.1312 - val_accuracy: 0.9547\n",
      "Epoch 46/50\n",
      "36/36 [==============================] - 16s 439ms/step - loss: 0.0066 - accuracy: 0.9974 - val_loss: 0.1293 - val_accuracy: 0.9515\n",
      "Epoch 47/50\n",
      "36/36 [==============================] - 17s 473ms/step - loss: 0.0080 - accuracy: 0.9974 - val_loss: 0.1257 - val_accuracy: 0.9547\n",
      "Epoch 48/50\n",
      "36/36 [==============================] - 16s 457ms/step - loss: 0.0072 - accuracy: 0.9974 - val_loss: 0.1087 - val_accuracy: 0.9579\n",
      "Epoch 49/50\n",
      "36/36 [==============================] - 15s 425ms/step - loss: 0.0065 - accuracy: 0.9974 - val_loss: 0.1105 - val_accuracy: 0.9612\n",
      "Epoch 50/50\n",
      "36/36 [==============================] - 16s 432ms/step - loss: 0.0069 - accuracy: 0.9965 - val_loss: 0.1142 - val_accuracy: 0.9579\n"
     ]
    }
   ],
   "source": [
    "\n",
    "model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])\n",
    "\n",
    "r = model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data =(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "final-blond",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the loss\n",
    "import matplotlib.pyplot as plt\n",
    "plt.plot(r.history['loss'], label='train loss')\n",
    "plt.plot(r.history['val_loss'], label='val loss')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# plot the accuracy\n",
    "plt.plot(r.history['accuracy'], label='train acc')\n",
    "plt.plot(r.history['val_accuracy'], label='val acc')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "moved-logging",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d (Conv2D)              (None, 150, 150, 16)      160       \n",
      "_________________________________________________________________\n",
      "max_pooling2d (MaxPooling2D) (None, 75, 75, 16)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_1 (Conv2D)            (None, 75, 75, 32)        4640      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 37, 37, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 37, 37, 64)        18496     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 18, 18, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 18, 18, 128)       73856     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 9, 9, 128)         0         \n",
      "_________________________________________________________________\n",
      "flatten (Flatten)            (None, 10368)             0         \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 256)               2654464   \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 128)               32896     \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 1)                 129       \n",
      "=================================================================\n",
      "Total params: 2,784,641\n",
      "Trainable params: 2,784,641\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7e329413",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('CNN+SVM_FOR_CYCLONE_DETECTION.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a7848987",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "model1 = load_model('CNN+SVM_FOR_CYCLONE_DETECTION.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "142d7a47",
   "metadata": {},
   "outputs": [],
   "source": [
    "#_, acc = model.evaluate(X_test, y_test)\n",
    "#print(\"Accuracy = \", (acc * 100.0), \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fa1e97db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[183   5]\n",
      " [ 13 108]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.97      0.95       188\n",
      "           1       0.96      0.89      0.92       121\n",
      "\n",
      "    accuracy                           0.94       309\n",
      "   macro avg       0.94      0.93      0.94       309\n",
      "weighted avg       0.94      0.94      0.94       309\n",
      "\n"
     ]
    }
   ],
   "source": [
    "mythreshold = 0.5\n",
    "\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "y_pred1 = (model1.predict(X_test)>= mythreshold).astype(int)\n",
    "cm=confusion_matrix(y_test, y_pred1)\n",
    "\n",
    "print(cm)\n",
    "\n",
    "print(classification_report(y_test,y_pred1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "db49f61a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[183   5]\n",
      " [ 10 111]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.97      0.96       188\n",
      "           1       0.96      0.92      0.94       121\n",
      "\n",
      "    accuracy                           0.95       309\n",
      "   macro avg       0.95      0.95      0.95       309\n",
      "weighted avg       0.95      0.95      0.95       309\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model2 = load_model('CNN_FOR_CYCLONE_DETECTION.h5')\n",
    "\n",
    "y_pred2 = (model2.predict(X_test)>= mythreshold).astype(int)\n",
    "cm=confusion_matrix(y_test, y_pred2)  \n",
    "print(cm)\n",
    "\n",
    "print(classification_report(y_test,y_pred2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "aaa1e977",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[166  22]\n",
      " [ 13 108]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.88      0.90       188\n",
      "           1       0.83      0.89      0.86       121\n",
      "\n",
      "    accuracy                           0.89       309\n",
      "   macro avg       0.88      0.89      0.88       309\n",
      "weighted avg       0.89      0.89      0.89       309\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model3 = load_model('ANN_FOR_CYCLONE_DETECTION.h5')\n",
    "y_pred3 = (model3.predict(X_test)>= mythreshold).astype(int)\n",
    "cm=confusion_matrix(y_test, y_pred3)  \n",
    "print(cm)\n",
    "\n",
    "print(classification_report(y_test,y_pred3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "083c63f6",
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
