{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cbf44fc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import tensorflow as tf\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "24849bba",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.keras.models.load_model(\"CNN+SVM_FOR_CYCLONE_DETECTION.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "2f173d35",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This is not a cyclone image 0.4695589\n"
     ]
    }
   ],
   "source": [
    "def prepare(filepath):\n",
    "    image = cv2.imread(filepath , cv2.IMREAD_GRAYSCALE)\n",
    "    ret,image = cv2.threshold(image, 180, 220, cv2.THRESH_BINARY)\n",
    "\n",
    "    kernel = np.ones((2,2), np.uint8)\n",
    "    image = cv2.erode(image, kernel)\n",
    "\n",
    "    image = cv2.medianBlur(image, 1)\n",
    "    image = cv2.resize(image, (300,300)) \n",
    "\n",
    "    image = cv2.resize(image, None, fx = 1, fy = 1, interpolation = cv2.INTER_AREA) \n",
    "    image = image/255.0\n",
    "    #image = np.expand_dims(image, axis=0)\n",
    "    return image.reshape(-1,300,300,1)\n",
    "    \n",
    "\n",
    "prediction = model.predict([prepare('No_cyclone.jpg')])\n",
    "\n",
    "if(prediction <= 0):\n",
    "    result = \"cyclone\"\n",
    "else:\n",
    "    result = \"no_cyclone\"\n",
    "    \n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e9ba56e",
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
