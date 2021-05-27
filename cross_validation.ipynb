{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a5cb9ccd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "pickle_in = open(\"X_data.pickle\",\"rb\") \n",
    "X_data = pickle.load(pickle_in)\n",
    "\n",
    "pickle_in = open(\"y_data.pickle\",\"rb\") \n",
    "y_data = pickle.load(pickle_in)\n",
    "\n",
    "X_data = np.array(X_data/255.0) \n",
    "y_data = np.array(y_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9ec9f287",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential, load_model\n",
    "from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3ff8f0ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1 = load_model('VGG16_FOR_CYCLONE_DETECTION.h5')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d1ec6b78",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10/10 [==============================] - 12s 1s/step - loss: 0.1111 - accuracy: 0.9623\n",
      "accuracy: 96.23%\n",
      "10/10 [==============================] - 14s 1s/step - loss: 0.1881 - accuracy: 0.9281\n",
      "accuracy: 92.81%\n",
      "10/10 [==============================] - 13s 1s/step - loss: 0.1262 - accuracy: 0.9623\n",
      "accuracy: 96.23%\n",
      "10/10 [==============================] - 13s 1s/step - loss: 0.1882 - accuracy: 0.9315\n",
      "accuracy: 93.15%\n",
      "10/10 [==============================] - 13s 1s/step - loss: 0.1313 - accuracy: 0.9521\n",
      "accuracy: 95.21%\n"
     ]
    }
   ],
   "source": [
    "#k-fold\n",
    "seed = 7\n",
    "np.random.seed(seed)\n",
    "\n",
    "kfold = KFold(n_splits=5, shuffle=True, random_state=seed)\n",
    "\n",
    "cvscores = []\n",
    "\n",
    "for train, test in kfold.split(X_data, y_data):\n",
    "\n",
    "\t# Compile model\n",
    "\tmodel1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n",
    "\t# Fit the model\n",
    "\t#model1.fit(X_data[train], y_data[train], epochs=5, batch_size=32)\n",
    "\t# evaluate the model\n",
    "\tscores = model1.evaluate(X_data[test], y_data[test])\n",
    "\tprint(\"%s: %.2f%%\" % (model1.metrics_names[1], scores[1]*100))\n",
    "\tcvscores.append(scores[1] * 100)\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "1546f6ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94.73% \n"
     ]
    }
   ],
   "source": [
    "print(\"%.2f%% \" % (np.mean(cvscores)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f93adfcc",
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
