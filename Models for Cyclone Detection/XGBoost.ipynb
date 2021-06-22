{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "comparative-small",
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
   "id": "adjacent-orleans",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Model, Sequential\n",
    "from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "import os\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "97d8fb0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_extractor = tf.keras.models.Sequential()\n",
    "\n",
    "feature_extractor.add( tf.keras.layers.Conv2D(filters=16,padding=\"same\",kernel_size=3, activation='relu', strides=2, input_shape=X_train.shape[1:] ))\n",
    "feature_extractor.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "\n",
    "feature_extractor.add( tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu' ))\n",
    "feature_extractor.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "feature_extractor.add( tf.keras.layers.Conv2D(filters=64,padding='same',kernel_size=3, activation='relu' ))\n",
    "feature_extractor.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "feature_extractor.add( tf.keras.layers.Conv2D(filters=128,padding='same',kernel_size=3, activation='relu' ))\n",
    "feature_extractor.add( tf.keras.layers.MaxPool2D(pool_size=2, strides=2 ))\n",
    "\n",
    "feature_extractor.add( tf.keras.layers.Flatten())\n",
    "\n",
    "#feature_extractor.add( tf.keras.layers.Dense(units=256, activation='relu' ))\n",
    "#feature_extractor.add( tf.keras.layers.Dense(units=128, activation='relu' ))\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "italic-tribune",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_for_training = feature_extractor.predict(X_train) #This is out X input to RF\n",
    "#X_for_training = feature_extractor.reshape(feature_extractor.shape[0], -1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "appointed-affect",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:43:55] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,\n",
       "              importance_type='gain', interaction_constraints='',\n",
       "              learning_rate=0.300000012, max_delta_step=0, max_depth=6,\n",
       "              min_child_weight=1, missing=nan, monotone_constraints='()',\n",
       "              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,\n",
       "              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,\n",
       "              tree_method='exact', validate_parameters=1, verbosity=None)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#XGBOOST\n",
    "import xgboost as xgb\n",
    "model = xgb.XGBClassifier()\n",
    "model.fit(X_for_training, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "final-convenience",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Send test data through same feature extractor process\n",
    "X_test_feature = feature_extractor.predict(X_test)\n",
    "#X_test_feature = X_test_feature.reshape(X_test_feature.shape[0], -1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "acute-camel",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now predict using the trained RF model. \n",
    "prediction = model.predict(X_test_feature)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "configured-original",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy =  0.912621359223301\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "print (\"Accuracy = \", metrics.accuracy_score(y_test, prediction))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "printable-preservation",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[174  14]\n",
      " [ 13 108]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.93      0.93      0.93       188\n",
      "           1       0.89      0.89      0.89       121\n",
      "\n",
      "    accuracy                           0.91       309\n",
      "   macro avg       0.91      0.91      0.91       309\n",
      "weighted avg       0.91      0.91      0.91       309\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Confusion Matrix - verify accuracy of each class\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "cm = confusion_matrix(y_test, prediction)\n",
    "print(cm)\n",
    "\n",
    "print(classification_report(y_test, prediction))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "federal-compound",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "#Check results on a few select images\n",
    "n=np.random.randint(0, X_test.shape[0])\n",
    "img = X_test[n]\n",
    "plt.imshow(img)\n",
    "input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)\n",
    "input_img_feature=feature_extractor.predict(input_img)\n",
    "#input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)\n",
    "prediction = model.predict(input_img_feature)[0] \n",
    "print(\"The prediction for this image is: \", prediction)\n",
    "print(\"The actual label for this image is: \", y_test[n])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "medium-exploration",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:49:18] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:50:29] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:51:39] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:52:44] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:53:47] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:54:40] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:55:39] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:56:56] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\envs\\tensorflow_gpu\\lib\\site-packages\\xgboost\\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].\n",
      "  warnings.warn(label_encoder_deprecation_msg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:58:13] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n"
     ]
    }
   ],
   "source": [
    "pickle_in = open(\"X_data.pickle\",\"rb\") \n",
    "X_data = pickle.load(pickle_in)\n",
    "\n",
    "pickle_in = open(\"y_data.pickle\",\"rb\") \n",
    "y_data = pickle.load(pickle_in)\n",
    "\n",
    "X_data = np.array(X_data/255.0) \n",
    "y_data = np.array(y_data)\n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "X = feature_extractor.predict(X_data)\n",
    "\n",
    "\n",
    "\n",
    "accuracies = cross_val_score(model, X, y_data, cv=10)\n",
    "accuracies.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8372ab23",
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
