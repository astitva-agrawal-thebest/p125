import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
X = np.load('image.npz')['arr_0']
y = pd.read_csv("data.csv")["labels"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def getPrediction(image):
    im_pil=Image.open(image)
    imgbw=im_pil.convert('L')
    imgresize=imgbw.resize((28,28),Image.ANTIALIAS)
    pf=20
    minpixel=np.percentile(imgresize,pf)
    imginvert=np.clip(imgresize-minpixel,0,255)
    maxpixel=np.max(imgresize)
    imginvert=np.asarray(imgresize)/maxpixel
    testsample=np.array(imginvert).reshape(1,784)
    testpred=clf.predict(testsample)
    return testpred[0]