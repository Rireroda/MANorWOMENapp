import os,glob
import numpy as np
from PIL import Image
import tensorflow as tf

base_dir = "./facedata/"
 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
train_man = os.path.join(train_dir, '/man')
test_man = os.path.join(test_dir, '/man')
train_woman = os.path.join(train_dir, 'woman')
test_woman = os.path.join(test_dir, 'woman')
dir_list = [train_dir, test_dir, train_man, test_man, train_woman, test_woman]
 
org_man_dir = './facedata/man/'
org_woman_dir = './facedata/woman/'
man_faces = os.listdir(org_man_dir)
woman_faces = os.listdir(org_woman_dir)

classes = ["man","woman"]
num_classes = len(classes)
image_size = 50
num_testdata = 50
 
X_train = []
X_test = []
Y_train = []
Y_test = []

#データの水増し

 
#男性女性のデータを選択するパート
for index, classlabel in enumerate(classes):
    photos_dir = "./facedata/" + classlabel
    files = glob.glob(photos_dir+"/*.jpg")
 
#画像データを50×50のnumpy形式に変換
    for i, file in enumerate(files):
        if i >= 253: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image)
        
#50枚をテストデータにする
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
            
        else:
            X_train.append(data)
            Y_train.append(index)
            
            #角度を5度づつ、±30度までずらしてｎ増し（trainデータのみ）
            for angle in range(-30,30,5):
                num = 1
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                
                X_train.append(data)
                Y_train.append(index)
 
                #反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)
                
                num += num

import matplotlib.pyplot as plt
plt.figure
#plt.imshow(img_r)
plt.figure

print(len(X_train),len(X_test),len(Y_train),len(Y_test))


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)
 
#分割したデータを保存
xy = (X_train,X_test,y_train,y_test)
np.save("./face_aug.npy",xy)

#Kerasインポート
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dropout,Flatten,Dense
from keras.utils import np_utils
import keras
import numpy as np
 
#データの正規化、カテゴリカル化
X_train = X_train.astype("float")/256
X_test = X_test.astype("float")/256
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

model = Sequential()
 
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape = X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(28,28)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',input_shape=(28,28)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
 
 
#モデル１　コンパイル
model.compile(loss=keras.losses.binary_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              #optimizer=keras.optimizers.Adam(learning_rate),
              #optimizer=keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])
#optimizer=keras.optimizers.Adam(learning_rate),
 
#学習を開始
hist = model.fit(X_train, y_train,
                 batch_size=128,
                 epochs=8,
                 validation_split=0.1,
                 verbose=1)
 
#スコア
scores1 = model.evaluate(X_test, y_test)
print('loss = {:.4} '.format(scores1[0]))
print('accuracy = {:.4%} '.format(scores1[1]))

model.save('./man_woman_cnn2.h5')

