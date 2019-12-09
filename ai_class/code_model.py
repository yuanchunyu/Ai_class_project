from keras.datasets import mnist
from keras import backend as K
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

X_train1=X_train.reshape(X_train.shape[0],784)
X_test1=X_test.reshape(X_test.shape[0],784)
X_train2=X_train.reshape(X_train.shape[0],28,28,1)

import pickle
import numpy
x=pickle.load(open('cnn.sav','rb'))
y=x.predict(X_train2[0:1])
result = numpy.where(y == numpy.amax(y))
 
print('Returned tuple of arrays :', result)
print('List of Indices of maximum element :', result[0]+1)
print('List of Indices of maximum element real :', Y_test[0])
import timeit
f= open("guru99.txt","w+")
print('svm............')



from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')
start = timeit.timeit()
svclassifier.fit(X_train1, Y_train)
end = timeit.timeit()
timerecord=end - start

#pickle.dump(svclassifier, open('svm.sav', 'wb'))
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
f.write("svm.............\r\n" )
f.write("time\r\n" )
f.write(str(timerecord)+"\r\n" )
predicted=svclassifier.predict(X_test1)
precision, recall, f1, _ = precision_recall_fscore_support(Y_test, predicted,average='weighted')
accuracy=accuracy_score(Y_test, predicted)
f.write('pression: '+str(precision)+"\r\n" )
f.write('recall: '+str(recall)+"\r\n" )
f.write('f1: '+str(f1)+"\r\n" )
f.write('accuracy: '+str(accuracy)+"\r\n" )
f.write(".............\r\n" )


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
start = timeit.timeit()
logisticRegr.fit(X_train1, Y_train)
end = timeit.timeit()
timerecord=end - start
#pickle.dump(logisticRegr, open('lg.sav', 'wb'))
f.write("lg.............\r\n" )
f.write("time\r\n" )
f.write(str(timerecord)+"\r\n" )
predicted=logisticRegr.predict(X_test1)
precision, recall, f1, _ = precision_recall_fscore_support(Y_test, predicted,average='weighted')
accuracy=accuracy_score(Y_test, predicted)
f.write('pression: '+str(precision)+"\r\n" )
f.write('recall: '+str(recall)+"\r\n" )
f.write('f1: '+str(f1)+"\r\n" )
f.write('accuracy: '+str(accuracy)+"\r\n" )
f.write(".............\r\n" )
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
start = timeit.timeit()
mnb.fit(X_train1, Y_train)
end = timeit.timeit()
timerecord=end - start
#pickle.dump(mnb, open('naive.sav', 'wb'))
f.write("mnb.............\r\n" )
f.write("time\r\n" )
f.write(str(timerecord)+"\r\n" )
predicted=mnb.predict(X_test1)
precision, recall, f1, _ = precision_recall_fscore_support(Y_test, predicted,average='weighted')
accuracy=accuracy_score(Y_test, predicted)
f.write('pression: '+str(precision)+"\r\n" )
f.write('recall: '+str(recall)+"\r\n" )
f.write('f1: '+str(f1)+"\r\n" )
f.write('accuracy: '+str(accuracy)+"\r\n" )
f.write(".............\r\n" )
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
start = timeit.timeit()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

end = timeit.timeit()
timerecord=end - start
f.write("cnn.............\r\n" )
f.write("time\r\n" )
f.write(str(timerecord)+"\r\n" )
predicted=model.predict(x_test)
#pickle.dump(model, open('cnn.sav', 'wb'))
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted,average='weighted')
accuracy=accuracy_score(Y_test, predicted)
f.write('pression: '+str(precision)+"\r\n" )
f.write('recall: '+str(recall)+"\r\n" )
f.write('f1: '+str(f1)+"\r\n" )
f.write('accuracy: '+str(accuracy)+"\r\n" )
f.write(".............\r\n" )
f.close()








