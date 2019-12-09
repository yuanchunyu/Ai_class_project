import os
from flask import Flask, render_template, request,json
import requests
import base64
import numpy
from PIL import Image
import io
import pickle
from keras import backend as K
import tensorflow as tf
import skimage.io
import keras

app = Flask(__name__)
graph = tf.get_default_graph()
loaded_model=pickle.load(open('lg.sav','rb'))
loaded_model1=pickle.load(open('svm.sav','rb'))
#loaded_model2=pickle.load(open('knn.sav','rb'))

loaded_model3=pickle.load(open('naive.sav','rb'))


#loaded_model4=pickle.load(open('cnn.sav','rb'))
def get_model():
    global model
#    model = load_model("model_new.h5")
    model=pickle.load(open('cnn.sav','rb'))
    model._make_predict_function()
    print("Model Loaded")
    return model
#loaded_model4=get_model()

def stringToBase64(s):
    return base64.b64encode(s.encode('utf-8'))


def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img
@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')
    
@app.route('/api/')
def api_get_name():
    name=request.args.get('name')
    
    
    #temp='''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'''
   # print(temp)
   # print(name)
    nae=name.split(';')
    name1=nae[1][7:]
   # print(type(name1))
    #print(type(temp))
    name1=name1.replace(" ", "+");
   # print(name1==temp)
    #print (name1)
    #print(temp)

    #xx=stringToBase64(name)
    #img=base64.b64decode(name)
    #xx=base64.b64encode(name).decode("utf-8")
    #im = numpy.frombuffer(base64.b64decode(img), numpy.uint8)
    #print (im.shape)
    img1=base64.b64decode(name1)
    image = Image.open(io.BytesIO(img1))
    size = 28, 28
    #im.thumbnail(size)
    #im = im.rotate(-90)
    im = image.resize(size)
    im = im.convert("L")
    
    loaded_model=pickle.load(open('lg.sav','rb'))
    print(loaded_model)
    ii = numpy.asarray(im).reshape(1,784)
    ii=ii/255
    
    ii1 = numpy.asarray(im).astype('float32').reshape(1,28,28,1)
    #ii1=ii1.astype('float32')
    #ii1 =ii1/255
    #ii1=numpy.ndarray(ii1)
    #import cv2
    #im = cv2.imread(image,mode='RGB')
    print(type(ii1))
    ii1=ii1.astype('float32')
    ii1=ii1/255
    pred_y = loaded_model.predict(ii)
    pred_y1 = loaded_model1.predict(ii)
   # pred_y2 = loaded_model2.predict(ii)
    pred_y3 = loaded_model3.predict(ii)
    print(ii1)
    #loaded_model4._make_predict_function()
   # loaded_model4._make_predict_function()
    global graph
    #pred_y4
    #with graph.as_default():
    loaded_model4=pickle.load(open('cnn.sav','rb'))
    pred_y4 = loaded_model4.predict(ii1)
    keras.backend.clear_session()
    print(pred_y4)
     
    re=numpy.where(pred_y4[0]==numpy.amax(pred_y4[0]))
    #print(pred_y4)
    #im = numpy.frombuffer(base64.b64decode(img1), numpy.uint8)
    print (ii.shape)
    #print(xx)
   # filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
   # with open(filename, 'wb') as f:
    #   f.write(img1)
    
    return json.jsonify({
        'name': 'lg predict: '+str(pred_y)+' <br>'
        +'svm predict: '+str(pred_y1)+' <br>'
       # +'knn predict: '+str(pred_y2)+' <br>'
        +'naive predict: '+str(pred_y3)+' <br>'
        
        +'cnn predict: '+str(re[0])+' <br>'
    })

@app.route('/ap/')
def api_get_name1():
    name=request.args.get('name')
    
    
    #temp='''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'''
   # print(temp)
   # print(name)
    nae=name.split(';')
    name1=nae[1][7:]
   # print(type(name1))
    #print(type(temp))
    name1=name1.replace(" ", "+");
   # print(name1==temp)
    #print (name1)
    #print(temp)

    #xx=stringToBase64(name)
    #img=base64.b64decode(name)
    #xx=base64.b64encode(name).decode("utf-8")
    #im = numpy.frombuffer(base64.b64decode(img), numpy.uint8)
    #print (im.shape)
    img1=base64.b64decode(name1)
    image = Image.open(io.BytesIO(img1))
    size = 28, 28
    #im.thumbnail(size)

    im = image.resize(size)
    im = im.convert("L")
    
    loaded_model=pickle.load(open('lg.sav','rb'))
    print(loaded_model)
    ii = numpy.asarray(im).reshape(1,784)
    ii=ii.astype('float32')
    ii=ii/255
    
    ii1 = numpy.asarray(im).astype('float32').reshape(1,28,28,1)
    #ii1=ii1.astype('float32')
    #ii1 =ii1/255
    #ii1=numpy.ndarray(ii1)
    #import cv2
    #im = cv2.imread(image,mode='RGB')
    print(type(ii1))
    ii1=ii1.astype('float32')
    ii1=ii1/255
    pred_y = loaded_model.predict(ii)
    pred_y1 = loaded_model1.predict(ii)
   # pred_y2 = loaded_model2.predict(ii)
    pred_y3 = loaded_model3.predict(ii)
    print(ii1)
    #loaded_model4._make_predict_function()
   # loaded_model4._make_predict_function()
    global graph
    #pred_y4
    #with graph.as_default():
    loaded_model4=pickle.load(open('cnn.sav','rb'))
    pred_y4 = loaded_model4.predict(ii1)
    keras.backend.clear_session()
    print(pred_y4)
     
    re=numpy.where(pred_y4[0]==numpy.amax(pred_y4[0]))
    #print(pred_y4)
    #im = numpy.frombuffer(base64.b64decode(img1), numpy.uint8)
    print (ii.shape)
    #print(xx)
   # filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
   # with open(filename, 'wb') as f:
    #   f.write(img1)
    
    return json.jsonify({
        'name': 'lg predict: '+str(pred_y)+' <br>'
        +'svm predict: '+str(pred_y1)+' <br>'
       # +'knn predict: '+str(pred_y2)+' <br>'
        +'naive predict: '+str(pred_y3)+' <br>'
        
        +'cnn predict: '+str(re[0])+' <br>'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', threaded=True)
    
    
    
