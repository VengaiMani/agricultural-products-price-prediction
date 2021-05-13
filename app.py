from flask import Flask,render_template,request,url_for
import pickle
import numpy as np
import configModel as VegModel;


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html',show="none")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    vegetable=[0,1,2,3,4,5]
    vegetableNames=["Beetroot","BitterGourd","Cabbage","Capsicum","Carrot","Cucumber"]
    l=[]
    maxVegProba=0.0
    maxVeg=0
    for i in vegetable:
        temp=int_features[0:3]
        temp.insert(0,i)
        final=[np.array(temp)]
        prediction=VegModel.get(model.predict(final))
        print("Prediction",prediction) 
        if(float(prediction)>maxVegProba):
            maxVegProba=float(prediction)
            maxVeg=i

    
    vname=vegetableNames[maxVeg].lower()
    text="Your predicted fruit or vegetable is "
    vname='./static/images/'+vname+'.jpg'

    print("Predicted fruit:",maxVeg)
    return render_template('index.html',pred=vegetableNames[maxVeg],vegImage=vname,yourFruit=text,show="inline-block")

if __name__=="__main__":
    app.run(debug=True)