from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np
app=Flask(__name__)
# df=pd.read_csv("dia.csv")
model=pickle.load(open("model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

@app.route('/',methods=['GET'])
def awal():
    return "Flask Vercel Example - Hello World", 200

@app.route('/contacts',methods=['GET'])
def contacts():
    return render_template('contacts.html')
@app.route('/health',methods=['GET'])
def health():
    return render_template('health.html')
@app.route('/menu',methods=['GET'])
def menu():
    return render_template('menu.html')
@app.route('/over',methods=['GET'])
def over():
    return render_template('over.html')
@app.route('/under',methods=['GET'])
def under():
    return render_template('under.html')



@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features[2]=features[2]/100 #convert cm to m
    features.append(features[3]/(features[2]*features[2]))
    features_arr = [np.array(features)]


    col=["Age","Height","Weight","NCP","bmi","CAEC","FAF"] 
    test=pd.DataFrame(data=features_arr,columns=["Gender","Age","Height","Weight","family_history_with_overweight","NCP","CAEC","FAF","bmi"])

    test[col]=scaler.transform(test[col])
    output = model.predict(test)

    # return render_template("health.html", prediksi = output)
    link="https://www.youtube.com/watch?v=xvFZjo5PgG0"
    status="Normal"
    if output==0:
        status="Under-Health Bar"
        link="under"
    elif output==2:
        status="Over-Health Bar"
        link="over"
    
    send="<a href=\"{link}\"> {status}</a>"

    #
    return render_template('health.html',prediksi=send.format(status=status,link=link))

if __name__ == '__main__':
    app.run(debug = True)
