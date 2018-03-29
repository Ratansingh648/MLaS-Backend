from flask import Flask
from flask import request
import pickle
from ML import LogisticModel,KNNModel

app = Flask(__name__)


# Define end point
@app.route('/logisticRegression/predict', methods=['get'])
def getLogisticPrediction():
    
    num_attributes = pickle.load(open('num_attributes','rb'))
    print(num_attributes)
    temp = [] 

    for i in range(1,num_attributes+1):
    	temp.append(float(request.args.get('p'+str(i))))

    parametersList = [temp]

    # Load model from disk
    logReg = pickle.load(open('logReg.pkl', 'rb'))

    # Predict
    pred = logReg.predict(parametersList)[0]
    setPred = pickle.load(open('num_classes','rb'))
    classLabel = "No matches"
    print(setPred)
    for i in range(0,len(setPred)):
    	if setPred[i] == pred:
    		classLabel = str(i)
    return classLabel


@app.route('/logisticRegression/generate_model',methods=['get'])
def trainLogisticModel():
    urlx=request.args.get('urlx')
    urly=request.args.get('urly')
    lm=LogisticModel(urlx,urly)
    pickle.dump(lm.get_model(), open('logReg.pkl', 'wb'))
    pickle.dump(lm.get_attributes(),open('num_attributes','wb'))
    pickle.dump(lm.get_classes(),open('num_classes','wb'))
    return "Logistic Regression Model trained !"


@app.route('/knn/predict/', methods=['get'])
def getKnnPrediction():
    
    num_attributes = pickle.load(open('num_attributes','rb'))
    temp = [] 

    for i in range(1,num_attributes+1):
    	temp.append(float(request.args.get('p'+str(i))))

    parametersList = [temp]


    # Load model from disk

    knnModel = pickle.load(open('knnModel.pkl', 'rb'))

    # Predict
    pred = knnModel.predict(parametersList)[0]
    setPred = pickle.load(open('num_classes','rb'))
    classLabel = "No matches"
    for i in range(0,len(setPred)):
    	if setPred[i] == str(pred):
    		classLabel = str(i)
    return classLabel


@app.route('/knn/generate_model',methods=['get'])
def trainKnnModel():
    urlx=request.args.get('urlx')
    urly=request.args.get('urly')
    k=int(request.args.get('k'))
    knnModel=KNNModel(urlx,urly,k)
    pickle.dump(knnModel.get_model(), open('knnModel.pkl', 'wb'))
    pickle.dump(knnModel.get_attributes(),open('num_attributes','wb'))
    pickle.dump(knnModel.get_classes(),open('num_classes','wb'))
    return "KNN Model trained !"



if __name__ == "__main__":
    app.run(debug='True')