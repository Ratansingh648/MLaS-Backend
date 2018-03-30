from flask import Flask
from flask import request
import pickle
from ML import LogisticModel,KNNModel
from flask_pymongo import PyMongo
from bson.binary import Binary

app = Flask(__name__)
mongo = PyMongo(app)


# Define end point
@app.route('/logisticRegression/predict', methods=['get'])
def getLogisticPrediction():
	model = mongo.db.model
	logisticRegressionParam = model.find_one({'Type':'Logistic Regression'})
	theModel= logisticRegressionParam['model']
	theAttributes = logisticRegressionParam['num_attributes']
	theClasses = logisticRegressionParam['num_classes']
	
	logReg = pickle.loads(theModel)
	num_attributes = pickle.loads(theAttributes)
	setPred = pickle.loads(theClasses)
	
	print(num_attributes)
	temp = []
	
	for i in range(1,num_attributes+1):
		temp.append(float(request.args.get('p'+str(i))))
	parametersList = [temp]
	
	pred = logReg.predict(parametersList)[0]
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

    theModel = pickle.dumps(lm.get_model())
    theAttributes = pickle.dumps(lm.get_attributes())
    theClasses = pickle.dumps(lm.get_classes())

    model = mongo.db.model
    model.insert({'Type':'Logistic Regression','model':theModel,'num_attributes':theAttributes,'num_classes':theClasses})
    return "Logistic Regression Model trained !"


@app.route('/knn/predict/', methods=['get'])
def getKnnPrediction():

	model = mongo.db.model
	knnParam = model.find_one({'Type':'KNN'})
	theModel= knnParam['model']
	theAttributes = knnParam['num_attributes']
	theClasses = knnParam['num_classes']
	
	knnModel = pickle.loads(theModel)
	num_attributes = pickle.loads(theAttributes)
	setPred = pickle.loads(theClasses)
	temp = []

	print(setPred)

	for i in range(1,num_attributes+1):
		temp.append(float(request.args.get('p'+str(i))))
	parametersList = [temp]

	pred = knnModel.predict(parametersList)[0]

	print(pred)

	classLabel = "No matches"
	for i in range(0,len(setPred)):
		if setPred[i] == pred:
			classLabel = str(i)
	return classLabel


@app.route('/knn/generate_model',methods=['get'])
def trainKnnModel():
    urlx=request.args.get('urlx')
    urly=request.args.get('urly')
    k=int(request.args.get('k'))

    knnModel=KNNModel(urlx,urly,k)
    
    theModel = pickle.dumps(knnModel.get_model())
    theAttributes = pickle.dumps(knnModel.get_attributes())
    theClasses = pickle.dumps(knnModel.get_classes())
    theK = pickle.dumps(knnModel.get_k())

    model = mongo.db.model
    model.insert({'Type':'KNN','model':theModel,'num_attributes':theAttributes,'num_classes':theClasses,'k': theK})
    
    return "KNN Model trained !"



if __name__ == "__main__":
    app.run(debug='True')