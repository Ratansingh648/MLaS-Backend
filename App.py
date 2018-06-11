from flask import *
from flask import request
import pickle
from ML import LogisticModel,KNNModel,DecisionTreeModel
from flask_pymongo import PyMongo
from bson.binary import Binary
from MachineLearning import *
import datetime
import uuid
import random
import pandas as pd
import numpy as np

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


@app.route("/decision_tree/generate_model",methods=['get'])
def trainDecisionTreeModel():
	urlx = request.args.get('urlx')
	urly = request.args.get('urly')
	criteria = request.args.get('criteria')

	treeModel = DecisionTreeModel(urlx,urly,criteria)

	theModel = pickle.dumps(treeModel.get_model())
	theAttributes = pickle.dumps(treeModel.get_attributes())
	theClasses = pickle.dumps(treeModel.get_classes())
	theCriteria = pickle.dumps(treeModel.get_criteria())

	model = mongo.db.model
	model.insert({'Type':'Decision Tree','model':theModel,'num_attributes':theAttributes,'num_classes':theClasses,'criteria': theCriteria})
	return "Decision Tree Model trained !"

@app.route('/decision_tree/predict/', methods=['get'])
def getDecisionTreePrediction():

	model = mongo.db.model
	treeParam = model.find_one({'Type':'Decision Tree'})
	theModel= treeParam['model']
	theAttributes = treeParam['num_attributes']
	theClasses = treeParam['num_classes']
	
	treeModel = pickle.loads(theModel)
	num_attributes = pickle.loads(theAttributes)
	setPred = pickle.loads(theClasses)
	temp = []

	print(setPred)

	for i in range(1,num_attributes+1):
		temp.append(float(request.args.get('p'+str(i))))
	parametersList = [temp]

	pred = treeModel.predict(parametersList)[0]

	print(pred)

	classLabel = "No matches"
	for i in range(0,len(setPred)):
		if setPred[i] == pred:
			classLabel = str(i)
	return classLabel

@app.route('/upload_training_data/', methods=['get'])
def uploadData():
	url=request.args.get('url')
	datasetName = request.args.get('name')
	createdOn = datetime.datetime.utcnow()
	loadDataObject = UploadData(url,datasetName)
	displayType = 'Dataset'

	theData = pickle.dumps(loadDataObject.data)
	
	loadedRawDataDB = mongo.db.rawdata
	loadedCleanDataDB = mongo.db.data

	dataId = uuid.uuid4().hex
	loadedRawDataDB.insert({'URL':url,'Data':theData,'Name': datasetName,'Created On': createdOn, 'DataID':dataId})
	loadedCleanDataDB.insert({'URL':url,'Data':theData,'Name': datasetName,'Created On': createdOn, 'DataID':dataId , 'Previous ID':dataId})

	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':dataId,'Type':displayType})

	message = 'Raw Dataset from URL ' + str(url) +' has been loaded into database with name ' + str(datasetName) + 'With a Object ID of ' + str(dataId)
	return message 

@app.route('/remove_missing_data/',methods=['get'])
def removeMissingData():
	dataId = request.args.get('dataId')
	missingValues = request.args.get('missingValues')
	missingColumns = int(request.args.get('missingColumns'))
	strategy = request.args.get('strategy')
	displayType = 'Dataset'

	mongoObject = mongo.db.data.find_one({'DataID': dataId })
	theData = mongoObject['Data']
	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	data = pickle.loads(theData)
	newDataID = uuid.uuid4().hex

	removeMissingDataObj = RemoveMissingData(data,missingColumns,missingValues,strategy)
	cleanData = removeMissingDataObj.cleanData

	theCleanData = pickle.dumps(cleanData)
	cleanedDataDB = mongo.db.data
	cleanedDataDB.insert({'URL':url,'Data': theCleanData,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'Missing Data has been removed from the dataset - ' + str(name) + '. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/categorical_to_labeled/',methods= ['get'])
def categoricalToLabel():
	dataId = request.args.get('dataId')
	column = int(request.args.get('column'))
	displayType = 'Dataset'

	mongoObject = mongo.db.data.find_one({'DataID': dataId })
	theData = mongoObject['Data']
	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	data = pickle.loads(theData)
	newDataID = uuid.uuid4().hex

	labeledData = LabelEncoding(data,column)
	cleanData = labeledData.cleanData

	theCleanData = pickle.dumps(cleanData)
	cleanedDataDB = mongo.db.data
	cleanedDataDB.insert({'URL':url,'Data': theCleanData,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'The dataset - ' + str(name) + ' has been labeled. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/onehot_encoding/',methods = ['get'])
def oneHotEncoding():
	dataId = request.args.get('dataId')
	column = int(request.args.get('column'))
	displayType = 'Dataset'

	mongoObject = mongo.db.data.find_one({'DataID': dataId })
	theData = mongoObject['Data']
	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	data = pickle.loads(theData)
	newDataID = uuid.uuid4().hex

	oneHotEncodedData = OneHotEncoding(data,column)
	cleanData = oneHotEncodedData.cleanData

	theCleanData = pickle.dumps(cleanData)
	cleanedDataDB = mongo.db.data
	cleanedDataDB.insert({'URL':url,'Data': theCleanData,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'The dataset - ' + str(name) + ' has been one hot Encoded. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/seperate_feature_target/',methods = ['get'])
def featureTargetSeperator():
	dataId = request.args.get('dataId')
	targetColumn = int(request.args.get('targetColumn'))
	displayType = 'FeatureTarget'

	mongoObject = mongo.db.data.find_one({'DataID': dataId })
	theData = mongoObject['Data']
	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	data = pickle.loads(theData)
	newDataID = uuid.uuid4().hex

	featureTargetSeperatorObj = FeatureTargetSeperator(data,targetColumn)
	X = featureTargetSeperatorObj.X
	Y = featureTargetSeperatorObj.Y

	theX = pickle.dumps(X)
	theY = pickle.dumps(Y)

	featureTargetDB = mongo.db.featureTarget
	featureTargetDB.insert({'URL':url,'X': theX,'Y': theY , 'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'The dataset - ' + str(name) + ' has been Seperated in X and Y. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/train_test_split/',methods = ['get'])
def trainTestSplitting():
	dataId = request.args.get('dataId')
	split = float(request.args.get('split'))
	state = int(request.args.get('state'))
	displayType = 'CrossValidation'

	mongoObject = mongo.db.featureTarget.find_one({'DataID': dataId })
	theX=mongoObject['X']
	theY=mongoObject['Y']
	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	newDataID = uuid.uuid4().hex

	X = pickle.loads(theX)
	Y = pickle.loads(theY)

	splitObj = CrossValidationSplitting(X,Y,split,state)

	X_train = splitObj.trainX
	X_test = splitObj.testX
	Y_train = splitObj.trainY
	Y_test = splitObj.testY

	theX_train = pickle.dumps(X_train)
	theY_train = pickle.dumps(Y_train)
	theX_test = pickle.dumps(X_test)
	theY_test = pickle.dumps(Y_test)

	splitDB = mongo.db.splitted
	splitDB.insert({'URL':url,'X_train': theX_train,'Y_train': theY_train,'X_test':theX_test,'Y_test':theY_test,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'The dataset - ' + str(name) + ' has been splitted in training and testing attributes and targets. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/scale_data/',methods = ['get'])
def scalingData():
	dataId = request.args.get('dataId')
	displayType = 'CrossValidation'

	mongoObject = mongo.db.splitted.find_one({'DataID': dataId })
	theX_train = mongoObject['X_train']
	theX_test = mongoObject['X_test']
	theY_train = mongoObject['Y_train']
	theY_test= mongoObject['Y_test']

	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	
	trainX = pickle.loads(theX_train)
	testX = pickle.loads(theX_test)
	newDataID = uuid.uuid4().hex

	scaleDataObj = Scaler(trainX,testX)
	trainX = scaleDataObj.scaleTrainX
	testX = scaleDataObj.scaleTestX

	theX_train = pickle.dumps(trainX)
	theX_test = pickle.dumps(testX)

	splitDB = mongo.db.splitted
	splitDB.insert({'URL':url,'X_train': theX_train,'Y_train': theY_train,'X_test':theX_test,'Y_test':theY_test,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'The dataset - ' + str(name) + ' has been Scaled. The new Object ID of the Dataset is '+ str(newDataID)
	return message


@app.route('/linear_regression/', methods = ['get'])
def linearRegressorModel():
	dataId = request.args.get('dataId')
	displayType = 'Model'

	mongoObject = mongo.db.splitted.find_one({'DataID':dataId})
	theX_train = mongoObject['X_train']
	theY_train = mongoObject['Y_train']
	theX_test = mongoObject['X_test']
	theY_test = mongoObject['Y_test']

	name = mongoObject['Name']
	url = mongoObject['URL']
	createdOn = datetime.datetime.utcnow()
	
	trainX = pickle.loads(theX_train)
	testX = pickle.loads(theX_test)
	trainY = pickle.loads(theY_train)
	testY = pickle.loads(theY_test)
	newDataID = uuid.uuid4().hex

	LinearRegressionObj = LinearRegressionModel(trainX,trainY,testX)
	model = LinearRegressionObj.model
	theModel = pickle.dumps(model)
	y_pred = LinearRegressionObj.testY
	cost = np.mean((y_pred-testY)**2)
	theY_pred = pickle.dumps(y_pred)
	theCost = pickle.dumps(cost)

	modelDB = mongo.db.model
	modelDB.insert({'URL':url,'Model': theModel,'Prediction':theY_pred,'Cost':theCost,'Name': name,'Created On': createdOn,'DataID':newDataID,'Training DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'Linear model was fitted for the dataset - ' + str(name) + ' and the MSE was found as' + str(cost) + ' with Model ID as ' + str(newDataID)
	return message


@app.route('/linear_regression_prediction/', methods = ['get'])
def linearRegressionPredictor():
	dataId = request.args.get('dataId')
	modelId = request.args.get('modelId')
	displayType = 'Prediction'

	mongoDataObject = mongo.db.test.find_one({'dataId':dataId})
	mongoModelObject = mongo.db.model.find_one({'Model ID':modelId})
	theX = mongoDataObject['Data']
	theModel = mongoModelObject['Model']
	name = mongoDataObject['Name']
	url = mongoDataObject['URL']
	createdOn = datetime.datetime.utcnow()
	
	X = pickle.loads(theX)
	model = pickle.loads(theModel)
	newDataID = uuid.uuid4().hex

	y_pred = model.predict(X)
	theY_pred = pickle.dumps(y_pred)
	
	predictionDB = mongo.db.prediction
	modelDB.insert({'URL':url,'Prediction': theY_pred,'Name': name,'Created On': createdOn,'DataID':newDataID,'Previous DataID': dataId})
	
	uuidDB = mongo.db.uuid
	uuidDB.insert({'UUID':newDataID,'Type':displayType})

	message = 'Linear model predicted the result for given dataset - ' + str(name) + 'with Prediction Data ID as' + str(newDataID)
	return message



@app.route('/view_data/',methods = ['get'])
def viewData():
	dataId = request.args.get('dataId')
	mongoMainObject = mongo.db.uuid.find_one({'UUID': dataId })
	displayType = mongoMainObject['Type']

	if displayType == 'Dataset':
		mongoObject = mongo.db.data.find_one({'DataID': dataId })
		name=mongoObject['Name']
		theData = mongoObject['Data']
		data = pickle.loads(theData)
		data = pd.DataFrame(data).head()
		return render_template('dataset.html',tables=[data.to_html()],titles = ['na',name])

	elif displayType == 'FeatureTarget':
		mongoObject = mongo.db.featureTarget.find_one({'DataID': dataId })
		name=mongoObject['Name']
		theX = mongoObject['X']
		theY = mongoObject['Y']
		X = pickle.loads(theX)
		Y = pickle.loads(theY)
		X = pd.DataFrame(X).head()
		Y = pd.DataFrame(Y).head()
		return render_template('dataset.html',tables=[X.to_html(),Y.to_html()] ,titles = ['na','Feature','Target'])

	elif displayType == 'CrossValidation':
		mongoObject = mongo.db.splitted.find_one({'DataID': dataId })
		name=mongoObject['Name']
		theX_train = mongoObject['X_train']
		theY_train = mongoObject['Y_train']
		theX_test = mongoObject['X_test']
		theY_test = mongoObject['Y_test']
		trainX = pickle.loads(theX_train)
		trainY = pickle.loads(theY_train)
		testX = pickle.loads(theX_test)
		testY = pickle.loads(theY_test)
		X_train = pd.DataFrame(trainX).head()
		Y_train = pd.DataFrame(trainY).head()
		X_test = pd.DataFrame(testX).head()
		Y_test = pd.DataFrame(testY).head()
		return render_template('dataset.html',tables=[X_train.to_html(),Y_train.to_html(),X_test.to_html(),Y_test.to_html()] ,titles = ['na','Training Feature','Training Target','Test Feature','Test Target'])

	elif displayType == 'Prediction':
		mongoObject = mongo.db.prediction.find_one({'DataID': dataId })
		theY = mongoObject['Prediction']
		Y = pickle.loads(theY)
		Y = pd.DataFrame(Y).head()
		return render_template('dataset.html',tables=[Y.to_html()] ,titles = ['na','Predictions'])

	elif displayType == 'Model':
		mongoObject = mongo.db.model.find_one({'DataID': dataId })
		theY = mongoObject['Prediction']
		Y = pickle.loads(theY)
		Y = pd.DataFrame(Y).head()
		return render_template('dataset.html',tables=[Y.to_html()] ,titles = ['na','Predictions'])

	
	else:
		return 'No Such Object ID found.'
    
if __name__ == "__main__":
    app.run(debug='True')