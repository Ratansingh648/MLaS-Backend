from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class UploadData():
	def __init__(self,url,name):
		self.url = url
		self.data = self.readData()
		self.name = name

	def readData(self):
		dataset = pd.read_csv(self.url)
		return dataset.values


class RemoveMissingData():
	"""docstring for RemoveMissingData"""
	def __init__(self,data,missingColumns,missingValues,method):
		self.missingValues = missingValues 
		self.data = data
		self.missingColumns = missingColumns
		self.strategy = method
		self.cleanData = self.getCleanData()

	def getCleanData(self):
		imputer = Imputer(missing_values = self.missingValues,strategy = self.strategy, axis = 0)
		Data = self.data
		print(Data.shape)
		Data[:,self.missingColumns:(self.missingColumns+1)] = imputer.fit_transform(Data[:,self.missingColumns:(self.missingColumns+1)])
		return Data


class LabelEncoding():
	"""docstring for ClassName"""
	def __init__(self, data, column):
		self.data = data
		self.column = column
		self.cleanData = self.getCleanData()

	def getCleanData(self):
		labelEncoder = LabelEncoder()
		Data = self.data
		Data[:,self.column] = labelEncoder.fit_transform(Data[:,self.column])
		return Data

class OneHotEncoding():
	"""docstring for ClassName"""
	def __init__(self, data,column):
		self.data = data
		self.column = column
		self.cleanData = self.getCleanData()

	def getCleanData(self):
		oneHotEncoder = OneHotEncoder(categorical_features = [self.column])
		Data = self.data
		Data = oneHotEncoder.fit_transform(Data).toarray()
		return Data

class FeatureTargetSeperator():
	"""docstring for ClassName"""
	def __init__(self,data,column):
		self.data = data
		self.column = column
		self.X = self.getX()
		self.Y = self.getY()

	def getX(self):
		data = self.data
		X = np.delete(data,self.column,1)
		return X

	def getY(self):
		data = self.data
		Y = data[:,self.column]
		return Y

		

class CrossValidationSplitting():
	def __init__(self,X,Y,split,state = 0):
		self.X = X
		self.Y = Y
		self.ratio = split
		self.state = state

		self.trainX, self.testX, self.trainY, self.testY = train_test_split(X,Y,test_size = self.ratio,random_state = self.state)


class LinearRegressionModel():
	"""docstring for ClassName"""
	def __init__(self, trainX, trainY, testX):
		self.trainX = trainX
		self.trainY = trainY
		self.testX = testX
		self.model = self.regressionModel()
		self.testY = self.predictY()
		

	def regressionModel(self):
		regressor = linear_model.LinearRegression()
		regressor.fit(self.trainX,self.trainY)
		return regressor

	def predictY(self):
		regressor = self.model
		y = regressor.predict(self.testX)
		return y

class Scaler():
	"""docstring for ClassName"""
	def __init__(self,trainX,testX):
		self.trainX = trainX
		self.testX = testX
		self.scaleTrainX = self.getScaleTrainX()
		self.scaleTestX = self.getScaleTestX()

	def getScaleTrainX(self):
		scaler = StandardScaler()
		scaleTrainX = scaler.fit_transform(self.trainX)
		return scaleTrainX

	def getScaleTestX(self):
		scaler = StandardScaler()
		scaleTestX = scaler.fit_transform(self.testX)
		return scaleTestX

		
		


		