from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


class LogisticModel():
      def __init__(self,urlx,urly):
          self.X=self.get_x(urlx)
          self.Y=self.get_Y(urly)
          self.model= self.fit_model()
          self.attributes = self.num_attributes()
          self.examples = self.num_examples()
          self.classes  =self.num_classes()

      def get_x(self,urlx):
          X=pd.read_csv(urlx)
          return X
      def get_Y(self,urly):
          Y=pd.read_csv(urly)
          Y = Y.values.ravel()
          return Y

      def num_attributes(self):
        num_attributes = self.X.shape[1]
        return num_attributes

      def num_examples(self):
        num_examples = self.X.shape[0]
        return num_examples

      def get_attributes(self):
        return self.attributes

      def num_classes(self):
        num_classes = list(set(list(self.Y)))
        return num_classes

      def get_classes(self):
        return self.classes

      def fit_model(self):
          logReg=linear_model.LogisticRegression()
          # print(self.X)
          # print(self.Y)
          model=logReg.fit(self.X, self.Y)
          return  model

      def get_model(self):
          return self.model


class KNNModel():
      def __init__(self,urlx,urly,k):
          self.X=self.get_x(urlx)
          self.Y=self.get_Y(urly)
          self.K = k
          self.model= self.fit_model()
          self.attributes = self.num_attributes()
          self.examples = self.num_examples()
          self.classes  =self.num_classes()

      def get_x(self,urlx):
          X=pd.read_csv(urlx)
          return X

      def get_Y(self,urly):
          Y=pd.read_csv(urly)
          Y=Y.values.ravel()
          return Y

      def num_attributes(self):
        num_attributes = self.X.shape[1]
        return num_attributes

      def num_examples(self):
        num_examples = self.X.shape[0]
        return num_examples

      def get_attributes(self):
        return self.attributes

      def num_classes(self):
        num_classes = list(set(list(self.Y)))
        return num_classes

      def get_classes(self):
        return self.classes

      def fit_model(self):
          knnModel=KNeighborsClassifier(n_neighbors = self.K)
          # print(self.X)
          # print(self.Y)
          model=knnModel.fit(self.X, self.Y)
          return  model

      def get_model(self):
          return self.model

      def get_k(self):
        return self.K



class DecisionTreeModel():
      def __init__(self,urlx,urly,criteria):
          self.X=self.get_x(urlx)
          self.Y=self.get_Y(urly)
          self.model= self.fit_model()
          self.attributes = self.num_attributes()
          self.examples = self.num_examples()
          self.classes  =self.num_classes()
          self.Criteria = criteria
          
      def get_x(self,urlx):
          X=pd.read_csv(urlx)
          return X
      def get_Y(self,urly):
          Y=pd.read_csv(urly)
          Y = Y.values.ravel()
          return Y

      def num_attributes(self):
        num_attributes = self.X.shape[1]
        return num_attributes

      def num_examples(self):
        num_examples = self.X.shape[0]
        return num_examples

      def get_attributes(self):
        return self.attributes

      def num_classes(self):
        num_classes = list(set(list(self.Y)))
        return num_classes

      def get_classes(self):
        return self.classes

      def fit_model(self,criteria):
          treeModel=tree.DecisionTreeClassifier(criterion = criteria)
          model=treeModel.fit(self.X, self.Y)
          return  model

      def get_model(self):
          return self.model

class MissingDataRemoval():
	"""docstring for MissingDataRemoval"""
	def __init__(self, urlx, missingValue,axis,strategy,column):
		self.X = self.get_x(urlx)
		self.axis = axis
		self.missingValue  = missingValue
		self.strategy = strategy
		self.column = column
		self.X_processed = self.get_processed_X()


	def get_x(self):
		return pd.read_csv(self.urlx)

	def get_processed_X(self):
		imputer = Imputer(missing_values = self.missingValue,strategy = self.strategy,axis = self.axis)
		X_processed[:,self.column] = imputer.fit_transform(X)
		return X_processed

class CategoricalData(object):
	"""docstring for CategoricalData"""
	def __init__(self, arg):
		super(CategoricalData, self).__init__()
		self.arg = arg
		

