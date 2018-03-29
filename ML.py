from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd



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



