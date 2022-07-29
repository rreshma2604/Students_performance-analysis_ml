# Regression
# Split the dataset on training and testing sets.
# Build Random Forest Regression model to predict a final year grade (G3).
# Evaluate your model using the test dataset.
# Plot the feature importance graph.
# Estimate mean square error and accuracy.
# Comment on your results

import pandas as pd
import numpy as np

# data visualization and missing values
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
seed=500
class rfr:
    
    def encodedata(self,stdencode):
    
        # categorical vars
        classes = stdencode.select_dtypes(include='object')
     
        # label encode final_grade
        le = preprocessing.LabelEncoder()
        
        #fit transform is used to transform all the string columns 
        for val in classes:  
            stdencode[val] = le.fit_transform(stdencode[val])
        return stdencode
    
    def createFeature(self,std):
        #processing or scaling the data
        # target and features
        target = std.finalScore
        intStudent = std.select_dtypes(include='int64')
        
        # regressors = [x for x in intStudent.columns if x not in ['finalScore']]
        regressors = [x for x in intStudent.columns]
        features = std.loc[:, regressors]
        return features
    
    def dummydata(self,classes,features):
        # Feature Encoding
        # In order to use categorical variables in the model fitting we need to encode them into dummies.
 

        # create new dataset with only continuous vars 
        dummies = pd.get_dummies(classes)

        # new dataset
        newDf = pd.concat([features,dummies], axis =1)
        return newDf

    def MSE(self,y_true,y_pred):
        mse = mean_squared_error(y_true, y_pred)
        print('MSE: %2.3f' % mse)
        return mse

    def R2(self,y_true,y_pred):    
        r2 = r2_score(y_true, y_pred)
        print('R2: %2.3f' % r2)     
        return r2

    def gridSearchrf(self,X_train, y_train):
        gsrfr = RandomForestRegressor(n_estimators = 200, random_state = seed,min_samples_split= 2, min_samples_leaf= 2)

        param_grid = {'max_depth': list(range(5, 10, 1)),
                  'max_features': [0.05, 0.1, 0.15, 0.2, 0.02,3]}

        grid_search = GridSearchCV(gsrfr, param_grid, 
                               n_jobs = -1, # no restriction on processor usage
                               cv = 10) # 5 fold cv
        grid_search.fit(X_train, y_train)
    
        bestPara = grid_search.best_params_
        print('Best combination:', grid_search.best_params_);
        return bestPara

    def rfregressor(self,max_depth,max_features,X_train, y_train, X_test, y_test):
            
        # with best parameters for 1000 estimators
        rfr = RandomForestRegressor(max_features = max_features, max_depth = max_depth, 
                                    n_estimators = 1000, random_state = seed)
        model_rf = rfr.fit(X_train, y_train)

        score_train = cross_val_score(rfr, X_train, y_train).mean()
#         score_test = cross_val_score(rfr, X_test, y_test, cv = 5).mean()

        print("Score with the train set = %.2f" % (score_train*100))

        # predictions
        y_pred_rf = model_rf.predict(X_test)
        d = {'true' : list(y_test),'predicted' : pd.Series(y_pred_rf)}
        dataf = pd.DataFrame(d).head()
        return (dataf,y_pred_rf,model_rf)
    
    def featureImportanceGraph(self, model_rf,features):
        # Let's see which features contributed to the result:
        plt.figure(figsize=(15,10))
        # first ten importances 
        importances = model_rf.feature_importances_[:35]
        indices = np.argsort(importances)

        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color = "r", align = 'center')
        plt.yticks(range(len(indices)), features)
        plt.xlabel('Relative Importance')
        plt.show()
    
    