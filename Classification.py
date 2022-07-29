# Classification
# Based on the final grade, divide the students into 3 categories. For example, poor achieving students (low), average achieving (medium), well achieving (high).

# Investigate class imbalance problem by producing the plot of the class distribution.
# If there is presence of class imbalance problem, use at least 2 techniques to balance the class distribution (Algorithm or Sampling technique).

# Build three classification models (Support vector machine, Random Forest classifier and Multi-Layer Perceptron Neural Networks).
# Evaluate your models using test dataset and provide the confusion matrix for all models.

# Report and compare performance of the models in terms of accuracy, precision, recall and F1-Score.
# Draw conclusions and provide recommendations. Please provide justification for chosen methods.
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import ADASYN 
from sklearn.neural_network import MLPClassifier

seed = 500

class classificationModel:
    
    def gridsearchClass(self,X_train, y_train):
        rd =  RandomForestClassifier(n_estimators=1000, random_state = seed)
        param_grid = {'max_depth': list(range(5, 10, 1)),
                      'max_features' : [0.05, 0.1, 0.15, 0.2,0.3,0.02]
                     }

        grid_search = GridSearchCV(rd, param_grid, 
                                   n_jobs = -1, # no restriction on processor usage
                                   cv = 5) # 7 fold cv
        grid_search.fit(X_train, y_train)
        bestComb = grid_search.best_params_
        print('Best combination:', grid_search.best_params_);
        return bestComb
        
    
    def rfc(self, max_features,max_depth,X_train, y_train,X_test, y_test):
        rd =  RandomForestClassifier(max_features = max_features, max_depth = max_depth)
        rd_model = rd.fit(X_train,y_train)

        score_train = cross_val_score(rd_model, X_train, y_train, cv = 5).mean()
        print("Score with the train set = %.2f" % (score_train*100))
        # predictions
        y_pred_rfc = rd_model.predict(X_test)

        d = {'true' : list(y_test),
             'predicted' : pd.Series(y_pred_rfc)
            }

        rfcRediction = pd.DataFrame(d).head()
        #printing the results
        print ('Confusion Matrix of Random Forest Classifier Model:')
        print(confusion_matrix(y_test, y_pred_rfc))
        print ('Accuracy Score of Random Forest Classifier Model before sampling:',accuracy_score(y_test, y_pred_rfc))
        print ('Report of Random Forest Classifier Model: ')
        print (classification_report(y_test, y_pred_rfc))
        return rfcRediction, y_pred_rfc, rd_model 
    
    def smoterfc(self,max_features,max_depth,X_train, y_train,X_test, y_test):
        print("Smote Sampling Technique")
        sm = SMOTE(random_state=seed)
        X_rfc, y_rfc = sm.fit_resample(X_train, y_train)
        rd =  RandomForestClassifier(bootstrap= True, max_features=max_features,max_depth=max_depth , n_estimators= 200)
        rd_model = rd.fit(X_rfc,y_rfc)
        predict_rd = rd_model.predict(X_test) 
        scores = cross_val_score(rd_model, X_train, y_train, cv=5)
        accurancy=round(scores.mean()*100,2)
        print("Accuracy of Random Forest Classifier Model after sampling",round(scores.mean()*100,2),"%")
        
    def svm(self, X_train, y_train,X_test, y_test):
        svc = SVC(kernel='rbf', C=1, gamma='auto')
        svc_model = svc.fit(X_train,y_train)
        predict_svc = svc_model.predict(X_test)
        print("Classification Report of SVM \n")
        print(classification_report(y_test,predict_svc))
        scores = cross_val_score(svc_model, X_train, y_train, cv=5)
        accuracy=round(scores.mean()*100,2)
        print("Accuracy of SVM model :",round(scores.mean()*100,2),"%")
     
    def smotesvm(self, X_train, y_train,X_test, y_test):
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)        
        svc = SVC(kernel='rbf', C=1, gamma='auto')      
        #Fitting the training data        
        svc_model = svc.fit(X_res,y_res)
        #Predicting on test        
        predict_rd = svc_model.predict(X_test)
        print("With sampling")        
        #print(confusion_matrix(y_test, predict_rd))        
        print(classification_report(y_test, predict_rd))
        scores = cross_val_score(svc_model, X_train, y_train, cv=5)
        accurancy=round(scores.mean()*100,2)
        print("Accuracy of SVM Model after sampling",round(scores.mean()*100,2),"%")
        
    def mpl(self, X_train, y_train,X_test, y_test):
        print("WITHOUT SAMPLING - MLP")
        mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter = 300,activation = 'relu',solver = 'adam')
        mlp_model=mlp_clf.fit(X_train, y_train)
        predict_mlp = mlp_clf.predict(X_test)
        scores = cross_val_score(mlp_model, X_train, y_train, cv=5)
        accurancy=round(scores.mean()*100,2)
        print("Accuracy of MLP BEFORE sampling",round(scores.mean()*100,2),"%")
        accurancy=accuracy_score(y_test,predict_mlp)*100   
        print(accurancy)
        
    def adsnmpl(self, X_train, y_train,X_test, y_test):
        print("WITH SAMPLING - MLP")
        adsn = ADASYN(sampling_strategy='auto', random_state=seed, n_neighbors=5, n_jobs=None)
        X_adsn,Y_adsn = adsn.fit_sample(X_train, y_train) 
        mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter = 300,activation = 'relu',solver = 'adam')
        mlp_model=mlp_clf.fit(X_adsn, Y_adsn)
        predict_mlp = mlp_clf.predict(X_test)

        scores = cross_val_score(mlp_model, X_train, y_train, cv=5)
        accurancy=round(scores.mean()*100,2)
        print("Accuracy of MLP after sampling",round(scores.mean()*100,2),"%")

