# Students_performance-analysis_ml
The aim is to extract high-level knowledge from raw data and provide useful automated tools for the education domain. Using school reports and questionnaires, real-world data was gathered recently. Under classification and regression tasks, the two core classes (mathematics and Portuguese) were modelled. The findings demonstrate that good predictive accuracy is possible. Although past evaluations significantly impact student achievement, an explanatory analysis revealed that other factors (such as the number of absences, the job and education of parents, and alcohol consumption) are also important. More effective student prediction tools are to be developed as a direct result of this research, improving educational quality, and improving school resource management.
SOLUTION REQUIREMENTS 

Two datasets about performance in two distinct subjects Mathematics (student-mat.csv) and Portuguese language (student-por.csv) are provided to build a predictive model to make informed decisions. The dataset used for implementation is taken from  UCI repository  named as Student Performance Data Set. This information pertains to secondary school student achievement in two Portuguese schools. The two datasets were modelled using binary/five-level classification and regression tasks in [Cortez and Silva, 2008]. There are a total of 33 columns in each dataset including student grades, demographic information as well as social and school-related variables. A comparison between the descriptive summaries of both datasets, student_mat and student_por, did not reveal anything unusual.

The primary goal in completing this assignment is to analyse and manage data using statistical approaches to comprehend transaction data and determine accuracy using regression and classification models. Note that the target attribute G3 has a strong relationship with the attributes G2 and G1. This is because G3 is the final year grade (3rd period), whereas G1 and G2 are the first and second-period grades. Predicting G3 without G2 and G1 is more difficult, but it is far more useful. Here forest regression and classification using models like Support vector machine, Random Forest classifier and Multi-Layer Perceptron Neural Networks. 



IMPLEMENTATION OF SOLUTION 

To build a predictive model to make informed decisions, 3 modules and a notebook file are needed. main.ipynb, Classification.py, eda.py, manageData.py, reg.py. 

1: Manage Data module(manageData.py) has a class loadDF and 3 function:




(i)	createDF()
* Function that converts two datasets into a dataframe. 
Parameter: Students details mergerd dataframe
Return: The result is a collection of merges dataframe(student), Mathematics dataframe(student_por), and Portuguese dataframe (studentmat) dataframes. 

(ii)	The dataPro() function is used to perform data processing. This function returns a student dataframe with a new final Grade column.

(iii)	Ifmissing() is used to cleaning the dataframe with student dataframe as parameter. In this function sum of null is calculated and printed. Since the datafront has no null values 
 


2: EDA module: This module is used for display the statistic report and for graph visualization.
The function used are given below:
(i)	+ statDetails(self, student):return df
(ii)	    -corrMatrix(self, data):        
(iii)	    -barGraphStudentPerformance(self,std):
(iv)	    - piechart(self, data):

3: Regression module(reg.py) has a class rfr with six functions:
(i)	encodedata() is a function used to encode and transform using label encoder. The function has a student dataframe as argument. This function returns an encoded dataframe.

(ii)	createFeature() is function used to create a feature dataframe to encode them into dummies. This function returns a dataframe that has integer column dataframe.


(iii)	dummydata(classes,features) is used to encode categorical variable for model fitting. This function returns a new encoded dataframe.

(iv)	mse() and R2() are two functions to that is used to calculate mean squared error and r2 score and the return this values. The parameters are ytest and prediction.

(v)	gridSearchrf() this functionis used to find the best parameter for regressor. The arguments are the xtrain and ytrain.  This function returns the best parameter.


(vi)	rfregressor() this function is used to apply the random forest regressor algorithm. The return value is dataframe with true and predicted values, predicted value and model fit value
 
(vii)	featureImportanceGraph() this function is used to print the feature performance graph. The parameters are fit model value(model_rf) and features. 

    
4: Classification- This module is used to predict the results using classification models like Support vector machine, Random Forest classifier and Multi-Layer Perceptron Neural Networks. Functions used are listed below: 
(i)	gridsearchClass(): return bestComb
(ii)	 rfc(): return rfcRediction, y_pred_rfc, rd_model 
(iii)	 smoterfc():        
(iv)	 svm():
(v)	   smotesvm():
(vi)	 mpl():
(vii)	 adsnmpl():
