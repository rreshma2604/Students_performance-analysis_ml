#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import style     


class edaClass(): 
    def statDetails(self, student):
        dataSeries = [student.count(),round(student.mean(),2),round(student.median(),2),round(student.var(),2),
                      round(student.std(),2),student.min(),student.quantile(q=0.25), student.quantile(q=0.5),
                      student.quantile(q=0.75), student.max(), student.kurtosis(),student.skew()]

        indexName= ['count','mean','median','variance','std','min','25%','50%','75%','Max','kurtosis','skewness']
        colName = list(student.select_dtypes(include='int64'))
        df =  pd.DataFrame(data =dataSeries,columns =colName , index = indexName)
        return df

    def corrMatrix(self, data):        
        # find the correlations
        corr = data.corr()
        plt.figure(figsize=(25,20))
        sns.heatmap(corr, annot=True, cmap="Reds")
        plt.title('Correlation Heatmap', fontsize=20) 
    
    def barGraphStudentPerformance(self,std):

        index =  ["low","medium","high"]
        travel = pd.crosstab(index=std.finalGrade, columns=std.internetAtHome)
        travel.plot.bar(colormap="mako_r", fontsize=12, figsize=(12,6))
        plt.title('Final Grade By Students with internet at home', fontsize=20)
        plt.ylabel('Number of Student', fontsize=12)
        plt.xlabel('Final Grade', fontsize=12)
        
        
        index = ["low","medium","high"]
        travel = pd.crosstab(index=std.finalGrade, columns=std.travelTime)
#         out_perc = travel.apply(perc).reindex(index)
        travel.plot.bar(colormap="mako_r", fontsize=12, figsize=(12,6))
        plt.title('Final Grade By Travel time', fontsize=20)
        plt.ylabel('Percentage of Student', fontsize=16)
        plt.xlabel('Final Grade', fontsize=16)
        
    def piechart(self, data):
        plt.figure(figsize=(10,10))
#         style.use('ggplot') 
        labels = data['Guardian'].value_counts()
        len(labels)
        labels.plot.pie(autopct ="%1.1f%%")
        
        plt.figure(figsize=(10,10))
#         style.use('ggplot') 
        labels = data['address'].value_counts()
        len(labels)
        labels.plot.pie(autopct ="%1.1f%%")
        
#        


