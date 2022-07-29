import pandas as pd
class loadDf:
    def createDf(self):
        try: 
            student_por = pd.read_csv("student-por.csv",sep=";")
            student_mat = pd.read_csv("student-mat.csv",sep=";")
            student_por['subject']='Portuguese'
            student_mat['subject']='Maths'
            student =  pd.concat([student_por,student_mat])
    
            # rename column labels
            student.columns = ['school','gender','age','address','familySize','parentsStatus','motherEducation','fatherEducation',
                       'motherJob','fatherJob','Reason','Guardian','travelTime','studyTime','Failures','educationSupport',
                       'familySupport','paidClasses','Activities','Nursery','higherEdu','internetAtHome','Romantic',
                       'familyRelationship','freeTime','goOut','weekdayAlcoholUsage','weekendAlcoholUsage','Health',
                       'Absences','period1Score','period2Score','finalScore','subject']

            return (student,student_por,student_mat)
        except FileNotFoundError:
            print("File not found")

    def dataPro(self,student):
    
        # convert final_score to categorical variable # High:15~20 medium:10~14 low:0~9
        student['finalGrade'] = 'na'
        student.loc[(student.finalScore >= 15) & (student.finalScore <= 20), 'finalGrade'] = 'High' 
        student.loc[(student.finalScore >= 10) & (student.finalScore <= 14), 'finalGrade'] = 'Medium' 
        student.loc[(student.finalScore >= 0) & (student.finalScore <= 9), 'finalGrade'] = 'Low' 
        return student
        
    def ifmissing(self,std):
        no_missing = std.isnull().sum()
        total_missing=no_missing.sum()    
        print("Total number of missing=",total_missing)