import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.impute import SimpleImputer
import os
from docx import Document
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier,LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve,r2_score
import numpy as np
import re
import openai
import json
import requests
from jinja2 import Environment, FileSystemLoader
from pycaret import classification,regression

# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('./apptemplates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

# # Loading the Jinja templates from the folder
# For the regression
linearSummary = env.get_template('linearSummary.txt')
linearSummary2 = env.get_template('linearSummary2.txt')
linearSummary3 = env.get_template('linearSummary3.txt')
linearSummary4 = env.get_template('linearSummary4.txt')
linearQuestion = env.get_template('linearQuestionset.txt')

# For Machine Learning Model
MachineLearningLinearModelSummary1=env.get_template('MachineLearningLinearModelSummary1.txt')
MachineLearningLinearModelSummary2=env.get_template('MachineLearningLinearModelSummary2.txt')
MLlinearQuestionSet=env.get_template('MLlinearQuestionSet.txt')
MachineLearningclassifierSummary1=env.get_template('MachineLearningClassifierModelSummary1.txt')
MachineLearningclassifierSummary2=env.get_template('MachineLearningClassifierModelSummary2.txt')
MLclassifierQuestionSet=env.get_template('MLclassifierQuestionSet.txt')

# For the classifier
classifiersummary1=env.get_template('classifiersummary1.txt')
classifiersummary2=env.get_template('classifiersummary2.txt')
classifiersummary3=env.get_template('classifiersummary3.txt')
classifierquestion=env.get_template('classifierQuestionset.txt')

# For ChatGPT
databackground = env.get_template('databackground.txt')
answerup=env.get_template('answer_upgrade.txt')

# For pycaret
automodelcompare1 = env.get_template('AMC1.txt')
automodelcompare2 = env.get_template('AMC2.txt')


class SettingForChatGPT():
    def __init__(self):
        pass

    def answer_update(self,user_question,questions,answers):
        default_answer = answerup.render(userquestion=user_question, questions=questions, answers=answers)
        return default_answer

    def set_background(self,Xcol,ycol,modelname,modelinformation):
        background = databackground.render(Xcol=Xcol, ycol=ycol, modelname=modelname, modelinformation=modelinformation)
        return background

class LoadQuestionBank():
    def __init__(self):
        pass

    def load_regression_questions(self):
        with open('./apptemplates/QuestionBank/linearQuestionBank.txt', 'r') as file:
            content = file.read()
        return content

    def load_ML_regression_questions(self):
        with open('./apptemplates/QuestionBank/MLlinearQuestionBank.txt', 'r') as file:
            content = file.read()
        return content

    def load_classifier_questions(self):
        with open('./apptemplates/QuestionBank/classifierQuestionBank.txt', 'r') as file:
            content = file.read()
        return content

    def load_ML_classifier_questions(self):
        with open('./apptemplates/QuestionBank/MLclassifierQuestionBank.txt', 'r') as file:
            content = file.read()
        return content


class AutoFindBestModel():
    def __init__(self):
        pass

    def model_compare(self, modelname, modeldetail, selected_criterion,n=1):
        modelcomparestory = automodelcompare1.render(best=modelname, detail=modeldetail, n_select=n, sort=selected_criterion)
        return modelcomparestory

class RegressionTemplateBasedTextGeneration():
    def __init__(self):
        pass

    def Q_and_A_about_R2(self,coef_pval_df,selected_dependent_var,modelname,r_squared):
        questions = linearQuestion.render(section=1, indeNum=len(coef_pval_df['Xcol']),
                                          xcol=coef_pval_df['Xcol'], ycol=selected_dependent_var)

        answers = linearSummary2.render(modelname=modelname, r2=r_squared)
        return (questions,answers)
    def Q_and_A_about_coefficients(self,coef_pval_df,selected_dependent_var):
        questions = []
        answers = []
        for index, row in coef_pval_df.iterrows():
            question = linearQuestion.render(section=2, xcol=row['Xcol'], ycol=selected_dependent_var)
            answer = linearSummary.render(coeff=row['Coefficients'], p=row['P-values'], xcol=row['Xcol'],
                                          ycol=selected_dependent_var)
            questions.append(question)
            answers.append(answer)
        return (questions,answers)

    def Q_and_A_about_importance(self,selected_independent_vars,selected_dependent_var,coefficients,p_values):
        # Update the most_important_x
        questions = linearQuestion.render(section=3, ycol=selected_dependent_var)
        most_important_x = selected_independent_vars[np.argmax(np.abs(coefficients[1:]))]
        # Display X variables with positive and negative slopes
        positive_slopes = [x for i, x in enumerate(selected_independent_vars) if coefficients[i + 1] > 0]
        negative_slopes = [x for i, x in enumerate(selected_independent_vars) if coefficients[i + 1] < 0]
        # Display X variables with P-values greater than and less than 0.05
        high_p_values = [x for i, x in enumerate(selected_independent_vars) if p_values[i + 1] > 0.05]
        low_p_values = [x for i, x in enumerate(selected_independent_vars) if p_values[i + 1] <= 0.05]
        answers = linearSummary3.render(ss=low_p_values, lenss=len(low_p_values), nss=high_p_values,
                                        lennss=len(high_p_values), pf=positive_slopes, lenpf=len(positive_slopes),
                                        nf=negative_slopes, lennf=len(negative_slopes),
                                        ycol=selected_dependent_var,
                                        imp=most_important_x)
        return (questions,answers)

    def Q_and_A_about_pvalues(self,coef_pval_df,selected_dependent_var):
        questions = []
        answers = []
        for index, row in coef_pval_df.iterrows():
            question = linearQuestion.render(section=4, xcol=row['Xcol'], ycol=selected_dependent_var)
            answer = linearSummary4.render(coeff=row['Coefficients'], p=row['P-values'], xcol=row['Xcol'],
                                           ycol=selected_dependent_var)
            questions.append(question)
            answers.append(answer)
        return (questions,answers)

    def Q_and_A_about_ML_importance(self,selected_independent_vars,selected_dependent_var,coefficients):
        questions = MLlinearQuestionSet.render(section=5, ycol=selected_dependent_var)
        # most_important_x = selected_independent_vars[np.argmax(np.abs(coefficients[1:]))]
        most_important_x = coefficients.loc[coefficients['Coefficients'].abs().idxmax(), 'Xcol']
        answers = MachineLearningLinearModelSummary1.render(ycol=selected_dependent_var, Xcol=most_important_x)
        return (questions,answers)

    def Q_and_A_about_ML_overfit(self,r2_train,selected_dependent_var,r2_test):
        questions = MLlinearQuestionSet.render(section=6, ycol=selected_dependent_var)
        answers = MachineLearningLinearModelSummary2.render(trainR2=r2_train, testR2=r2_test)
        return (questions,answers)


class ClassifierTemplateBasedTextGeneration():
    def __init__(self):
        pass
    def Q_and_A_about_accuracy(self,accuracy,modelname):
        questions = classifierquestion.render(section=1, modelname=modelname)
        answers = classifiersummary1.render(r2=accuracy, modelName=modelname)
        return (questions,answers)

    def Q_and_A_about_coefficients(self,coeff_df,selected_dependent_var):
        questions = []
        answers = []
        text = ''
        for target_names, row in coeff_df.iterrows():
            question = classifierquestion.render(section=2, classes=target_names, ycol=selected_dependent_var)
            positivecol = []
            negativecol = []
            notchangecol = []
            i = 0
            for selected_independent_vars, coeff in row.items():
                if i != 0:
                    if float(coeff) > 0:
                        positivecol.append(selected_independent_vars)
                    elif float(coeff) < 0:
                        negativecol.append(selected_independent_vars)
                    else:
                        notchangecol.append(selected_independent_vars)
                i = i + 1

            answer = classifiersummary2.render(classes=target_names, pc=positivecol, nc=negativecol,
                                               ncc=notchangecol)
            questions.append(question)
            text = text + answer + '\n'
        answers = text
        return (questions,answers)

    def Q_and_A_about_importance(self,coefficients,selected_dependent_var,coeff_df,target_class):
        feature_importance = np.mean(np.abs(coefficients), axis=0)
        questions = classifierquestion.render(section=3, ycol=selected_dependent_var)
        most_important_x = coeff_df.columns[np.argmax(feature_importance)+1]
        # most_important_x = [col for col in coeff_df.columns if (np.abs(coeff_df[col]) == coeff_df.abs().max().max()).any()]
        print(most_important_x)
        answer = classifiersummary3.render(imp=most_important_x)
        text=''
        if len(target_class) > 2:
            for target_name, row in coeff_df.iterrows():
                most_important_feature = row.abs().idxmax()
                text = text + f"\nFor the {target_name},the most important feature is {most_important_feature}."
            answers = text + '\n' + answer
        else:
            answers = answer
        return (questions,answers)

    def Q_and_A_about_ML_overfit(self,train_accuracy,test_accuracy):
        questions = MLclassifierQuestionSet.render(section=4)
        answers = MachineLearningclassifierSummary1.render(trainR2=train_accuracy, testR2=test_accuracy)
        return (questions,answers)

    def Q_and_A_about_ML_importance(self,coeff_df,selected_dependent_var):
        questions = MLclassifierQuestionSet.render(section=5)
        print(coeff_df)
        max_value = coeff_df.abs().max().max()

        # arr_abs = np.abs(coeff_df)
        #
        # max_value = arr_abs.max()

        most_important_x = coeff_df.columns[coeff_df.abs().iloc[0] == max_value][0]
        answers = MachineLearningclassifierSummary2.render(Xcol=most_important_x, ycol=selected_dependent_var,
                                                          )
        return (questions,answers)

