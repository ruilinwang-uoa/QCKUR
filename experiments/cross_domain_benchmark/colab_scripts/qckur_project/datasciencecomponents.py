import pandas as pd
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier,LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve,r2_score
import numpy as np
from pycaret import classification,regression

class DataEngineering:
    def clean_data(self, data, threshold, Xcol=[], ycol=''):
        """clean nan"""
        if Xcol != [] and ycol != '':
            data = data[Xcol + [ycol]]
        data = data.replace('?', np.nan)
        data = data.loc[:, data.isnull().mean() < threshold]
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        for i in data.columns:
            imputer = imputer.fit(data[[i]])
            data[[i]] = imputer.transform(data[[i]])
        return data

class FindBestModel:
    def __init__(self):
        pass

    def find_best_regression(self,X,y,selected_dependent_var,selected_criterion,selected_independent_vars):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dataset = pd.concat([X_train, y_train], axis=1)
        reg = regression.setup(data=dataset, target=selected_dependent_var)
        exclude=['rf','gbr','catboost','lightgbm','et','ada','xgboost','llar','lar','huber','dt','omp','par','en','knn','dummy']
        best_model = regression.compare_models(exclude=exclude, n_select=1, sort=selected_criterion)
        comapre_results = regression.pull()
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        coefficients = np.append(best_model.intercept_, best_model.coef_)
        r_squared = r2_score(y_test, best_model.predict(X_test))
        data_dict = {'Coefficients': coefficients[1:]}
        data_dict['P-values'] =p_values[1:]
        coef_pval_df = pd.DataFrame(data_dict, index=selected_independent_vars)
        coef_pval_df.index.name = "Xcol"
        coef_pval_df = coef_pval_df.reset_index()
        modeldetail=str(best_model)
        modelname=FindBestModel.more_readable_model_name(self,modeldetail)
        return (modelname,modeldetail,selected_criterion,comapre_results)

    def find_best_classifier(self, X, y, selected_dependent_var, selected_criterion, selected_independent_vars):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        dataset = pd.concat([X_train, y_train], axis=1)
        clf = classification.setup(data=dataset, target=selected_dependent_var)
        exclude = ['rf','dt','qda','knn','lightgbm','et','catboost','xgboost','gbc','ada','dummy']
        best_model = classification.compare_models(exclude=exclude, n_select=1, sort=selected_criterion)
        comapre_results = classification.pull()
        print(comapre_results)
        coefficients=best_model.coef_
        target_names = best_model.classes_
        if len(target_names) == 2 and best_model.coef_.shape[0] == 1:
            # Special handling for binary classification case
            coeff_df = pd.DataFrame(best_model.coef_, columns=selected_independent_vars, index=[target_names[0]])
        else:
            # The usual case for multiclass classification
            coeff_df = pd.DataFrame(best_model.coef_, columns=selected_independent_vars, index=target_names)
        # Make predictions on the test set
        print(coeff_df)
        y_pred = best_model.predict(X_test)
        # Calculate and output the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        modeldetail = str(best_model)
        modelname=FindBestModel.more_readable_model_name(self,modeldetail)
        return (modelname,modeldetail,selected_criterion,comapre_results)


    def more_readable_model_name(self,modeldetail):
        modeldetail=modeldetail
        if "Ridge" in modeldetail and "BayesianRidge" not in modeldetail:
            translatedmodel = "Ridge Model"
        elif "LinearDiscriminant" in modeldetail:
            translatedmodel = "Linear Discriminant Analysis"
        elif "GradientBoosting" in modeldetail:
            translatedmodel = "Gradient Boosting Model"
        elif "AdaBoost" in modeldetail:
            translatedmodel = "Ada Boost"
        elif "LGBMClassifier" in modeldetail:
            translatedmodel = "Light Gradient Boosting Machine Classifier"
        elif "DummyClassifier" in modeldetail:
            translatedmodel = "Dummy Classifier"
        elif "KNeighborsClassifier" in modeldetail:
            translatedmodel = "K Neighbors Classifier"
        elif "SGDClassifier" in modeldetail:
            translatedmodel = "SGD Classifier"
        elif "LGBMRegressor" in modeldetail:
            translatedmodel = "Light Gradient Boosting Machine"
        elif "RandomForest" in modeldetail:
            translatedmodel = "Random Forest Model"
        elif "XGBRegressor" in modeldetail:
            translatedmodel = "Extreme Gradient Boosting"
        elif "XGBClassifier" in modeldetail:
            translatedmodel = "Extreme Gradient Boosting Classifier"
        elif "Logistic" in modeldetail:
            translatedmodel = "Logistic Model"
        elif "QuadraticDiscriminant" in modeldetail:
            translatedmodel = "Quadratic Discriminant Analysis"
        elif "GaussianNB" in modeldetail:
            translatedmodel = "Naive Bayes"
        elif "ExtraTrees" in modeldetail:
            translatedmodel = "Extra Trees model"
        elif "DecisionTree" in modeldetail:
            translatedmodel = "Decision Tree Model"
        elif "Lasso" in modeldetail and "LassoLars" not in modeldetail:
            translatedmodel = "Lasso Regression	"
        elif "LassoLars" in modeldetail:
            translatedmodel = "Lasso Least Angle Regression	"
        elif "BayesianRidge" in modeldetail:
            translatedmodel = "Bayesian Ridge"
        elif "LinearRegression" in modeldetail:
            translatedmodel = "Linear Regression"
        elif "HuberRegressor" in modeldetail:
            translatedmodel = "Huber Regressor"
        elif "PassiveAggressiveRegressor" in modeldetail:
            translatedmodel = "Passive Aggressive Regressor"
        elif "OrthogonalMatchingPursuit" in modeldetail:
            translatedmodel = "Orthogonal Matching Pursuit"
        elif "AdaBoostRegressor" in modeldetail:
            translatedmodel = "AdaBoost Regressor"
        elif "KNeighborsRegressor" in modeldetail:
            translatedmodel = "K Neighbors Regressor"
        elif "ElasticNet" in modeldetail:
            translatedmodel = "Elastic Net"
        elif "DummyRegressor" in modeldetail:
            translatedmodel = "Dummy Regressor"
        elif "Lars" in modeldetail:
            translatedmodel = "Least Angle Regression"
        modelname=translatedmodel
        return modelname


class DataScienceRegressionComponents:
    def __init__(self):
        pass

    def train_sm_linear_regression(self, X, y):
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return (model.params, model.pvalues, model.rsquared)
    def train_sk_linear_regression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        coefficients = np.append(model.intercept_, model.coef_)
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        r_squared = model.score(X, y)
        return (coefficients,p_values,r_squared)

    def train_ridge_regression(self, X, y,alpha):
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        coefficients = np.append(model.intercept_, model.coef_)
        r_squared = r2_score(y_test, model.predict(X_test))
        return (coefficients,p_values,r_squared)
    def train_lasso_regression(self, X, y,alpha):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        model = Lasso(alpha=alpha).fit(X_train, y_train)
        coefficients = np.append(model.intercept_, model.coef_)
        r_squared = r2_score(y_test, model.predict(X_test))
        return (coefficients,p_values,r_squared)

    def train_BayesianRidge_regression(self, X, y,alpha_1,alpha_2,lambda_1,lambda_2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2).fit(X_train,
                                                                                                          y_train)
        # Get the coefficients for each predictor
        coefficients = np.append(model.intercept_, model.coef_)
        # Calculate R-squared
        y_pred = model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        return (coefficients,p_values,r_squared)

    def train_gradient_boosting_regression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create Gradient Boosting Regressor
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        # Get the coefficients for each predictor
        coefficients = np.append(0, model.feature_importances_)
        # Calculate R-squared
        y_pred = model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)
        return (coefficients,p_values,r_squared,r2_train,r2_test)

    def train_random_forest_regression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create Random Forest Regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        p_values = sm.OLS(y, sm.add_constant(X)).fit().pvalues
        # Get the coefficients for each predictor
        coefficients = np.append(0, model.feature_importances_)
        # Calculate R-squared
        y_pred = model.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)
        return (coefficients,p_values,r_squared,r2_train,r2_test)

    def coefficients_with_Pvalues(self,coefficients,p_values,selected_independent_vars):
        # Create a dataframe with coefficients and p-values (if available) for independent variables only
        data_dict = {'Coefficients': coefficients[1:]}
        data_dict['P-values'] = p_values[1:]
        coef_pval_df = pd.DataFrame(data_dict, index=selected_independent_vars)
        coef_pval_df.index.name = "Xcol"
        coef_pval_df = coef_pval_df.reset_index()
        return (coef_pval_df)

class DataScienceClassifierComponents:
    def __init__(self):
        pass

    def set_coefficients(self,model,selected_independent_vars,selected_dependent_var):
        coefficients = model.coef_
        target_names = model.classes_
        print(model.coef_)
        print(model.classes_)
        if len(target_names) == 2 and model.coef_.shape[0] == 1:
            # Special handling for binary classification case
            coeff_df = pd.DataFrame(model.coef_, columns=selected_independent_vars, index=[target_names[0]])
        else:
            # The usual case for multiclass classification
            coeff_df = pd.DataFrame(model.coef_, columns=selected_independent_vars, index=target_names)
        # Add pseudo column for index values
        coeff_df.insert(0, selected_dependent_var, coeff_df.index)
        # Reset index
        coeff_df.reset_index(drop=True, inplace=True)
        return (coeff_df,coefficients)

    def set_feature_importances(self,model,selected_independent_vars):

        coefficients = model.feature_importances_
        coeff_df = pd.DataFrame([model.feature_importances_], columns=selected_independent_vars)
        return (coefficients,coeff_df)

    def set_accuracy(self,model,X_train,y_train,X_test,y_test):
        y_pred = model.predict(X_test)
        # Calculate and output the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        return (accuracy,train_accuracy,test_accuracy)
    def train_logistic_regression(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        coeff_df,coefficients=DataScienceClassifierComponents.set_coefficients(self,model,selected_independent_vars,selected_dependent_var)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)

    def train_linear_discriminant_analysis(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        coeff_df,coefficients=DataScienceClassifierComponents.set_coefficients(self,model,selected_independent_vars,selected_dependent_var)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)

    def train_SVC_classifier(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        coeff_df,coefficients=DataScienceClassifierComponents.set_coefficients(self,model,selected_independent_vars,selected_dependent_var)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)

    def train_ridge_classifier(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RidgeClassifier()
        model.fit(X_train, y_train)
        coeff_df,coefficients=DataScienceClassifierComponents.set_coefficients(self,model,selected_independent_vars,selected_dependent_var)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)

    def train_random_forest_classifier(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        coefficients,coeff_df=DataScienceClassifierComponents.set_feature_importances(self,model,selected_independent_vars)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)

    def train_decision_tree_classifier(self, X, y,selected_independent_vars,selected_dependent_var):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        coefficients,coeff_df=DataScienceClassifierComponents.set_feature_importances(self,model,selected_independent_vars)
        accuracy, train_accuracy, test_accuracy=DataScienceClassifierComponents.set_accuracy(self,model,X_train,y_train,X_test,y_test)
        return (model,coeff_df,accuracy,coefficients,train_accuracy,test_accuracy)
