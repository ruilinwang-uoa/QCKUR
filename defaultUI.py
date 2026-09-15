import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
import re
from datasciencecomponents import DataScienceRegressionComponents,DataScienceClassifierComponents,DataEngineering,FindBestModel
from NLGcomponents import RegressionTemplateBasedTextGeneration,ClassifierTemplateBasedTextGeneration,SettingForChatGPT,AutoFindBestModel,LoadQuestionBank
from LLMcomponents import SettingForLLM
# Load data science components
ds_regression_components = DataScienceRegressionComponents()
ds_data_engineering=DataEngineering()
ds_classifier_components= DataScienceClassifierComponents()
ds_find_best_model=FindBestModel()
# Load NLG components
set_for_GPT=SettingForChatGPT()
nlg_template_text_generate=RegressionTemplateBasedTextGeneration()
nlg_classifier_template_text_generate=ClassifierTemplateBasedTextGeneration()
auto_find_best_model=AutoFindBestModel()
set_question_bank=LoadQuestionBank()
# Load LLM components
set_for_LLM=SettingForLLM()


def extract_first_integer(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return 0

class MyApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry('800x600')
        self.window.title('My Application')
        self.csv_data = None
        self.independent_vars = None
        self.dependent_var = None
        self.choice = None
        self.criterion = None

    def start(self):
        self.step1()
        self.window.mainloop()

    def step1(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 1: Select a CSV file").pack()
        tk.Button(self.window, text="Select", command=self.load_csv).pack()
        tk.Button(self.window, text="Next", command=self.step2, anchor='e').pack(side='bottom')

    def load_csv(self):
        filename = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"),))
        if filename:
            self.csv_data = pd.read_csv(filename)

    def step2(self):
        if self.csv_data is None:
            messagebox.showerror("No File", "Please select a CSV file first.")
            return
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 2: Select independent and dependent variables").pack()

        frame = tk.Frame(self.window)
        frame.pack()

        tk.Label(frame, text="Please select the dependent variable first:").pack(side='left')
        self.dependent_var = ttk.Combobox(frame)
        self.dependent_var.pack(side='left')

        tk.Label(frame, text="Please select the independent variables:").pack(side='right')
        self.independent_vars = tk.Listbox(frame, selectmode='multiple')
        self.independent_vars.pack(side='right')

        headers = self.csv_data.columns.tolist()
        self.dependent_var['values'] = headers
        for header in headers:
            self.independent_vars.insert(tk.END, header)

        tk.Button(self.window, text="Back", command=self.step1, anchor='w').pack(side='bottom')
        tk.Button(self.window, text="Next", command=self.save_selections_and_go_to_step3, anchor='e').pack(side='bottom')

    def save_selections_and_go_to_step3(self):
        self.selected_independent_vars = self.get_selected_independent_vars()
        self.selected_dependent_var = self.dependent_var.get()
        self.step3()

    def save_criterion_and_go_to_step4(self):
        if self.criterion.get()=="":
            if self.choice.get()==3:
                self.selected_criterion="r2"
            elif self.choice.get()==4:
                self.selected_criterion="Accuracy"
            else:
                self.selected_criterion="unnecessary"
        else:
            self.selected_criterion = self.criterion.get()
        if self.choice.get() == 1:
            self.step4_1()
        elif self.choice.get() == 2:
            self.step4_2()
        elif self.choice.get() == 3:
            self.step4_3()
        elif self.choice.get() == 4:
            self.step4_4()

    def step3(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 3: Select a model type and criterion").pack()

        self.choice = tk.IntVar()

        tk.Radiobutton(self.window, text=f"Option 1: Use a regression model to explore the data.", variable=self.choice, value=1).pack()
        tk.Radiobutton(self.window, text=f"Option 2: Use a classifier model to explore the data.", variable=self.choice, value=2).pack()
        tk.Radiobutton(self.window, text=f"Option 3: Automatically fit a best regression model to explore the data.", variable=self.choice, value=3).pack()
        tk.Radiobutton(self.window, text=f"Option 4: Automatically fit a best classifier model to explore the data.", variable=self.choice, value=4).pack()

        tk.Label(self.window, text="Criterion:").pack()
        self.criterion = tk.Entry(self.window, text="Enter criterion for best model:")
        self.criterion.config(text=f"User input: {self.criterion.get()}")
        self.criterion.pack(pady=10)

        tk.Button(self.window, text="Back", command=self.step2, anchor='w').pack(side='bottom')

        tk.Button(self.window, text="Next", command=self.save_criterion_and_go_to_step4, anchor='e').pack(side='bottom')


    def step4_1(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 4: Choose a regression model.").grid()

        self.model_var = tk.IntVar()

        # Use Statsmodels Linear Regression
        self.statsmodels_button = tk.Radiobutton(self.window, text="Use Statsmodels Linear Regression",
                                                 variable=self.model_var, value=0)
        self.statsmodels_button.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10)

        # Use Scikit-learn Linear Regression
        self.sklearn_button = tk.Radiobutton(self.window, text="Use Scikit-learn Linear Regression",
                                             variable=self.model_var, value=1)
        self.sklearn_button.grid(row=3, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Ridge Regression option
        self.ridge_checkbutton = tk.Radiobutton(self.window, text="Use Ridge Regression", variable=self.model_var,
                                                value=2)
        self.ridge_checkbutton.grid(row=4, column=0, sticky=tk.E+tk.W, padx=10)

        self.ridge_alpha_label = tk.Label(self.window, text="Enter Alpha value (default 1.0):")
        self.ridge_alpha_label.grid(row=5, column=0, sticky=tk.E+tk.W, padx=10)
        self.ridge_alpha_entry = tk.Entry(self.window)
        self.ridge_alpha_entry.grid(row=6, column=0, sticky=tk.E+tk.W, padx=10, pady=5)


        # Use GB Regression
        self.GB_button = tk.Radiobutton(self.window, text="Use Gradient Boosting Regression",
                                             variable=self.model_var, value=5)
        self.GB_button.grid(row=7, column=0, sticky=tk.E+tk.W, padx=10)


        # Use RF Regression
        self.RF_button = tk.Radiobutton(self.window, text="Use Random Forest Regression",
                                             variable=self.model_var, value=6)
        self.RF_button.grid(row=8, column=0, sticky=tk.E+tk.W, padx=10)



        # Add Lasso Regression option
        self.lasso_checkbutton = tk.Radiobutton(self.window, text="Use Lasso Regression", variable=self.model_var,
                                                value=3)
        self.lasso_checkbutton.grid(row=2, column=1, sticky=tk.E+tk.W, padx=10)

        self.lasso_alpha_label = tk.Label(self.window, text="Enter Alpha value (default 1.0):")
        self.lasso_alpha_label.grid(row=3, column=1, sticky=tk.E+tk.W, padx=10)
        self.lasso_alpha_entry = tk.Entry(self.window)
        self.lasso_alpha_entry.grid(row=4, column=1, sticky=tk.E+tk.W, padx=10, pady=5)

        # Add Bayesian Ridge Regression option
        self.bayesian_ridge_checkbutton = tk.Radiobutton(self.window, text="Use Bayesian Ridge Regression",
                                                         variable=self.model_var, value=4)
        self.bayesian_ridge_checkbutton.grid(row=5, column=1, sticky=tk.E+tk.W, padx=10)

        self.bayesian_ridge_label = tk.Label(self.window, text="Enter two Alpha values, and Lambda values:")
        self.bayesian_ridge_label.grid(row=6, column=1, sticky=tk.E+tk.W, padx=10)
        self.alpha_1_entry = tk.Entry(self.window, width=8)
        self.alpha_1_entry.grid(row=7, column=1, sticky=tk.E+tk.W, padx=10)
        self.alpha_2_entry = tk.Entry(self.window, width=8)
        self.alpha_2_entry.grid(row=7, column=1, padx=(100, 0), sticky=tk.E+tk.W)
        self.lambda_1_entry = tk.Entry(self.window, width=8)
        self.lambda_1_entry.grid(row=7, column=1, padx=(190, 0), sticky=tk.E+tk.W)
        self.lambda_2_entry = tk.Entry(self.window, width=8)
        self.lambda_2_entry.grid(row=7, column=1, padx=(280, 0), sticky=tk.E+tk.W)

        tk.Button(self.window, text="Next", command=self.perform_regression, anchor='e').grid()


    def step4_2(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 4: Choose a classifier model.").grid()

        self.model_var = tk.IntVar()

        # Use Logistic Regression
        self.Logistic_button = tk.Radiobutton(self.window, text="Use Logistic Regression",
                                                 variable=self.model_var, value=0)
        self.Logistic_button.grid(row=2, column=0, sticky=tk.E+tk.W, padx=10)

        # Use Linear Discriminant Analysis
        self.LDA_button = tk.Radiobutton(self.window, text="Use Linear Discriminant Analysis",
                                             variable=self.model_var, value=1)
        self.LDA_button.grid(row=3, column=0, sticky=tk.E+tk.W, padx=10)

        # Add SVM - Linear Kernel option
        self.SVM_checkbutton = tk.Radiobutton(self.window, text="Use SVM - Linear Kernel", variable=self.model_var,
                                                value=2)
        self.SVM_checkbutton.grid(row=4, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Ridge Classifier option
        self.RC_checkbutton = tk.Radiobutton(self.window, text="Use Ridge Classifier", variable=self.model_var,
                                                value=3)
        self.RC_checkbutton.grid(row=5, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Random Forest Classifier option
        self.RF_checkbutton = tk.Radiobutton(self.window, text="Use Random Forest Classifier", variable=self.model_var,
                                                value=4)
        self.RF_checkbutton.grid(row=6, column=0, sticky=tk.E+tk.W, padx=10)

        # Add Ridge Classifier option
        self.DT_checkbutton = tk.Radiobutton(self.window, text="Use Decision Tree Classifier", variable=self.model_var,
                                                value=5)
        self.DT_checkbutton.grid(row=7, column=0, sticky=tk.E+tk.W, padx=10)

        tk.Button(self.window, text="Next", command=self.perform_classifier, anchor='e').grid()


    def step4_3(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 4: Here is the regression model selected for you.").pack()

        X = self.csv_data[self.selected_independent_vars]
        y = self.csv_data[self.selected_dependent_var]

        self.modelname, self.modeldetail, self.selected_criterion,comapre_results=ds_find_best_model.find_best_regression(X,y,self.selected_dependent_var,self.selected_criterion,self.selected_independent_vars)

        modelcomparestory=auto_find_best_model.model_compare(self.modelname, self.modeldetail, self.selected_criterion,1)
        print(comapre_results)
        print(modelcomparestory)
        self.modelchoose_label = tk.Text(self.window, width=100, height=40)
        self.modelchoose_label.delete(1.0, tk.END)
        self.modelchoose_label.insert(tk.END, modelcomparestory)
        self.modelchoose_label.pack(pady=10)
        tk.Button(self.window, text="Next", command=self.step5, anchor='e').pack()

    def step4_4(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 4: Here is the classifier model selected for you.").pack()

        X = self.csv_data[self.selected_independent_vars]
        y = self.csv_data[self.selected_dependent_var]

        self.modelname, self.modeldetail, self.selected_criterion,comapre_results=ds_find_best_model.find_best_classifier(X,y,self.selected_dependent_var,self.selected_criterion,self.selected_independent_vars)

        modelcomparestory=auto_find_best_model.model_compare(self.modelname, self.modeldetail, self.selected_criterion,1)
        print(modelcomparestory)
        self.modelchoose_label = tk.Text(self.window, width=100, height=40)
        self.modelchoose_label.delete(1.0, tk.END)
        self.modelchoose_label.insert(tk.END, modelcomparestory)
        self.modelchoose_label.pack(pady=10)
        tk.Button(self.window, text="Next", command=self.step5, anchor='e').pack()

    def perform_regression(self):
        threshold=0.8
        self.csv_data=ds_data_engineering.clean_data( self.csv_data, threshold, Xcol=self.selected_independent_vars, ycol=self.selected_dependent_var)
        X = self.csv_data[self.selected_independent_vars]
        y = self.csv_data[self.selected_dependent_var]

        if self.model_var.get() == 0:
            self.coefficients,self.p_values,self.r_squared =ds_regression_components.train_sm_linear_regression( X, y)
        elif self.model_var.get()==1:
            self.coefficients, self.p_values, self.r_squared = ds_regression_components.train_sk_linear_regression(X, y)
        elif self.model_var.get()==2:
            alpha = float(self.ridge_alpha_entry.get() or 1.0)
            self.coefficients, self.p_values, self.r_squared = ds_regression_components.train_ridge_regression(X, y,alpha)
        elif self.model_var.get()==3:
            alpha = float(self.lasso_alpha_entry.get() or 1.0)
            self.coefficients, self.p_values, self.r_squared = ds_regression_components.train_lasso_regression(X, y,alpha)
        elif self.model_var.get()==4:
            alpha_1 = float(self.alpha_1_entry.get() or 1e-06)
            alpha_2 = float(self.alpha_2_entry.get() or 1e-06)
            lambda_1 = float(self.lambda_1_entry.get() or 1e-06)
            lambda_2 = float(self.lambda_2_entry.get() or 1e-06)
            self.coefficients, self.p_values, self.r_squared = ds_regression_components.train_BayesianRidge_regression(X, y,alpha_1,alpha_2,lambda_1,lambda_2)
        elif self.model_var.get()==5:
            self.coefficients, self.p_values, self.r_squared,self.r2_train,self.r2_test = ds_regression_components.train_gradient_boosting_regression(X, y)
        elif self.model_var.get()==6:
            self.coefficients, self.p_values, self.r_squared,self.r2_train,self.r2_test = ds_regression_components.train_random_forest_regression(X, y)

        self.coef_pval_df = ds_regression_components.coefficients_with_Pvalues(self.coefficients,self.p_values,self.selected_independent_vars)
        print(self.coef_pval_df)

        self.step5()

    def perform_classifier(self):
        X = self.csv_data[self.selected_independent_vars]
        y = self.csv_data[self.selected_dependent_var]

        if self.model_var.get() == 0:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_logistic_regression(X, y,self.selected_independent_vars,self.selected_dependent_var)

        elif self.model_var.get()==1:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_linear_discriminant_analysis(X, y,self.selected_independent_vars,self.selected_dependent_var)

        elif self.model_var.get()==2:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_SVC_classifier(X, y,self.selected_independent_vars,self.selected_dependent_var)

        elif self.model_var.get()==3:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_ridge_classifier(X, y,self.selected_independent_vars,self.selected_dependent_var)

        elif self.model_var.get()==4:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_random_forest_classifier(X, y,self.selected_independent_vars,self.selected_dependent_var)

        elif self.model_var.get()==5:
            self.model, self.coeff_df, self.accuracy, self.coefficients, self.train_accuracy, self.test_accuracy=ds_classifier_components.train_decision_tree_classifier(X, y,self.selected_independent_vars,self.selected_dependent_var)

        print("The coefficients and accuracy of the model are as follows:")
        print(self.coeff_df)
        print(self.test_accuracy)
        print("---------------------")
        self.step5()


    def step5(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 5: Question Asking").pack()

        self.user_text_label = tk.Label(self.window, text="Enter a question about the dataset or the model:")
        self.user_text_label.pack(pady=10)

        self.user_text = tk.StringVar()
        self.user_text_entry = tk.Entry(self.window, textvariable=self.user_text, width=50)
        self.user_text_entry.pack(pady=10)

        self.api_key_label = tk.Label(self.window, text="Enter your API key:")
        self.api_key_label.pack(pady=10)

        self.api_key = tk.StringVar()
        self.api_key_entry = tk.Entry(self.window, textvariable=self.api_key, show='*', width=50)
        self.api_key_entry.pack(pady=10)

        self.gpt_model_label = tk.Label(self.window, text="Enter GPT model name (default: gpt-3.5-turbo):")
        self.gpt_model_label.pack(pady=10)

        self.gpt_model_name = tk.StringVar()
        self.gpt_model_entry = tk.Entry(self.window, textvariable=self.gpt_model_name)
        self.gpt_model_entry.pack(pady=10)

        self.background_text_label = tk.Label(self.window, text="Enter background about the dataset:")
        self.background_text_label.pack(pady=10)

        self.background_text = tk.StringVar()
        self.background_text_entry = tk.Entry(self.window, textvariable=self.background_text, width=50)
        self.background_text_entry.pack(pady=10)

        self.target_text_label = tk.Label(self.window, text="Enter target text:")
        self.target_text_label.pack(pady=10)

        self.target_text = tk.StringVar()
        self.target_text_entry = tk.Entry(self.window, textvariable=self.target_text, width=50)
        self.target_text_entry.pack(pady=10)

        tk.Button(self.window, text="Next", command=self.set_background, anchor='e').pack()

    def step6(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 6: Question Answering").pack()

        self.answer_label = tk.Text(self.window, width=100, height=40)
        self.answer_label.delete(1.0, tk.END)
        self.answer_label.insert(tk.END, self.GPTanswer)
        self.answer_label.pack(pady=10)

        self.ask_another_button = tk.Button(self.window, text="Ask another question", command=self.step8)
        self.ask_another_button.pack( pady=10)

    def step8(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 7: Asking another question").pack()

        self.ask_another_question_label = tk.Label(self.window, text="Enter another question for Chat-GPT:")
        self.ask_another_question_label.pack(pady=10)

        self.another_question_text = tk.StringVar()
        self.another_question_entry = tk.Entry(self.window, textvariable=self.another_question_text, width=50)
        self.another_question_entry.pack(pady=10)

        self.next_button = tk.Button(self.window, text="Next", command=self.go_to_step9)
        self.next_button.pack(side=tk.LEFT, pady=10)


    def step9(self):
        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 8: Showing the asnwer").pack()

        self.new_answer_label = tk.Text(self.window, width=100, height=40)
        self.new_answer_label.delete(1.0, tk.END)
        self.new_answer_label.insert(tk.END, self.new_answer)
        self.new_answer_label.pack(pady=10)

        self.ask_another_button = tk.Button(self.window, text="Ask another question", command=self.step8)
        self.ask_another_button.pack(side=tk.LEFT, pady=10)

        self.make_summary=tk.Button(self.window, text="Summary and End", command=self.step10)
        self.make_summary.pack(side=tk.LEFT, pady=10)

    def step10(self):

        for widget in self.window.winfo_children():
            widget.destroy()
        tk.Label(self.window, text="Step 9: Summary").pack()

        question = "Now, to summarize the above information. Write a short text report using the answers to user questions about the dataset and model. Note that this report only focuses on questions raised by users about the dataset and model. And no need to give an explanation of how this conclusion was reached."

        user_target_text = self.target_text.get()
        if user_target_text:
            question=question+" When you summarize, please keep your writing style and format consistent with what follows. \n"+user_target_text

        payload, messages = set_for_LLM.set_payload(question, self.gpt_model_name, self.messages)
        output, messages = set_for_LLM.send_response_receive_output(self.url, self.headers, payload, messages)

        self.messages = messages
        self.summary=output

        self.summary_label = tk.Text(self.window, width=100, height=40)
        self.summary_label.delete(1.0, tk.END)
        self.summary_label.insert(tk.END, self.summary)
        self.summary_label.pack(pady=10)

        set_for_LLM.save_chat_history_to_docx(self.messages)


        self.again_button = tk.Button(self.window, text="Do it again", command=self.step1)
        self.again_button.pack(side=tk.LEFT, pady=10)

    def go_to_step9(self):

        self.user_question=self.another_question_text.get()
        self.question_number_select()
        if self.choice.get()==1 or self.choice.get()==3:
            self.regression_answer()
            print(self.messages)

        elif self.choice.get()==2 or self.choice.get()==4:
            self.classifier_answer()
            print(self.messages)

        self.new_answer = self.output
        print(self.messages)

        self.step9()

    def set_background(self):
        user_api_key = self.api_key.get()
        if not user_api_key:
            self.answerbyGPT.config(text="Invalid API key. Please enter a valid API key.")
            return
        key=user_api_key

        os.environ['OPENAI_API_KEY'] = key
        gpt_model_name_input = self.gpt_model_name.get()
        gpt_model_name = gpt_model_name_input if gpt_model_name_input else "gpt-3.5-turbo"
        self.gpt_model_name=gpt_model_name

        if self.choice.get() == 1:
            if self.model_var.get() == 0:
                self.modelname="Statsmodels Linear Regression"
            elif self.model_var.get() == 1:
                self.modelname="Scikit-learn Linear Regression"
            elif self.model_var.get() == 2:
                self.modelname="Ridge Regression"
            elif self.model_var.get() == 3:
                self.modelname="Lasso Regression"
            elif self.model_var.get() == 4:
                self.modelname="Bayesian Ridge Regression"
            elif self.model_var.get() == 5:
                self.modelname = "Gradient Boosting Regressor"
            elif self.model_var.get() == 6:
                self.modelname = "Random Forest Regressor"
            modelinformation=f"The R-squared of the model is: {self.r_squared:.4f}\n"+str(self.coef_pval_df)

        elif self.choice.get() == 2:
            if self.model_var.get() == 0:
                self.modelname = "Logistic Regression"
            elif self.model_var.get() == 1:
                self.modelname = "Linear Discriminant Analysis"
            elif self.model_var.get() == 2:
                self.modelname = "SVM-Linear Kernel"
            elif self.model_var.get() == 3:
                self.modelname = "Ridge Classifier"
            elif self.model_var.get() == 4:
                self.modelname = "Random Forest Classifier"
            elif self.model_var.get() == 5:
                self.modelname = "Decision Tree Classifier"
            modelinformation = f"The accuracy of the model is: {self.accuracy:.4f}\n" + str(self.coeff_df)

        elif self.choice.get() == 3:
            modelinformation = f"The R-squared of the model is: {self.r_squared:.4f}\n" + str(self.coef_pval_df)

        elif self.choice.get() == 4:
            modelinformation = f"The accuracy of the model is: {self.accuracy:.4f}\n" + str(self.coeff_df)

        if self.background_text !="":
            modelinformation=modelinformation+'\n'+self.background_text.get()

        print(modelinformation)

        background = set_for_GPT.set_background(self.selected_independent_vars, self.selected_dependent_var, self.modelname, modelinformation)
        self.url, self.background, self.chatmodel, self.headers, self.messages=set_for_LLM.set_chatGPT(background,key)

        if self.choice.get() == 1 or self.choice.get() ==3:
            self.go_to_step6_1()
        elif self.choice.get() == 2 or self.choice.get() ==4:
            self.go_to_step6_2()

    def go_to_step6_1(self):
        self.user_question=self.user_text.get()
        self.question_number_select()
        self.regression_answer()
        print(self.messages)
        self.GPTanswer=self.output
        self.step6()

    def regression_answer(self):
        section_num = self.section_num
        questions=[]
        answers=[]
        if section_num==1:
            questions,answers=nlg_template_text_generate.Q_and_A_about_R2(self.coef_pval_df,self.selected_dependent_var,self.modelname,self.r_squared)
        elif section_num==2:
            questions,answers=nlg_template_text_generate.Q_and_A_about_coefficients(self.coef_pval_df,self.selected_dependent_var)
        elif section_num==3:
            questions, answers = nlg_template_text_generate.Q_and_A_about_importance(self.selected_independent_vars,self.selected_dependent_var,self.coefficients,self.p_values)
        elif section_num==4:
            questions, answers = nlg_template_text_generate.Q_and_A_about_pvalues(self.coef_pval_df,self.selected_dependent_var)
        elif section_num==5:
            questions, answers = nlg_template_text_generate.Q_and_A_about_ML_importance(self.selected_independent_vars,self.selected_dependent_var,self.coef_pval_df)
        elif section_num == 6:
            questions, answers = nlg_template_text_generate.Q_and_A_about_ML_overfit(self.r2_train,self.selected_dependent_var,self.r2_test)
        elif section_num==0:
            questions="There are no matching default questions in the template."
            answers="Please answer the question based on the analysis results."

        print(questions)
        print(answers)

        default_answer=set_for_GPT.answer_update(self.user_question,questions,answers)

        print(default_answer)
        payload, messages = set_for_LLM.set_payload(default_answer, self.gpt_model_name, self.messages)
        output, messages = set_for_LLM.send_response_receive_output(self.url, self.headers, payload, messages)
        self.messages = messages
        self.output=output


    def go_to_step6_2(self):

        self.user_question=self.user_text.get()
        self.question_number_select()

        self.classifier_answer()

        self.GPTanswer=self.output
        self.step6()

    def classifier_answer(self):
        section_num=self.section_num

        questions=[]
        answers=[]
        text=''
        if section_num == 1:
            questions,answers=nlg_classifier_template_text_generate.Q_and_A_about_accuracy(self.accuracy,self.modelname)
        elif section_num == 2:
            questions,answers=nlg_classifier_template_text_generate.Q_and_A_about_coefficients(self.coeff_df,self.selected_dependent_var)
        elif section_num == 3:
            questions,answers=nlg_classifier_template_text_generate.Q_and_A_about_importance(self.coefficients,self.selected_dependent_var,self.coeff_df,self.model.classes_)
        elif section_num == 4:
            questions,answers=nlg_classifier_template_text_generate.Q_and_A_about_ML_overfit(self.train_accuracy,self.test_accuracy)
        elif section_num == 5:
            questions,answers=nlg_classifier_template_text_generate.Q_and_A_about_ML_importance(self.coeff_df,self.selected_dependent_var)

        elif section_num == 0:
            questions="There are no matching default questions in the template."
            answers="Please answer the question based on the analysis results."

        default_answer = set_for_GPT.answer_update(self.user_question, questions, answers)
        print(default_answer)


        payload, messages = set_for_LLM.set_payload(default_answer, self.gpt_model_name, self.messages)
        output, messages = set_for_LLM.send_response_receive_output(self.url, self.headers, payload, messages)

        self.messages = messages
        self.output=output


    def question_number_select(self):
        if self.choice.get() == 1:
            if self.model_var.get()<5:
                content=set_question_bank.load_regression_questions()
            else:
                content=set_question_bank.load_ML_regression_questions()

        elif self.choice.get() == 3:
            content=set_question_bank.load_regression_questions()

        elif self.choice.get() == 2:
            if self.model_var.get()<4:
                content=set_question_bank.load_classifier_questions()

            else:
                content=set_question_bank.load_ML_classifier_questions()
        elif self.choice.get() == 4:
            content = set_question_bank.load_classifier_questions()

        query = "My question is: " + self.user_question + "\nPlease refer to the following question bank and choose the Section number (for example, if you choose Section 5, please return 5.) that matches the meaning of my question. Please note that as long as the meaning matches, there is no need for word-for-word correspondence. My entry may have spelling or grammatical mistakes, please ignore those mistakes. Returns 0 if no section matches. Only answer an integer as you choose, do not reply with any information other than the integer, do not reply why you chose the section number. Following is the question bank: \n"+content

        print(query)

        payload, messages = set_for_LLM.set_payload(query, self.gpt_model_name, self.messages)
        output, messages = set_for_LLM.send_response_receive_output(self.url, self.headers, payload, messages)

        self.section_num = extract_first_integer(output)
        print(output)
        print(self.section_num)


    def get_selected_independent_vars(self):
        return [self.independent_vars.get(i) for i in self.independent_vars.curselection()]


if __name__ == '__main__':
    app = MyApp()
    app.start()


