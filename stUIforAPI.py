import streamlit as st
import pandas as pd
import re
import os
from LocalLLMcomponents import SettingForOllama
from datasciencecomponents import DataScienceRegressionComponents,DataScienceClassifierComponents,DataEngineering,FindBestModel
from NLGcomponents import RegressionTemplateBasedTextGeneration,ClassifierTemplateBasedTextGeneration,SettingForChatGPT,AutoFindBestModel,LoadQuestionBank
from LLMcomponents import SettingForLLM
import glob
import json

# 从字符串中提取第一个整数，用于解析匹配结果
def extract_first_integer(string):
    match = re.search(r"\d+", string)
    if match:
        return int(match.group())
    return 0

# 初始化 LLM 与组件
set_for_localLLM = SettingForOllama()
set_for_GPT = SettingForChatGPT()
ds_data_engineering = DataEngineering()
ds_regression = DataScienceRegressionComponents()
ds_classifier = DataScienceClassifierComponents()
nlg_reg = RegressionTemplateBasedTextGeneration()
nlg_cls = ClassifierTemplateBasedTextGeneration()
loader = LoadQuestionBank()
set_for_LLM=SettingForLLM()



def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_step = 1
        st.session_state.df = None
        st.session_state.dependent_var = None
        st.session_state.independent_vars = []
        st.session_state.background_text = ''
        st.session_state.llm_bg = []
        st.session_state.llm_messages = []
        st.session_state.user_questions = []
        st.session_state.recommended_models = []
        st.session_state.reg_choice = 'None'
        st.session_state.cls_choice = 'None'
        st.session_state.selected_model = None
        st.session_state.fit_results = {}
        st.session_state.single_question = ''
        st.session_state.single_answer = ''
        st.session_state.llm_mode = 'ollama'
        st.session_state.openai_model = 'gpt-4'
        st.session_state.openai_key = ''
        st.session_state.prototype_llm_call_logs = []
        st.session_state.prototype_question_logs = []
        st.session_state.prototype_summary_logs = []
        st.session_state.prototype_metrics_path = "prototype_llm_metrics.json"

        # 自动加载题库文件夹中的所有 txt 文件
        base_path = r".\apptemplates\QuestionBank\modelquestionbanks"
        txt_files = glob.glob(os.path.join(base_path, "*.txt"))
        st.session_state.model_question_bank_paths = txt_files

        # 加载题库内容
        banks = {}
        for path in txt_files:
            model_name = os.path.splitext(os.path.basename(path))[0].replace('_questions', '')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                banks[model_name] = content
                st.write(f"Loaded {os.path.basename(path)}...")
            except Exception as e:
                banks[model_name] = ''
                st.warning(f"Unable to load question bank {model_name}: {e}")

        st.session_state.model_question_banks = banks
        st.session_state.model_match_counts = {name: 0 for name in banks.keys()}
        st.session_state.recommended_models = []


def append_latest_openai_log(stage, question=None):
    log = set_for_LLM.get_last_call_log()

    if not log:
        return None

    enriched_log = {
        **log,
        "stage": stage,
        "question": question,
        "selected_model": st.session_state.get("selected_model", None),
        "dependent_var": st.session_state.get("dependent_var", None),
        "independent_vars": st.session_state.get("independent_vars", []),
    }

    st.session_state.prototype_llm_call_logs.append(enriched_log)
    return enriched_log


def summarize_llm_logs(logs):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_runtime_seconds = 0.0
    success_count = 0
    failure_count = 0

    for log in logs:
        usage = log.get("usage", {}) or {}

        total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
        total_completion_tokens += usage.get("completion_tokens", 0) or 0
        total_tokens += usage.get("total_tokens", 0) or 0
        total_runtime_seconds += log.get("runtime_seconds", 0.0) or 0.0

        if log.get("success"):
            success_count += 1
        else:
            failure_count += 1

    return {
        "llm_call_count": len(logs),
        "success_count": success_count,
        "failure_count": failure_count,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "runtime_seconds": total_runtime_seconds,
    }


def save_prototype_metrics_to_json():
    metrics = {
        "selected_model": st.session_state.get("selected_model", None),
        "dependent_var": st.session_state.get("dependent_var", None),
        "independent_vars": st.session_state.get("independent_vars", []),
        "background_text": st.session_state.get("background_text", ""),
        "question_logs": st.session_state.get("prototype_question_logs", []),
        "summary_logs": st.session_state.get("prototype_summary_logs", []),
        "all_llm_call_logs": st.session_state.get("prototype_llm_call_logs", []),
        "overall_summary": summarize_llm_logs(
            st.session_state.get("prototype_llm_call_logs", [])
        ),
    }

    with open(st.session_state.prototype_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)



# 统计用户问题与各模型题库的匹配次数
def compute_model_match_counts(user_questions):
    counts = {name: 0 for name in st.session_state.model_question_banks.keys()}
    # 确保背景已发送给 LLM
    # set_for_localLLM.set_chat_background([], st.session_state.background_text)  # 如需启用对话上下文
    if st.session_state.llm_mode == "ollama":
        for q in user_questions:
            for model_name, full_text in st.session_state.model_question_banks.items():
                if not full_text:
                    continue
                output = set_for_localLLM.question_matching(q, full_text)
                if extract_first_integer(output) != 0:
                    counts[model_name] += 1
    else:
        for q in user_questions:
            for model_name, full_text in st.session_state.model_question_banks.items():
                if not full_text:
                    continue

                st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(
                    q,
                    st.session_state.openai_chatmodel,
                    st.session_state.llm_messages
                )

                output, st.session_state.llm_messages = set_for_LLM.send_response_receive_output(
                    st.session_state.openai_url,
                    st.session_state.openai_headers,
                    st.session_state.payload,
                    st.session_state.llm_messages
                )

                append_latest_openai_log(
                    stage="model_recommendation_matching",
                    question=q
                )
                if extract_first_integer(output) != 0:
                    counts[model_name] += 1
    return counts

# 步骤1：数据上传
def step_upload_data():
    st.header("📤 Step 1: Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"], help="Standard CSV supported, max size 200MB"
    )
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.session_state.data_preview_expanded = True
        except Exception as e:
            st.error(f"❌ File read error: {e}")
            st.session_state.df = None
    if st.session_state.df is not None:
        with st.expander("Data preview", expanded=st.session_state.data_preview_expanded):
            st.dataframe(st.session_state.df, height=300, use_container_width=True)
            st.subheader("Data summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total rows", len(st.session_state.df))
            c2.metric("Number of variables", len(st.session_state.df.columns))
            c3.metric("Missing values", st.session_state.df.isna().sum().sum())
    c1, c2 = st.columns([1,1])
    with c1:
        if st.session_state.df is not None and st.button("Re-upload"):
            st.session_state.df = None
            st.rerun()
    with c2:
        if st.session_state.df is not None and st.button("Next  →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()
# --- 新增部分：让用户选择题库文件夹 ---
    st.divider()
    st.subheader("📁 Optional: Change question bank folder")

    custom_qb_path = st.text_input("Enter a new question bank folder path:", value=r".\apptemplates\QuestionBank\modelquestionbanks")

    if st.button("Load new question bank path"):
        import glob, os
        if os.path.isdir(custom_qb_path):
            txt_files = glob.glob(os.path.join(custom_qb_path, "*.txt"))
            if not txt_files:
                st.warning("⚠️ No .txt question bank files found in this folder.")
            else:
                banks = {}
                for path in txt_files:
                    model_name = os.path.splitext(os.path.basename(path))[0].replace('_questions', '')
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            banks[model_name] = f.read()
                        st.write(f"Loaded：{os.path.basename(path)}")
                    except Exception as e:
                        banks[model_name] = ''
                        st.warning(f"⚠️ Failed to load {path}: {e}")
                st.session_state.model_question_banks = banks
                st.session_state.model_match_counts = {k: 0 for k in banks.keys()}
                st.success(f"✅ Successfully loaded {len(txt_files)} question bank files.")
        else:
            st.error("❌ The provided path is not a valid folder.")




# 步骤2：变量选择
def step_select_variables():
    st.header("📊 Step 2: Variable Selection")
    if st.session_state.df is None:
        st.warning("Please upload the data file first")
        st.session_state.current_step = 1
        st.rerun()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Variable (Y)")
        st.session_state.dependent_var = st.selectbox("Select a dependent variable", st.session_state.df.columns)
    with col2:
        st.subheader("Feature Variable(s) (X)")
        avail = [c for c in st.session_state.df.columns if c != st.session_state.dependent_var]
        st.session_state.independent_vars = st.multiselect("Select the independent variable(s)", avail, default=avail[:3])
    if len(st.session_state.independent_vars) < 1:
        st.error("Please select at least one independent variable")
        return
    prev, _, nxt = st.columns([1,8,1])
    with prev:
        if st.button("← Back"):
            st.session_state.current_step = 1
            st.rerun()
    with nxt:
        if st.button("Next  →", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

# 步骤3：确认变量选择
def step_confirm_selection():
    st.header("✅ Step 3: Confirm Selection")

    st.subheader("Current Selection")
    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background:#f0f2f6">
    🔍 **Analysis Target**：Predict/Analyze {st.session_state.dependent_var}
    🛠️ **Features Used**：{', '.join(st.session_state.independent_vars)}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Variable Details", expanded=True):
        tab1, tab2 = st.tabs(["Target Variable Analysis", "Feature Overview"])

        with tab1:
            st.write(st.session_state.df[st.session_state.dependent_var].describe())
            if pd.api.types.is_numeric_dtype(st.session_state.df[st.session_state.dependent_var]):
                st.line_chart(st.session_state.df[st.session_state.dependent_var])
            else:
                st.bar_chart(st.session_state.df[st.session_state.dependent_var].value_counts())

        with tab2:
            st.dataframe(st.session_state.df[st.session_state.independent_vars].describe())

    st.divider()
    st.subheader("🤖 Choose LLM Source")

    llm_source = st.radio(
        "Select the LLM source for question matching and answering:",
        options=["Use local Ollama", "Call OpenAI API"],
        index=0,
        key="llm_source_choice"
    )

    if llm_source == "Call OpenAI API":
        st.session_state.llm_mode = "api"
        st.text_input("Enter model name", key="openai_model")
        st.text_input("Enter OpenAI API Key", type="password", key="openai_key")
    else:
        st.session_state.llm_mode = "ollama"

    col_prev, col_mid, col_next = st.columns([1, 8, 1])
    with col_prev:
        if st.button("← Reselect"):
            st.session_state.current_step = 2
            st.rerun()

    with col_next:
        if st.button("Start Modeling →", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

# 步骤4：输入数据集背景知识
def step_input_background():

    st.header("🌐 Step 4: Enter Dataset Background")
    st.markdown("Provide background information about the dataset for the LLM to reference during question matching. ")

    text = st.text_area("Enter background knowledge：", height=200, value=st.session_state.background_text)

    if text:
        st.session_state.background_text = text
        st.success("✅ Background information saved!")

        if st.session_state.llm_mode == "ollama":
            # 使用本地 LLM
            messages, bg = set_for_localLLM.set_chat_background([], text)
            st.session_state.llm_bg = bg
            st.session_state.llm_messages = messages
        else:
            # 使用 OpenAI API
            key = st.session_state.openai_key
            model = st.session_state.openai_model
            url, bg, chatmodel, headers, messages = set_for_LLM.set_chatGPT(text, key, model)

            st.session_state.llm_bg = bg
            st.session_state.llm_messages = messages
            st.session_state.openai_url = url
            st.session_state.openai_headers = headers
            st.session_state.openai_chatmodel = chatmodel

    prev, _, nxt = st.columns([1, 8, 1])
    with prev:
        if st.button("← Back to confirmation"):
            st.session_state.current_step = 3
            st.rerun()
    with nxt:
        if st.button("Next  →", type="primary"):
            if not st.session_state.background_text:
                st.error("Please enter background knowledge first!")
            else:
                st.session_state.current_step = 5
                st.rerun()


# 步骤5：批量输入问题并推荐模型
def step_input_questions():
    st.header("❓ Step 5: Batch Input Questions")
    with st.expander("Format instructions", expanded=True):
        st.markdown("1. Question one\n2. Question two\n3. Question three")
    text = st.text_area("Enter question list:", height=200)
    if text:
        parts = re.split(r"\n\s*\d+\.\s*", text.strip())
        st.session_state.user_questions = [p.strip() for p in parts if p.strip()]
    if st.session_state.user_questions:
        st.success("Parsed the following questions：")
        for i, q in enumerate(st.session_state.user_questions, 1): st.write(f"{i}. {q}")
        counts = compute_model_match_counts(st.session_state.user_questions)
        st.session_state.model_match_counts = counts
        max_cnt = max(counts.values()) if counts else 0
        recommended = [m for m,c in counts.items() if c==max_cnt and c>0]
        st.session_state.recommended_models = recommended
        st.subheader("🔍 Matching Statistics")
        for m,c in counts.items(): st.write(f"- {m}: {c}")
        if recommended:
            st.success(f"⭐ Recommended：{', '.join(recommended)}")
        else:
            st.warning("⚠️ No suitable model matched")



    prev,mid,nxt = st.columns([1,1,1])
    with prev:
        if st.button("← Back to background"): st.session_state.current_step=4; st.rerun()
    with mid:
        if st.button("Skip recommendation"): st.session_state.current_step=6; st.rerun()
    with nxt:
        if st.button("Next  →"):
            if not st.session_state.recommended_models:
                st.error("Please complete model matching first!")
            else:
                st.session_state.current_step=6; st.rerun()

# 步骤6：选择并拟合模型
def step_select_model():
    st.header("⚙️ Step 6: Select and Fit Model")
    reg_opts = ["Statsmodels Linear Regression","Scikit-learn Linear Regression","Ridge Regression","Gradient Boosting Regression","Random Forest Regression","Lasso Regression","Bayesian Ridge Regression","Auto-fit Best Regression"]
    cls_opts = ["Logistic Regression","Linear Discriminant Analysis","SVM - Linear Kernel","Ridge Classifier","Random Forest Classifier","Decision Tree Classifier","Auto-fit Best Classifier"]
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Regression")
        st.session_state.reg_choice = st.radio("Select a regression model", ["None"]+reg_opts)
        if st.session_state.reg_choice=="Ridge Regression": st.number_input("Ridge Alpha", key='ridge_alpha')
        if st.session_state.reg_choice=="Lasso Regression": st.number_input("Lasso Alpha", key='lasso_alpha')
        if st.session_state.reg_choice=="Bayesian Ridge Regression":
            st.number_input("Alpha1", key='bayes_a1')
            st.number_input("Alpha2", key='bayes_a2')
            st.number_input("Lambda1", key='bayes_l1')
            st.number_input("Lambda2", key='bayes_l2')
        if st.session_state.reg_choice=="Auto-fit Best Regression": st.text_input("Criterion", key='reg_crit')
    with c2:
        st.subheader("Classification")
        st.session_state.cls_choice = st.radio("Select a classification model", ["None"]+cls_opts)
        if st.session_state.cls_choice=="Auto-fit Best Classifier": st.text_input("Criterion", key='cls_crit')
    if st.button("Next  →"):
        r, c = st.session_state.reg_choice, st.session_state.cls_choice
        if (r!="None")==(c!="None"):
            st.error("Only one model can be selected!"); return
        df = st.session_state.df
        X = df[st.session_state.independent_vars]
        y = df[st.session_state.dependent_var]
        results = {}
        if r!="None":
            # 数据清洗
            df_clean = ds_data_engineering.clean_data(df, 0.8, Xcol=st.session_state.independent_vars, ycol=st.session_state.dependent_var)
            Xc = df_clean[st.session_state.independent_vars]
            yc = df_clean[st.session_state.dependent_var]
            # 拟合回归
            if r=="Statsmodels Linear Regression": coef,pval,r2 = ds_regression.train_sm_linear_regression(Xc,yc)
            elif r=="Scikit-learn Linear Regression": coef,pval,r2 = ds_regression.train_sk_linear_regression(Xc,yc)
            elif r=="Ridge Regression": coef,pval,r2 = ds_regression.train_ridge_regression(Xc,yc,st.session_state.ridge_alpha)
            elif r=="Lasso Regression": coef,pval,r2 = ds_regression.train_lasso_regression(Xc,yc,st.session_state.lasso_alpha)
            elif r=="Bayesian Ridge Regression": coef,pval,r2 = ds_regression.train_BayesianRidge_regression(
                Xc,yc,st.session_state.bayes_a1,st.session_state.bayes_a2,st.session_state.bayes_l1,st.session_state.bayes_l2)
            elif r=="Gradient Boosting Regression": coef,pval,r2,train_r2,test_r2 = ds_regression.train_gradient_boosting_regression(Xc,yc); results.update({'train_r2':train_r2,'test_r2':test_r2})
            elif r=="Random Forest Regression": coef,pval,r2,train_r2,test_r2 = ds_regression.train_random_forest_regression(Xc,yc); results.update({'train_r2':train_r2,'test_r2':test_r2})
            elif r=="Auto-fit Best Regression": name,detail,crit,_ = ds_data_engineering.find_best_regression(Xc,yc,st.session_state.dependent_var,st.session_state.reg_crit,st.session_state.independent_vars); results.update({'model_name':name,'detail':detail,'criterion':crit})
            # 生成系数与P值表
            coef_df = ds_regression.coefficients_with_Pvalues(coef,pval,st.session_state.independent_vars)
            results.update({'coeff_df':coef_df,'r_squared':r2})
            st.session_state.selected_model = r
            st.session_state.fit_results = results
        else:
            # 拟合分类
            if c=="Logistic Regression": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_logistic_regression(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Linear Discriminant Analysis": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_linear_discriminant_analysis(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="SVM - Linear Kernel": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_SVC_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Ridge Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_ridge_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Random Forest Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_random_forest_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            elif c=="Decision Tree Classifier": model,coeff_df,acc,coeff,train_acc,test_acc = ds_classifier.train_decision_tree_classifier(X,y,st.session_state.independent_vars,st.session_state.dependent_var)
            else: name,detail,crit,_ = ds_data_engineering.find_best_classifier(X,y,st.session_state.dependent_var,st.session_state.cls_crit,st.session_state.independent_vars); results.update({'model_name':name,'detail':detail,'criterion':crit}); model=None; coeff_df=None; acc=None; train_acc=None; test_acc=None
            results.update({'coeff_df':coeff_df,'accuracy':acc,'train_accuracy':train_acc,'test_accuracy':test_acc})
            st.session_state.selected_model = c
            st.session_state.fit_results = results
        st.session_state.current_step = 7
        st.rerun()

# 步骤7：单题提问
def step_input_single_question():
    st.header("❓ Step 7: Ask a Single Question")
    st.session_state.single_question = st.text_input("Enter your question：", value="", key='q_input')
    if st.session_state.single_question and st.button("Submit question →"):
        q = st.session_state.single_question
        question_log_start_index = len(st.session_state.prototype_llm_call_logs)
        # 匹配模板节号
        if 'Regression' in st.session_state.selected_model:
            content = loader.load_regression_questions()
        else:
            content = loader.load_classifier_questions()
        if st.session_state.llm_mode == "ollama":
            sec = extract_first_integer(set_for_localLLM.question_matching(q, content))
        else:
            st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(q,
                                                                                              st.session_state.openai_chatmodel,
                                                                                              st.session_state.llm_messages)
            output, st.session_state.llm_messages = set_for_LLM.send_response_receive_output(
                st.session_state.openai_url,
                st.session_state.openai_headers,
                st.session_state.payload,
                st.session_state.llm_messages
            )

            append_latest_openai_log(
                stage="single_question_matching",
                question=q
            )

            sec = extract_first_integer(output)
        # 获取模板问答
        if 'Regression' in st.session_state.selected_model:
            coef_df = st.session_state.fit_results.get('coeff_df')
            if sec==1:
                qs,ans = nlg_reg.Q_and_A_about_R2(coef_df, st.session_state.dependent_var, st.session_state.selected_model, st.session_state.fit_results.get('r_squared'))
            elif sec==2:
                qs,ans = nlg_reg.Q_and_A_about_coefficients(coef_df, st.session_state.dependent_var)
            elif sec==3:
                qs,ans = nlg_reg.Q_and_A_about_importance(st.session_state.independent_vars, st.session_state.dependent_var, st.session_state.fit_results.get('coeff'), coef_df)
            elif sec==4:
                qs,ans = nlg_reg.Q_and_A_about_pvalues(coef_df, st.session_state.dependent_var)
            elif sec==5:
                qs,ans = nlg_reg.Q_and_A_about_ML_importance(st.session_state.independent_vars, st.session_state.dependent_var, coef_df)
            elif sec==6:
                qs,ans = nlg_reg.Q_and_A_about_ML_overfit(st.session_state.fit_results.get('train_r2'), st.session_state.dependent_var, st.session_state.fit_results.get('test_r2'))
            else:
                qs,ans = [],[]
        else:
            coeff_df = st.session_state.fit_results.get('coeff_df')
            if sec==1:
                qs,ans = nlg_cls.Q_and_A_about_accuracy(st.session_state.fit_results.get('accuracy'), st.session_state.selected_model)
            elif sec==2:
                qs,ans = nlg_cls.Q_and_A_about_coefficients(coeff_df, st.session_state.dependent_var)
            elif sec==3:
                qs,ans = nlg_cls.Q_and_A_about_importance(st.session_state.fit_results.get('coeff'), st.session_state.dependent_var, coeff_df, st.session_state.fit_results.get('model').classes_)
            elif sec==4:
                qs,ans = nlg_cls.Q_and_A_about_ML_overfit(st.session_state.fit_results.get('train_accuracy'), st.session_state.fit_results.get('test_accuracy'))
            elif sec==5:
                qs,ans = nlg_cls.Q_and_A_about_ML_importance(coeff_df, st.session_state.dependent_var)
            else:
                qs,ans = [],[]
        # 基于模板回答更新
        default_answer = set_for_GPT.answer_update(q, qs, ans)
        if st.session_state.llm_mode == "ollama":
            out, msgs = set_for_localLLM.send_response_receive_output(default_answer, st.session_state.llm_bg, st.session_state.llm_messages)
            st.session_state.llm_messages = msgs
            st.session_state.single_answer = out
        else:
            st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(
                default_answer,
                st.session_state.openai_chatmodel,
                st.session_state.llm_messages
            )

            out, msgs = set_for_LLM.send_response_receive_output(
                st.session_state.openai_url,
                st.session_state.openai_headers,
                st.session_state.payload,
                st.session_state.llm_messages
            )

            append_latest_openai_log(
                stage="answer_refinement",
                question=q
            )

            st.session_state.llm_messages = msgs
            st.session_state.single_answer = out


        question_logs = st.session_state.prototype_llm_call_logs[question_log_start_index:]

        st.session_state.prototype_question_logs.append(
            {
                "question": q,
                "matched_section": sec,
                "answer": st.session_state.single_answer,
                "llm_logs": question_logs,
                "summary": summarize_llm_logs(question_logs)
            }
        )

        save_prototype_metrics_to_json()


        st.session_state.current_step = 8
        st.rerun()

# 步骤8：显示回答并可再次提问
def step_show_single_answer():
     st.header("💬 Step 8: Answer Result")
     # 显示前一步的回答
     st.write(st.session_state.single_answer)
     # 按钮栏：再次提问 或 生成总结
     col_reask, col_summary = st.columns([1,1])
     with col_reask:
         if st.button("Ask again"):
             st.session_state.current_step = 7
             st.rerun()
     with col_summary:
         if st.button("Generate summary"):
             # 构造总结提示语
             summary_prompt = (
                 "Please summarize all the above Q&A interactions into a concise report focusing on the user's questions "
                 "about the dataset and the chosen model, without explaining the process."
             )
             # 调用 LLM 生成总结
             if st.session_state.llm_mode == "ollama":
                 out, msgs = set_for_localLLM.send_response_receive_output(
                     summary_prompt,
                     st.session_state.llm_bg,
                     st.session_state.llm_messages
                 )
                 # 更新消息历史并保存 summary
                 st.session_state.llm_messages = msgs
                 st.session_state.summary = out
             else:
                 st.session_state.payload, st.session_state.llm_messages = set_for_LLM.set_payload(summary_prompt,
                                                                                                   st.session_state.openai_chatmodel,
                                                                                                   st.session_state.llm_messages)
                 out, msgs = set_for_LLM.send_response_receive_output(
                     st.session_state.openai_url,
                     st.session_state.openai_headers,
                     st.session_state.payload,
                     st.session_state.llm_messages
                 )

                 summary_log = append_latest_openai_log(
                     stage="summary_generation",
                     question=None
                 )

                 if summary_log:
                     st.session_state.prototype_summary_logs.append(summary_log)

                 # 更新消息历史并保存 summary
                 st.session_state.llm_messages = msgs
                 st.session_state.summary = out

                 save_prototype_metrics_to_json()
             # 在界面显示总结
             st.subheader("📄 Summary Report")
             st.write(st.session_state.summary)
             # 将聊天记录保存为 docx，通知用户
             set_for_localLLM.save_chat_history_to_docx(st.session_state.llm_messages)
             st.success("✅ Report generated and saved as data_report.docx in the server root directory.")

 # 主流程
def main():
     st.set_page_config(page_title="Intelligent Data Analysis Assistant", layout="wide")
     initialize_session_state()
     steps=["上传","变量","确认","背景","批量问","模型选择","单题问答","答案展示"]
     st.progress((st.session_state.current_step-1)/len(steps))
     if st.session_state.current_step==1: step_upload_data()
     elif st.session_state.current_step==2: step_select_variables()
     elif st.session_state.current_step==3: step_confirm_selection()
     elif st.session_state.current_step==4: step_input_background()
     elif st.session_state.current_step==5: step_input_questions()
     elif st.session_state.current_step==6: step_select_model()
     elif st.session_state.current_step==7: step_input_single_question()
     elif st.session_state.current_step==8: step_show_single_answer()

if __name__=="__main__":
     main()

