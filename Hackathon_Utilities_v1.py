import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
from dowhy import CausalModel
import openai
from sqlalchemy import create_engine  # For database connections
from OpenAITokenHackathon import openai_token_for_param
import os
from openai import AzureOpenAI
import warnings
warnings.filterwarnings('ignore')


# Simulated GPT API (Replace with actual GPT API call)
def simulate_gpt_api(prompt):
    endpoint = os.getenv("ENDPOINT_URL", "https://psgexptest.openai.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "HackathonExperiment")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", openai_token_for_param)

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint = endpoint,
        api_key = subscription_key,
        api_version = "2024-02-15-preview",
    )

    completion = client.chat.completions.create(
        model=deployment,
        messages= prompt,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    return completion.choices[0].message.content

# Step 1: Dynamic Data Loading
def load_data(data_source, data_type='csv', db_info=None):
    if data_type == 'csv':
        df = pd.read_csv(data_source)
    elif data_type == 'database':
        db_engine = create_engine(f"{db_info['dialect']}://{db_info['username']}:{db_info['password']}@{db_info['host']}:{db_info['port']}/{db_info['database']}")
        df = pd.read_sql(data_source, con=db_engine)
    else:
        raise ValueError("Unsupported data type. Please provide 'csv' or 'database'.")
    return df

# Step 2: Generate a Detailed Summary Using GPT API (or Simulated GPT API)
def convert_string_to_dict(input_string):
    return dict(pair.split(':') for pair in input_string.strip('{}').split(','))


def generate_objective_variables(sentence, col_names):
    # Proper string formatting using f-string
    prompt = (
        f"You are an AI that extracts causal relationships from sentences. "
        f"Your task is to identify the treatment and outcome from the given sentence "
        f"and output them in the format {{treatment: <treatment_variable>, outcome_var: <outcome_variable>}}.\n\n"
        f"Here are the column names from the data: {col_names}\n\n"
        f"Here are some examples:\n"
        f"- Input: 'How does Overall Quality impact House Price?'\n"
        f"  Output: {{treatment: OverallQual, outcome_var: SalePrice}}\n\n"
        f"- Input: 'What is the effect of education level on salary?'\n"
        f"  Output: {{treatment: Education, outcome_var: Salary}}\n\n"
        f"- Input: 'How does advertising spending influence product sales?'\n"
        f"  Output: {{treatment: AdSpending, outcome_var: ProductSales}}\n\n"
        f"Now, identify the treatment and outcome for the following sentence: '{sentence}'"
    )

    # Create the message structure for the GPT API
    message = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps extract treatment variable and outcome variable from a sentence."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Simulated call to GPT API (you can replace with actual API call)
    print('Entered Message: ',sentence)
    summary = simulate_gpt_api(message)
    return summary


# Step 3: Automate Model Selection Based on Task
def select_and_train_model(X, y=None, task='classification'):
    model = None
    feature_importance_df = None

    if task == 'classification':
        # Train RandomForest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Classification Accuracy: {accuracy}")

        # Calculate feature importance
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

    elif task == 'regression':
        # Train RandomForest Regressor
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        print(f"Regression MSE: {mse}")

        # Calculate feature importance
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

    elif task == 'clustering':
        # Train KMeans Clustering
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        print(f"KMeans Clusters: {np.unique(model.labels_)}")
        
        # Feature importance isn't applicable for clustering, return None for the DataFrame

    return model, feature_importance_df


# Step 4: Generate SHAP explanations for the model
def generate_shap_explanations(model, X, task='supervised',route='Classification'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Plot summary of SHAP values
    # shap.summary_plot(shap_values, X)

    if route=='Classification':
        shap_values=shap_values[:,:,1]

    df=pd.DataFrame([X.columns,np.abs(shap_values).mean(axis=0).T])
    df=df.T
    df.columns=["Features",'Contribution']
    df.sort_values(by=['Contribution'],ascending=False,inplace=True)
    top_feature_name=df.iloc[0,0]
    top_feature_name
    
    # Return SHAP values for further causal analysis
    return df,top_feature_name

# Step 5: Perform Causal Inference using DoWhy
from dowhy import CausalModel

def causal_inference(df, treatment_var, outcome_var, common_causes=None, mode='Classification'):
    # Initialize the causal model
    if mode=='Classification':
        common_causes=['Age']
    model = CausalModel(
        data=df,
        treatment=treatment_var,
        outcome=outcome_var,
        common_causes=common_causes
    )

    # Identify causal effect
    identified_estimand = model.identify_effect()

    # Choose the appropriate method based on classification or regression
    if mode=='Classification':
        # Use propensity score stratification for classification tasks
        causal_estimate = model.estimate_effect(
            identified_estimand, 
            method_name="backdoor.linear_regression" 
        )
    else:
        causal_estimate = model.estimate_effect(
            identified_estimand, 
            method_name="backdoor.linear_regression"
        )

    # Get the confidence interval for the causal estimate
    confidence_interval = causal_estimate.get_confidence_intervals()

    # Return the causal estimate and confidence interval
    return causal_estimate, confidence_interval

def top_k_confounders(data, outcome = '', treatment = '', features = '', categorical_columns = None, top_k = 10):
 
    """
        Regresses outcome ~ features, treatment ~ features and returns the top_k commom variables as confounders
        any categorical features other than outcome and treatment need to be One Hot encoded
        treatment and outcome should be numerical values
        data: DataFrame containing data to be analysed
        top_k : return the top 10 commom variable among the 2 regressions
    """

 
    if categorical_columns == None:
        categorical = list(data.apply(lambda series: series.dtype).loc[lambda df: df.eq('object')].index)
 
    X = data.drop(columns = outcome)
    X_ohe = (X.pipe(pd.get_dummies, prefix_sep = '_OHE_', columns = categorical, dtype='uint8'))
    Y = data[outcome]
    top_k_1 = 40
    print("Regressing outcome ~ X")
    RMmodel1 = RandomForestRegressor().fit(X_ohe,Y)
    feature_df = pd.concat([pd.Series(RMmodel1.feature_importances_), 
                            pd.DataFrame(X_ohe.columns.values)], 
                           axis = 1, ignore_index=True).sort_values(by = 0, ascending=False).head(top_k_1)
 
    top_k_2 = 20
    print("Regressing treatment ~ X")
    Y = data[treatment]
    X = X_ohe.drop(columns = treatment)
    RMmodel2 = RandomForestRegressor().fit(X,Y)
    feature_df_treatment = pd.concat([pd.Series(RMmodel2.feature_importances_), 
                                      pd.DataFrame(X.columns.values)], 
                                     axis = 1, ignore_index=True).sort_values(by = 0, ascending=False).head(top_k_2)
    feature_df_treatment.columns = ['Importance', 'FName']
    feature_df.columns = ['Importance', 'FName']
    confounders_df = feature_df.merge(feature_df_treatment, on = 'FName', how = 'inner')
    return list(confounders_df['FName'].values)


# Step 6: Generate a Detailed Summary Using GPT API (or Simulated GPT API)
def generate_detailed_summary(model_type, shap_importance,top_feature, causal_estimate,confidence_interval,confounders):
    prompt = (
        f"You are an expert in causal inference. Please explain the causal analysis wrt the treatment feature in simple terms. Keep the summary within 100 words\n\n"
        f"Model Type: {model_type}\n"
        f"Feature Importance Table: {shap_importance}\n"
        f"Treatment Feature: {top_feature}\n"
        f"Treatment feature's Causal Estimate: {causal_estimate.value}\n"
        f"Treatment feature's Causal Estimate Confidence Interval: {confidence_interval}\n"
        f"Top Confounder Variables identified by th Model: {confounders}\n"
        f"Summarize actionable insights from the above. Answer in bullet points- with apporpiate captions"
    )
    message = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps explain causal relationships among features with the outcome of the data."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    summary = simulate_gpt_api(message)
    return summary

# Main function to handle the entire flow
def main(data_source, data_type='csv', task='classification', ask=None, db_info=None, outcome_var=None):
    # Step 1: Load data from any source
    df = data_source.copy()
    dict_soln=generate_objective_variables(ask,df.columns.tolist())
    result = convert_string_to_dict(dict_soln)
    treatment,outcome_var=result['treatment'].strip(),result[' outcome_var'].strip()
    
    # Step 2: Prepare Data (Handle missing values, splits, etc.)
    X = df.drop([outcome_var], axis=1) if outcome_var else df
    y = df[outcome_var] if outcome_var else None
    
    # Step 3: Train the model based on the task
    model,feature_importance_df = select_and_train_model(X, y, task=task)
    
    confounders=top_k_confounders(df, outcome = outcome_var, treatment = treatment, features = df.columns.tolist(), top_k = 10)
    # Step 5: Perform Causal Inference (only if treatment and outcome are provided)
    causal_estimate = None
    if outcome_var:
        causal_estimate,confidence_interval = causal_inference(df, treatment, outcome_var,common_causes=confounders,mode=task)

    # Step 6: Generate Detailed Summary using GPT API
    result_output=generate_detailed_summary(task, feature_importance_df,treatment, causal_estimate,confidence_interval,confounders)
    return result_output
        
