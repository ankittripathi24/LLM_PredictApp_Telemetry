import pandas as pd
import numpy as np
import csv
import os
import sys
from langchain_openai import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
# Removed unused import
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# DeepEval
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval.metrics import HallucinationMetric


def identify_delimiter(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        elif ',' in first_line:
            return ','
        else:
            raise ValueError("Unknown delimiter in the CSV file.")


def load_config(config_file_path):
    config = {}
    with open(config_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            config = row
            break  # Assuming the CSV has only one row of configuration
    return config

def preprocess_data(data, config):
    print(" before preprocessec data is: ", data)

    # Iterate over all columns and print their types
    print("Data types of all columns:")
    for column in data.columns:
        print(f"Column: {column}, Type: {data[column].dtype}")

    # Print percentage of null or NaN values for each column
    print("Percentage of null or NaN values for each column:")
    columns_to_drop = []
    for column in data.columns:
        null_percentage = data[column].isnull().mean() * 100
        print(f"Column: {column}, Null Percentage: {null_percentage:.2f}%")
        if null_percentage > 50:
            columns_to_drop.append(column)

    # Drop columns with more than 50% null values
    if columns_to_drop:
        print(f"Dropping columns with more than 50% null values: {columns_to_drop}")
        data = data.drop(columns=columns_to_drop)


    # # Ensure the _time column is retained
    # time_column = data['_time'] if '_time' in data.columns else None

    # # Drop columns that do not have double or integer values
    # data = data.select_dtypes(include=[np.number])
    
    # Drop columns with names ending in '_qc'
    data = data.loc[:, ~data.columns.str.endswith('_qc')]
    
    # # Reattach the _time column if it was dropped
    # if time_column is not None:
    #     data['_time'] = time_column
    
    print(" before inbetween preprocessec data is: ", data)

    if config['missing_data_handling'] == 'Drop':
        data = data.dropna()
    elif config['missing_data_handling'] == 'Fill':
        data = data.fillna(method='ffill').fillna(method='bfill')

    print(" inbetween preprocessec data is: ", data)

    if config['normalization'] == 'Min-Max':
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)
        pass
    elif config['normalization'] == 'Z-Score':
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        pass
    print(" preprocessec data is: ", data)
    return data


def summarize_data(data, window_size=10, max_rows=100):
    """Summarize the data using a rolling window and downsampling to fit within the model's context length."""
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Apply rolling window and calculate the mean
    summarized_data = numeric_data.rolling(window=window_size).mean().dropna()
    
    # Downsample the data if it exceeds max_rows
    if len(summarized_data) > max_rows:
        summarized_data = summarized_data.sample(n=max_rows, random_state=1)
    
    return summarized_data

# from langchain.callbacks.confident_callback import DeepEvalCallbackHandler
from langchain_community.callbacks.confident_callback import DeepEvalCallbackHandler
from langchain.prompts import PromptTemplate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


def identify_trends(data, llm_azure_openai):
    """Identify trends in the data."""

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    # Initialize the DeepEvalCallbackHandler
    deep_eval_handler = DeepEvalCallbackHandler(
        metrics=[answer_relevancy_metric]
    )

    llm_azure_openai = AzureChatOpenAI(
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        # azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
        # openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        callbacks=[deep_eval_handler],
        temperature=0.7
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=[data],
        template="""
        You are provided with time series data in CSV format that may contain various patterns or trends. Your task is to analyze and identify any observable trends in the data.

        Please follow these guidelines for your analysis:

        Carefully examine the data for any identifiable patterns such as increasing, decreasing, cyclic, or seasonal trends.
        Provide a concise and detailed description of the trends observed. Ensure the analysis covers the nature, magnitude, and direction of trends.
        If trends are complex, summarize them clearly, being mindful of clarity over verbosity.
        If you are unable to identify any trends, state “unidentifiable trend.”
        This information will be used as input for a subsequent prompt for time series prediction.

        The time series data is provided below, in the format:
        {data}
        """
    )

    # Create the LLM chain
    llm_chain = LLMChain(
        llm=llm_azure_openai,
        prompt=prompt_template  # Add the callback handler to the chain
    )

    # Convert the data to string format
    data_str = data.to_string(index=False)
    
    # Run the LLM chain with the data
    response = llm_chain.run({
        "data": data_str
    })

    print(answer_relevancy_metric.is_successful())
    test_case = LLMTestCase(
        input=prompt_template,
        actual_output = response,
        retrieval_context=["The data appears to be quite random with no clear upward or downward trend over the entire series"]
    )
    context=["The data appears to be quite random with no clear upward or downward trend over the entire series"]
    test_case_1 = LLMTestCase(
        input="What was the blond doing?",
        actual_output=response,
        context=context
    )
    # ---------------------------------------------------------------
    # Initialize metrics

    metric_Relevancy = AnswerRelevancyMetric(threshold=0.7)
    metric_Toxicity = ToxicityMetric(threshold=0.7)
    metric_Hallucination = HallucinationMetric(threshold=0.5)

    metric_Hallucination.measure(test_case_1)
    print("metric_Hallucination score: ",metric_Hallucination.score)
    print("metric_Hallucination score: ",metric_Hallucination.reason)

    # Initialize metric with threshold
    # metric2 = AnswerRelevancyMetric(threshold=0.7)

    metric_Relevancy.measure(test_case)
    print("metric_Relevancy score: ", metric_Relevancy.score)
    print("metric_Relevancy reason:",metric_Relevancy.reason)

    metric_Toxicity.measure(test_case)
    print("metric_ Toxicity score: ", metric_Toxicity.score)
    print("metric_ Toxicity reason:",metric_Toxicity.reason)
    # -------------------------------------------------------------

    return response


def predict_with_langchain_agent(data, config, trend, llm_azure_openai, steps=10):
    # llm_azure_openai = AzureChatOpenAI(
    #     openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
    #     azure_deployment="gpt-4o-mini",
    #     api_version="2024-08-01-preview",
    #     temperature=0.7
    # )
    
    # Ensure the data has a timestamp column
    if '_time' in data.columns:
        data.rename(columns={'_time': 'timestamp'}, inplace=True)
    elif 'timestamp' in data.columns:
        pass
    else:
        print("FAILED")
        print("data is: ", data)
        # break the execution here
        return "False"
    
    print("data is: ", data)

    # Create a Pandas DataFrame agent with allow_dangerous_code set to True
    agent = create_pandas_dataframe_agent(llm_azure_openai, data, verbose=True,  allow_dangerous_code=True)
    
    # Define the prompt for the agent
    prompt = f"""
       You are given time series data in CSV format and a description of the trend observed in the data. Your task is to predict the next {steps} steps in the series.

        Please follow these steps:

        Read and analyze the provided time series data.
        Predict the next {steps} steps based on the observed trend.
        Provide the predicted values as a list.
        Calculate and report the accuracy and precision of your prediction.

        The trend in the data is described as follows: “{trend}”

        Please predict the next {steps} steps, report the predicted values as a list, and also include the accuracy and precision of your prediction.
    """

    prompt = f'''
    You are provided with time series data in CSV format and a description of the trend observed in the data. Your task is to predict the next {steps} steps in the series.

    Please follow these steps:

    1. **Data Preparation:**
    - Read the provided time series data.
    - Split the data into training (80%) and testing (20%) sets.

    2. **Prediction:**
    - Use the appropriate algorithm to build a predictive model based on the training data.
    - Predict the next {steps} steps based on the model.

    3. **Validation:**
    - Test the accuracy and precision of the prediction using the testing data (20%).
    - Iterate the prediction process, adjusting as necessary to achieve the highest accuracy.

    4. **Output:**
    - Provide the predicted values for the next {steps} steps as a list.
    - Report the highest accuracy and precision achieved during the iterations.

    The time series data is provided below, surrounded by triple quotes.
    """
    {data}
    """

    The trend in the data is described as follows: "{trend}"

    Please proceed with the prediction and validation based on these instructions.
    '''
    
    # Run the agent with the prompt and handle parsing errors
    response = agent.run(input=prompt, handle_parsing_errors=True)
    
    return response

def findings_with_langchain_agent(data, llm_azure_openai, steps=10):
    
    
    # Ensure the data has a timestamp column
    if '_time' in data.columns:
        data.rename(columns={'_time': 'timestamp'}, inplace=True)
    else:
        data['timestamp'] = pd.date_range(start='1/1/2020', periods=len(data), freq='D')
    
    # Create a Pandas DataFrame agent with allow_dangerous_code set to True
    agent = create_pandas_dataframe_agent(llm_azure_openai, data, verbose=True, allow_dangerous_code=True)
    
    # Define the prompt for the agent
    prompt = f"""
    Given the following time series data, Generate natural language summaries of the key findings from the data analysis.
    Write a detailed report on the key insights and trends observed in the data.
    """
    
    # Run the agent with the prompt and handle parsing errors
    response = agent.run(input=prompt, handle_parsing_errors=True)
    
    return response

def predict_with_langchain(data, config, steps=10):
    llm_azure_openai = AzureChatOpenAI(
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
        azure_deployment="gpt-4o-mini",
        api_version="2024-08-01-preview",
        temperature=0.7
    )

    prompt_template = PromptTemplate(
        input_variables=["data", "algorithm", "steps"],
        template="""
        Given the following time series data:
        {data}

        Create predictions of the next {steps} steps. Provide the predicted values.
        Return the predictions as a list of values without any additional explanations or code suggestions.
        """
    )

    llm_chain = LLMChain(
        llm=llm_azure_openai,
        prompt=prompt_template
    )

    data_str = data.to_string(index=False)
    response = llm_chain.run({
        "data": data_str,
        "algorithm": config['algorithm'],
        "steps": steps
    })

    return response

# def test_deepval(llm_azure_openai):
    

#     metric2 = AnswerRelevancyMetric(model=llm_azure_openai)
#     metric = ToxicityMetric(threshold=0.5)
#     test_case = LLMTestCase(
#         input="How is Sarah as a person?",
#         # Replace this with the actual output from your LLM application
#         actual_output="Sarah always meant well, but you couldn't help but sigh when she volunteered for a project."
#     )

#     # Initialize metric with threshold
#     metric2 = AnswerRelevancyMetric(threshold=0.7)

#     metric.measure(test_case)
#     print(metric.score)
#     print(metric.reason)

#     metric2.measure(test_case)
#     print(metric2.score)
#     print(metric2.reason)

def main(config_file_path, data_file_path, llm_azure_openai):

    # test = test_deepval(llm_azure_openai)  # Commented out as test_deepval is not defined

    print("Loading configurations and data...")
    config = load_config(config_file_path)

    if config['data_format'] == 'CSV':
        # data = pd.read_csv(data_file_path, delimiter=';')
        delimiter = identify_delimiter(data_file_path)
        data = pd.read_csv(data_file_path, delimiter=delimiter)
    
    elif config['data_format'] == 'JSON':
        data = pd.read_json(data_file_path)

    print("Preprocessing data...")
    data = preprocess_data(data, config)

    print("Summarizing data...")
    data = summarize_data(data)

    print("Identifying trends with LangChain and Azure OpenAI...")
    trends = identify_trends(data, llm_azure_openai)
    print("Trends Identified:")
    print(trends)

    print("Finding key insights with LangChain and Azure OpenAI...")
    findings = findings_with_langchain_agent(data,  llm_azure_openai)
    print("Findings Identified:")
    print(findings)

    print("Making predictions with LangChain and Azure OpenAI...")
    forecast = predict_with_langchain_agent(data, config, trends, llm_azure_openai)


    print("Results:")
    if config['task'] == 'Prediction':
        print("Prediction Results:")
        print(forecast)
    elif config['task'] == 'Forecasting':
        print("Forecasting Results:")
        print(forecast)
    elif config['task'] == 'Anomaly Detection':
        print("Anomaly Detection Results:")
        # Placeholder for anomaly detection logic

    # Extract unique ID from config_file_path
    unique_id = os.path.basename(config_file_path).split('_')[-1].replace('.csv', '')
    forecast_file_name = f"transactions\\forecast_{unique_id}.txt"

    # Save the forecast to a TXT file
    with open(forecast_file_name, 'w') as file:
        # for value in forecast:
        file.write(f"{forecast}\n")

    print(f"Forecast saved to {forecast_file_name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python time_series_prediction.py <config_file_path> <data_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    data_file_path = sys.argv[2]

    os.environ["AZURE_OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "Standard"
    os.environ["AZURE_OPENAI_MODEL_NAME"] = "gpt-4"

    os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

    llm_azure_openai = AzureChatOpenAI(
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        # azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
        # openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0.7
    )

    main(config_file_path, data_file_path, llm_azure_openai)