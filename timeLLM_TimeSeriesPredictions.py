import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
openai.api_key = "<paste your api key here>"
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NBEATS
from neuralforecast.losses.numpy import mae

def identify_delimiter(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        elif ',' in first_line:
            return ','
        else:
            raise ValueError("Unknown delimiter in the CSV file.")

def read_csv_with_dynamic_delimiter(file_path):
    delimiter = identify_delimiter(file_path)
    data = pd.read_csv(file_path, delimiter=delimiter)
    return data

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(train, label='Train')
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='yellow')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.title(f'{model_name} - NLL/D: {nll:.2f}')
    plt.show()

# Example usage
if __name__ == "__main__":
    data_file_path = r"transactions\XIOTLab_DataAsset_XIOTLabAspect.csv"  # Use raw string to avoid escape sequence issues
    data = read_csv_with_dynamic_delimiter(data_file_path)
    
    # Filter relevant columns
    data = data[['_time', 'RoomTemperature']]
    
    # Convert _time to datetime with the correct format
    data['_time'] = pd.to_datetime(data['_time'], format="%Y-%m-%dT%H:%M:%S.%fZ", errors='coerce')
    data['_time'].fillna(pd.to_datetime(data['_time'], format="%Y-%m-%dT%H:%M:%SZ", errors='coerce'), inplace=True)
    
    # Remove duplicate timestamps
    data = data[~data['_time'].duplicated()]
    
    # Set _time as index
    data.set_index('_time', inplace=True)
    
    # Rename columns for compatibility with NeuralForecast
    data.rename(columns={'RoomTemperature': 'y'}, inplace=True)
    
    # Add unique_id column
    data['unique_id'] = 1
    
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    # Prepare the data for NeuralForecast
    Y_train_df = train.reset_index().rename(columns={'_time': 'ds'})
    Y_test_df = test.reset_index().rename(columns={'_time': 'ds'})
    
    # Initialize and fit the models
    nbeats = NBEATS(h=12, input_size=36, max_steps=100)
    mlp = MLP(h=12, input_size=36, max_steps=100)
    nf = NeuralForecast(models=[nbeats, mlp], freq='M')
    
    nf.fit(df=Y_train_df, val_size=12)
    
    # Identify missing combinations of unique_id and ds
    missing_combinations = nf.get_missing_future(futr_df=Y_test_df)
    
    # Create the future DataFrame
    futr_df = pd.concat([Y_test_df, missing_combinations]).drop_duplicates().reset_index(drop=True)
    
    # Predict using the future DataFrame
    forecasts = nf.predict(futr_df=futr_df)
    
    # Remove duplicate timestamps from Y_test_df
    Y_test_df = Y_test_df[~Y_test_df['ds'].duplicated()]
    
    # Add predictions to Y_test_df
    Y_test_df.set_index('ds', inplace=True)
    forecasts.set_index('ds', inplace=True)
    Y_test_df['NBEATS'] = forecasts['NBEATS']
    Y_test_df['MLP'] = forecasts['MLP']
    
    # Calculate MAE
    # Remove the calculation for TimeLLM if it's not available
    if 'TimeLLM' in Y_test_df.columns:
        mae_timellm = mae(Y_test_df['y'], Y_test_df['TimeLLM'])
    else:
        mae_timellm = None
    mae_nbeats = mae(Y_test_df['y'], Y_test_df['NBEATS'])
    mae_mlp = mae(Y_test_df['y'], Y_test_df['MLP'])

    data = {'Time-LLM': [mae_timellm], 
            'N-BEATS': [mae_nbeats],
            'MLP': [mae_mlp]}

    metrics_df = pd.DataFrame(data=data)
    metrics_df.index = ['mae']

    print(metrics_df)
    print(metrics_df.style.highlight_min(color='lightgreen', axis=1))
    
    # Plot the predictions
    pred_dict = {
        'median': forecasts['NBEATS'] + forecasts['MLP'],
        'samples': np.array([forecasts['NBEATS'], forecasts['MLP']])
    }
    plot_preds(train['y'], test['y'], pred_dict, model_name='NBEATS + MLP')