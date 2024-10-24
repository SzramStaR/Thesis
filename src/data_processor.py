import pandas as pd
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt

def fetch_data(model_name):
    api_token = "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"
    url = f"https://huggingface.co/api/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    response = requests.get(url,headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.json(),response.status_code
        raise Exception(f"Failed to fetch the data for model {model_name}")
    
    
    
def preprocess_data(file_name):
    wanted_cols = ['fullname','#Params (B)','Type','Average ⬆️','IFEval Raw','IFEval','BBH Raw','BBH','MATH Lvl 5 Raw','MATH Lvl 5','GPQA Raw','GPQA','MUSR Raw','MUSR','MMLU-PRO Raw','MMLU-PRO'] #Name + evals to take + model type + average + params
    df_csv = pd.read_csv(file_name,usecols=wanted_cols)
    model_df = df_csv[(df_csv['fullname'] == leaderboard_data['modelId'])]
    if not model_df.empty:
        model_type = model_df['Type'].iloc[0]  # Extract the single value
        same_model_types_df = df_csv[df_csv['Type'] == model_type]
        return(same_model_types_df)
    else:
        return("Model not found in the CSV file.")
    
def model_card(preprocessed):
    pass

def preprocess_top_10_average(chosen_model,preprocessed): #based on average
    top_10_models = preprocessed.sort_values(by='Average ⬆️',ascending=False).head(10)
    #if chosen model not in the top 10
    if chosen_model not in top_10_models['fullname'].values:
        chosen_model_data = preprocessed[preprocessed['fullname'] == chosen_model]
        top_10_models = pd.concat([top_10_models,chosen_model_data])
    melted_df = top_10_models.melt(id_vars=['fullname'], value_vars=['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO'], var_name='Benchmark', value_name='Score')
    return melted_df #for performance, i guess
    # return top_10_models
    
def preprocess_best_in_class(chosen_model,preprocessed,benchmark): #based benchmark performance
    best_in_class =  preprocessed.sort_values(by=benchmark,ascending=False).head(10)
    #if chosen model not in the top 10
    if chosen_model not in best_in_class['fullname'].values:
        chosen_model_data = preprocessed[preprocessed['fullname'] == chosen_model]
        best_in_class = pd.concat([best_in_class,chosen_model_data])
    melted_df = best_in_class.melt(id_vars=['fullname'], value_vars=['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO'], var_name='Benchmark', value_name='Score') #for performance?
    return melted_df
    return best_in_class


def preprocess_smilar_performance(chosen_model,preprocessed,threshold=1): #based on average, threshold +-5 
    chosen_model_df = preprocessed[preprocessed['fullname'] == chosen_model]
    if chosen_model_df.empty:
        raise ValueError(f"Chosen model: {chosen_model} not found in the dataset")
    chosen_model_avg = chosen_model_df['Average ⬆️'].iloc[0]
    #apply threshold
    lower = chosen_model_avg - threshold
    upper = chosen_model_avg + threshold
    similar_perform = preprocessed[(preprocessed['Average ⬆️']>=lower) & (preprocessed['Average ⬆️']<=upper)]
    # Separate better and worse models
    better_models = similar_perform[similar_perform['Average ⬆️'] > chosen_model_avg].sort_values(by='Average ⬆️', ascending=False).head(5)
    worse_models = similar_perform[similar_perform['Average ⬆️'] < chosen_model_avg].sort_values(by='Average ⬆️', ascending=False).head(5)
    combined_models = pd.concat([better_models,worse_models])
    if chosen_model not in combined_models['fullname'].values:   
        combined_models = pd.concat([better_models, chosen_model_df, worse_models])
    
    melted_df = similar_perform.melt(id_vars=['fullname'], value_vars=['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO'], var_name='Benchmark', value_name='Score') #for performance?
    # return melted_df  #should be sorted?
    return combined_models
    

def top_10_bar_plot(top_10_data):
    #Create plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=top_10_data, x='Benchmark', y='Score', hue='fullname')
    plt.title('Benchmark Scores for Top 10 Models and Chosen Model')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()
    
model = "openai-community/gpt2"
leaderboard_data = fetch_data(model_name=model)
# pretty_json = json.dumps(leaderboard_data,indent=4)
# print(pretty_json)
# print(leaderboard_data)
# df = pd.read_parquet("hf://datasets/open-llm-leaderboard/contents/data/train-00000-of-00001.parquet") Leaderboard data to csv
# df.to_csv("openllm.csv",index=False)
# print(df)

file_name = "openllm.csv"
preprocessed = preprocess_data(file_name=file_name)
# print(preprocessed)
print(preprocess_smilar_performance(chosen_model=model,preprocessed=preprocessed))




