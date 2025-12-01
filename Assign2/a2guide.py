# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME/blob/main/YOUR_NOTEBOOK_NAME.ipynb)
#
# # Guided notebook for Assignment 2
#
# Here are some example codes to help you start the assignment 2, including the package importing, model initializing, and dataset loading. Note that this is just a rough example, you are free to choose the way to finish the assignment.

# %%
# Preparation

# %%
# Check if GPU is available, if not you may need to change a runtime with GPU
from detoxify import Detoxify
from google.colab import userdata
from huggingface_hub import login
import tqdm
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
!nvidia-smi

# %%
# Install required packages
!pip install transformers accelerate torch datasets huggingface_hub
# More packages depending on your need

# %%
# Import

# %%
# Load HF token and login if necessary
# First you need to upload your API keys in "Secrets"
login(token=userdata.get('HF_TOKEN'))

# %% [markdown]
# ## Setup an LLM

# %%
# Load the model
# this is just an example, make your own choice on the model.
model_name = "microsoft/phi-2"
# Note that different models may have different ways to initialize, adapt by yourself.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0"  # specify the model hosting on GPU
)

# You can also define model from APIs, such as gemini
# import google.generativeai as genai
# genai.configure(api_key="API_KEY")
# model = genai.GenerativeModel("gemini-2.5-flash")

# %%
model

# %% [markdown]
# ## Use the LLM

# %%
# The following codes illustrate how to use the model to generate content based on given inputs.
# NOTE that different model has different way of generating backup_responses, adapt by yourself
prompt = "Hello, how are you?"
formatted_prompt = f"Instruct: {prompt}\nOutput:"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(
    **inputs,
    max_length=200,
    do_sample=True,
    temperature=0.7,
    # add/reduce parameters upon your desire
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

# %%
# Extract only the answer after "Output:"
output_text.split("Output:", 1)[-1].strip()

# %% [markdown]
# ## Load Benchmark dataset
#
# First check the Github and HuggingFace repository of the benchmark dataset you choose, get familiar with the composition of the dateset, and then start loading. The following codes are one way to load the dataset, you are free to use other methods.

# %% [markdown]
# ### DecodingTrust

# %%
# Load the Toxicity subset of the DecodingTrust dataset
# choose your own subsets (at least 3)
dataset = load_dataset("AI-Secure/DecodingTrust", "toxicity")
dataset  # check the structure and info of this dataset

# %%
# Choose one split by its key
toxic_gpt4_split = dataset["toxic.gpt4"]
# check the contents of each feature
toxic_gpt4_split[0]
# TODO: process the data for further usage,
# for example extract the prompts, analyse the scores, etc.

# %% [markdown]
# ### TrustLLM

# %%
# For the TrustLLM dataset, the subsets have different feature names,
# therefore it is needed to specify data_files= to load certain subsets/splits
dataset = load_dataset("TrustLLM/TrustLLM-dataset",
                       data_files="ethics/explicit_moralchoice.json")
dataset

# %% [markdown]
# ## Generate backup_responses
#
# After extracting the prompts from the benchmark dataset, the next step is to put them into the LLM and generate backup_responses.

# %%
# Randomly sample 50 items from the toxic_gpt4_split for debugging
toxic_gpt4_split_sub_random = toxic_gpt4_split.shuffle(
    seed=42).select(range(50))

# Extract only the 'text' from the prompts for the random sample
toxicity_prompts_random = [item['text']
                           for item in toxic_gpt4_split_sub_random['prompt']]
print("Randomly sampled prompts:")
print(toxicity_prompts_random)

# To perform balanced sampling based on 'toxicity',
# we can convert the dataset to a pandas DataFrame first.
toxic_gpt4_df = toxic_gpt4_split.to_pandas()

# Extract the toxicity scores from the 'prompt' column
toxic_gpt4_df['toxicity_score'] = toxic_gpt4_df['prompt'].apply(
    lambda x: x['toxicity'])

# Perform stratified sampling based on the 'toxicity_score'
# Create bins for the toxicity scores for stratification
toxic_gpt4_df['toxicity_bin'] = pd.cut(
    toxic_gpt4_df['toxicity_score'], bins=10, labels=False)

# Sample 5 items from each toxicity bin, if possible
balanced_sample_df = toxic_gpt4_df.groupby('toxicity_bin').apply(
    lambda x: x.sample(n=5, replace=True, random_state=42)).reset_index(drop=True)

# Extract the prompts from the balanced sample
toxicity_prompts_balanced = [item['text']
                             for item in balanced_sample_df['prompt']]
print("\nBalanced sampled prompts based on toxicity:")
print(toxicity_prompts_balanced)

# In your implementation, if the available resources are not enough for looping through all rows,
# you are allowed to use a smaller portion by sampling over rows in a reasonable manner (balanced sampling for example).
# Otherwise it is better to run through all rows in the split to generate enough number of backup_responses for analysis.

# %%
# Loop through the prompts to gather the LLM generated backup_responses
responses = []
for prompt in tqdm.tqdm(toxicity_prompts_balanced):   # use the prompts you extract
    # Use a suitable way of generating response based on the model you choose
    # You may need to change the following codes
    formatted_prompt = f"Instruct: {prompt}\nOutput:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_length=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
        # add/reduce parameters upon your desire
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output_text.split("Output:", 1)[-1].strip()
    responses.append({"prompt": prompt, "response": response})
# Convert into a pandas DataFrame could be helpful for further analysis, but not mandatory.
df = pd.DataFrame(responses)
df

# %% [markdown]
# ## Evaluation
#
# Investigate the generated backup_responses, what can you observe? Explore corresponding references to find out suitable metrics to evaluate the results (for instance toxicity scores, gender bias, and etc.). Eventually in your report the following items must be included:
#
# * specific example prompts and outputs illustrating ethical risks;
# * Visualization of your quantitative results (e.g. bar plot, histogram, boxplot, and so on);
# * Conclusion and discussion on your finding.
#
#

# %%
# For example you can evaluate the toxicity via Detoxify
#!pip install -U detoxify
# Load Detoxify toxicity classifier
tox_model = Detoxify('original')
tox_scores = tox_model.predict(df['response'].tolist())
# Add toxicity scores to the DataFrame
df['toxicity'] = tox_scores['toxicity']
df[['prompt', 'response', 'toxicity']].sort_values(
    by='toxicity', ascending=False)

# After getting the toxicity scores you can further analyse them via plots or statistical measurements
# Remember to include your analysis in your report

# For other subsets, find your own way to analyse the generated repsonses, from your understanding of the dataset's features
