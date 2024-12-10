import pandas as pd

data_set = pd.read_csv('../data/amazon_alexa.tsv', sep='\t')
data_set.head()

data=data_set[['verified_reviews','feedback']]
data.columns = ['review','label']

data.head()

# count occs of each label
label_counts = data.value_counts('label')


# get the nb of rows to drop from the majority class
rows_to_drop = label_counts.max()- label_counts.min()

#drop rows from the majority class
if rows_to_drop>0:
    data_majority = data[data['label']==1]
    data_balanced = data.drop(data_majority.sample(rows_to_drop).index)
else:
    data_balanced = data.copy()

#check the new class balance
print(data_balanced['label'].value_counts())


import re

def clean_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return ""  # or return text if you want to leave non-string values unchanged
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove single chars
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Lowercase text
    text = text.lower()

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    # Trim leading and trailing space
    text = text.strip()

    return text


data_balanced.head()


# extract review column as a list
reviews = data_balanced['review'].tolist()

#clean the text in the list
cleaned_reviews = [clean_text(review) for review in reviews]

# add cleaned reviews as a new column to the dataframe
data_balanced['clean_review'] = cleaned_reviews



total_rows = len(data_balanced)
test_size = int(total_rows*0.95)

# randomly sample train_size rows for the training set
test_set = data_balanced.sample(test_size)

#get the remaining rows for the test set
train_set = data_balanced.drop(test_set.index)

import pathlib
import textwrap

import google.generativeai as genai 


from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))




from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY is None:
    print("GOOGLE_API_KEY environment variable not set.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel('gemini-pro')

response= model.generate_content('What is the meaning of life?')
to_markdown(response.text)


test_set_sample = test_set.sample(20)

test_set_sample['pred_label']=''

test_set_sample

json_data = test_set_sample[['clean_review','pred_label']].to_json(orient='records')
print(json_data)

prompt = f"""
You are an expert linguier, who is good at classifying customer review sentinence into Positive/Negative. 
Help me classify customer reviews into: Positive(label=1), and Negative(label=0).
Customer reviews are provided between three back ticks.
In your output, only return the Json code back as output - which is provided between three backticks. 
Your task is to update predicted labels under 'pred_label' in the Json code. 
Don't make any changes to Json code format, please. 


```
{json_data}
""" 
print(prompt)

response = model.generate_content(prompt)

print(response.text)

import json

#clean the data by stripping the backticks
json_data = response.text.strip("`")

try:
    data_ = json.loads(json_data)
    df_sample = pd.DataFrame(data_)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    df_sample = pd.DataFrame()

df_sample

test_set_sample['pred_label'] = df_sample['pred_label'].values
test_set_sample

#plotting confusion matrix for prediction
from sklearn.metrics import confusion_matrix

y_true = test_set_sample["label"]
y_pred = test_set_sample['pred_label']

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))



import google.generativeai as genai

def get_completion(prompt, model="gemini-pro"):    
    # Create the model
    model = genai.GenerativeModel(model)
    
    # Generate the response
    response = model.generate_content(prompt)
    
    return response.text

#test
prompt = "Why is the sky blue?"

response = get_completion(prompt)
print(response)

test_set.shape

test_set_total = test_set.sample(100)

test_set_total['pred_label'] = ''

test_set_total

batches = []
batch_size = 25
for i in range(0, len(test_set_total), batch_size):
    batches.append(test_set_total[i : i + batch_size]) # append batches instead of assigning

import time
import json

def gemini_completion_function(batch, current_batch, total_batch):
    """Function works in three steps:
    # Step-1: Convert the DataFrame to JSON using the to_json() method.
    # Step-2: Preparing the Gemini Prompt
    # Step-3: Calling Gemini API
    """
    print(f"Now processing batch#: {current_batch+1} of {total_batch}")
    
    # Convert DataFrame to JSON
    json_data = batch[['clean_review','pred_label']].to_json(orient='records')
    
    prompt = f"""You are an expert linguist, who is good at classifying customer review sentiments into Positive/Negative labels.
    Help me classify customer reviews into: Positive(label=1), and Negative(label=0).
    Customer reviews are provided between three backticks below.
    In your output, only return the Json code back as output - which is provided between three backticks.
    Your task is to update predicted labels under 'pred_label' in the Json code.
    Don't make any changes to Json code format, please.
    Error handling instruction: In case a Customer Review violates API policy, please assign it default sentiment as Negative (label=0).
    
    ```
    {json_data}
    ```
    """
    
    try:
        # Generate content using Gemini
        response = model.generate_content(prompt)
        
        # Add delay to manage rate limits
        time.sleep(5)
        
        # Return the text response
        return response.text
    
    except Exception as e:
        print(f"Error in Gemini API call for batch {current_batch+1}: {e}")
        # Return original JSON data if API call fails
        return json_data

batch_count = len(batches)
responses = []

for i in range(0,batch_count):
  responses.append(gemini_completion_function(batches[i],i,batch_count))


df_total = pd.DataFrame()  # empty df

for response in responses:
    if isinstance(response, str):
        json_data = response.strip("`")
    else:
        json_data = response.text.strip("`")
    data = json.loads(json_data)
    
    df_temp = pd.DataFrame(data)
    
    df_total = pd.concat([df_total, df_temp], ignore_index=True)

display(df_total)


test_set_total['pred_label'] = df_total['pred_label'].values
test_set_total


from sklearn.metrics import confusion_matrix, accuracy_score

y_true = test_set_total["label"]
y_pred = test_set_total["pred_label"]


print(confusion_matrix(y_true, y_pred))
print(f"\nAccuracy: {accuracy_score(y_true, y_pred)}")

