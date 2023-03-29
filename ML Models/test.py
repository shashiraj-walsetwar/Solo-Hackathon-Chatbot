import openai
import pandas as pd

# Load the dataset from the CSV file
df = pd.read_csv('Database\InputData.csv')

# Define the GPT-3 prompt
prompt = "The following is a conversation between a user and a chatbot.\n\nUser: "

# Loop through each row in the dataset and append to the prompt
for index, row in df.iterrows():
    prompt += str(row["question"]) + "\n\nBot: " + str(row["answer"]) + "\n\nUser: "
    # print(f'Question: {row["question"]}')
    # print(f'Answer:   {row["answer"]}')

print(f'Prompt: {prompt}')