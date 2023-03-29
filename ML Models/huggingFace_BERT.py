import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

# Load the CSV file
df = pd.read_csv('Database\InputData.csv')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Tokenize the question-answer pairs
questions = list(df["question"])
answers = list(df["answer"])
encoded = tokenizer(str(questions), str(answers), padding=True, truncation=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5,
    save_steps=500
)

# Define the training data
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = QADataset(encoded)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train the model
trainer_output = trainer.train()
training_loss = trainer_output.training_loss

# Define the chatbot function
def chatbot(question):
    # Tokenize the question
    inputs = tokenizer.encode_plus(question, None, add_special_tokens=True, return_tensors="pt")
    
    # Generate an answer using the model
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer

# Test the chatbot
while True:
    user_input = input("You: ")
    response = chatbot(user_input)
    print("Bot: " + response)
