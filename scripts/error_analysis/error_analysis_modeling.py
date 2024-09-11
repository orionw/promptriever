import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load and preprocess the data
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'text': item['query'],
                'label': 0 if item['label'] == 'run1' else 1
            })
    return data

data = load_data('error_analysis/comparison.jsonl')

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create Hugging Face Datasets
train_dataset = Dataset.from_dict({
    'text': [item['text'] for item in train_data],
    'label': [item['label'] for item in train_data]
})

test_dataset = Dataset.from_dict({
    'text': [item['text'] for item in test_data],
    'label': [item['label'] for item in test_data]
})

# BERT Model Training
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# Train the model
trainer.train()

# Evaluate the BERT model
bert_eval_results = trainer.evaluate()
print(f"BERT Model Evaluation Results: {bert_eval_results}")

# Calculate accuracy manually
predictions = trainer.predict(tokenized_test_dataset)
bert_predictions = np.argmax(predictions.predictions, axis=1)
bert_accuracy = accuracy_score(test_dataset['label'], bert_predictions)
print(f"BERT Model Accuracy on Test Set: {bert_accuracy}")

# Bag of Words Classifier
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([item['text'] for item in train_data])
y_train = [item['label'] for item in train_data]

X_test = vectorizer.transform([item['text'] for item in test_data])
y_test = [item['label'] for item in test_data]

bow_classifier = MultinomialNB()
bow_classifier.fit(X_train, y_train)

bow_predictions = bow_classifier.predict(X_test)
bow_accuracy = accuracy_score(y_test, bow_predictions)
print(f"Bag of Words Classifier Accuracy on Test Set: {bow_accuracy}")

# majority class baseline
majority_class = max(set(y_test), key=y_test.count)
majority_class_predictions = [majority_class] * len(y_test)
majority_class_accuracy = accuracy_score(y_test, majority_class_predictions)
print(f"Majority Class Baseline Accuracy on Test Set: {majority_class_accuracy}")