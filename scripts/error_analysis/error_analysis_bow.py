import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

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

# Prepare the data
X_train = [item['text'] for item in train_data]
y_train = [item['label'] for item in train_data]
X_test = [item['text'] for item in test_data]
y_test = [item['label'] for item in test_data]

# Create and train the BoW classifier
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

bow_classifier = MultinomialNB()
bow_classifier.fit(X_train_vec, y_train)

# Evaluate the classifier
y_pred = bow_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Bag of Words Classifier Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['run1', 'run2']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('error_analysis/bow_confusion_matrix.png')
plt.close()

# Feature Importance
feature_importance = bow_classifier.feature_log_prob_[1] - bow_classifier.feature_log_prob_[0]
feature_names = vectorizer.get_feature_names_out()

# Top N most important features
N = 20
top_features = sorted(zip(feature_names, feature_importance), key=lambda x: abs(x[1]), reverse=True)[:N]

plt.figure(figsize=(12, 8))
plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
plt.title(f'Top {N} Most Important Features')
plt.xlabel('Log Probability Difference')
plt.ylabel('Features')
plt.savefig('error_analysis/bow_feature_importance.png')
plt.close()

# Sample Explanation
def explain_prediction(classifier, vectorizer, text):
    x = vectorizer.transform([text])
    proba = classifier.predict_proba(x)[0]
    feature_names = vectorizer.get_feature_names_out()
    feature_values = x.toarray()[0]
    
    feature_importance = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    sorted_idx = feature_importance.argsort()
    
    top_positive = [(feature_names[i], feature_importance[i]) for i in sorted_idx[-10:] if feature_values[i] > 0]
    top_negative = [(feature_names[i], feature_importance[i]) for i in sorted_idx[:10] if feature_values[i] > 0]
    
    return proba, top_positive, top_negative

sample_idx = 0
sample_text = X_test[sample_idx]
true_label = y_test[sample_idx]
pred_label = y_pred[sample_idx]

print(f"\nSample Text: {sample_text}")
print(f"True Label: {'run1' if true_label == 0 else 'run2'}")
print(f"Predicted Label: {'run1' if pred_label == 0 else 'run2'}")

proba, top_positive, top_negative = explain_prediction(bow_classifier, vectorizer, sample_text)
print(f"Probability: run1 - {proba[0]:.4f}, run2 - {proba[1]:.4f}")
print("\nTop positive features:")
for feature, importance in top_positive:
    print(f"{feature}: {importance:.4f}")
print("\nTop negative features:")
for feature, importance in top_negative:
    print(f"{feature}: {importance:.4f}")

# Analysis of misclassifications
misclassified = [(X_test[i], y_test[i], y_pred[i]) for i in range(len(y_test)) if y_test[i] != y_pred[i]]

print("\nSample Misclassifications:")
for text, true_label, pred_label in misclassified[:5]:
    print(f"Text: {text}")
    print(f"True Label: {'run1' if true_label == 0 else 'run2'}")
    print(f"Predicted Label: {'run1' if pred_label == 0 else 'run2'}")
    print("---")

# Length analysis
train_lengths = [len(text.split()) for text in X_train]
test_lengths = [len(text.split()) for text in X_test]

plt.figure(figsize=(10, 5))
plt.hist(train_lengths, bins=20, alpha=0.5, label='Train')
plt.hist(test_lengths, bins=20, alpha=0.5, label='Test')
plt.title('Distribution of Text Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('error_analysis/text_lengths.png')
plt.close()

# Class distribution
class_distribution = pd.Series(y_train + y_test).value_counts()
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['run1', 'run2'])
plt.savefig('error_analysis/class_distribution.png')
plt.close()

print("\nExplainability Analysis Complete")