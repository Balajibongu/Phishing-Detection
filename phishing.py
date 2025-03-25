import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, jaccard_score, f1_score, log_loss,
                             precision_score, recall_score, confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel(r'C:\Users\Balaji\OneDrive\Desktop\Project\dataset_phishing.xlsx', engine='openpyxl')

# Data preprocessing
mapping = {'legitimate': 0, 'phishing': 1}
df['status'] = df['status'].map(mapping)
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['status']
threshold = 0.1
relevant_features = target_corr[abs(target_corr) > threshold].index.tolist()
X = df[relevant_features].drop('status', axis=1)
y = df['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train and evaluate models
def train_and_evaluate(model, name):
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Jaccard Index": jaccard_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "Log Loss": log_loss(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions)
    }
    
    for metric, value in metrics.items():
        print(f"{name} {metric}: {value:.4f}")
    
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()
    
    print(f"{name} Classification Report:\n{classification_report(y_test, predictions)}")
    
    return model

# Train models
rf = train_and_evaluate(RandomForestClassifier(), "Random Forest")
svm = train_and_evaluate(SVC(), "SVM")
xgb = train_and_evaluate(XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                       subsample=0.8, colsample_bytree=0.8, min_child_weight=1, 
                                       gamma=0, objective='binary:logistic'), "XGBoost")
tree = train_and_evaluate(DecisionTreeClassifier(), "Decision Tree")
nn = train_and_evaluate(MLPClassifier(), "Neural Network")

# Deep Neural Network Model
dnn = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = dnn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate DNN
dnn_loss, dnn_accuracy = dnn.evaluate(X_test_scaled, y_test)
print(f"Deep Neural Network Accuracy: {dnn_accuracy * 100:.2f}%")

# Save models
pickle.dump(rf, open("rf_model.pkl", "wb"))
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(xgb, open("xgb_model.pkl", "wb"))
pickle.dump(tree, open("tree_model.pkl", "wb"))
pickle.dump(nn, open("nn_model.pkl", "wb"))
with open("Phishing_model.pkl", "wb") as file:
    pickle.dump(rf, file)

# Save the deep learning model separately
model.save("DeepNeuralNetwork_model.h5")
pickle.dump(scaler, open("scaler.pkl", "wb"))
