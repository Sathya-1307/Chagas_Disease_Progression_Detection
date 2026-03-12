# Step 1: Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Load dataset
data = pd.read_csv("chagas_clustering_dataset.csv")

# Step 3: Preprocess numeric columns
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Step 4: Features and target
target_col = 'chagas'
features = data.select_dtypes(include='number').drop(columns=[target_col, 'exam_id'])  # drop exam_id and target
target = data[target_col]

# Step 5: Scale features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Step 6: Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)

# Step 7: Reshape for LSTM [samples, timesteps, features]
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

print("Train shape:", X_train_lstm.shape)
print("Test shape:", X_test_lstm.shape)

# Step 8: Build Stacked LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 9: Train the model
history = model.fit(
    X_train_lstm, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_lstm, y_test),
    verbose=1
)

# Step 10: Predictions
y_pred_prob = model.predict(X_test_lstm)
y_pred = (y_pred_prob > 0.5).astype(int)

# Step 11: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)

# Confusion matrix to calculate sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity (Recall for positive class): {sensitivity:.4f}")
print(f"Specificity (Recall for negative class): {specificity:.4f}")
