import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# -------- CREATE TIME SEQUENCES --------
def create_sequences(X, y, time_steps=10):

    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


# -------- TRAIN MODEL --------
def train_lstm_model(X, y):

    # Convert to numpy (FIXES YOUR ERROR)
    X = np.array(X)
    y = np.array(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create sequences
    X, y = create_sequences(X, y, time_steps=10)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    # -------- MODEL --------
    model = Sequential()

    model.add(LSTM(128, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weights
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\nTraining Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    return model