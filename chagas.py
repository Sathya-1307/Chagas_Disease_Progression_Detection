from preprocessing import load_and_preprocess
from stacking_model import train_lstm_model

def main():

    X, y = load_and_preprocess("signals_features.csv")

    print("Dataset loaded successfully")
    print("Features shape:", X.shape)

    model = train_lstm_model(X, y)

if __name__ == "__main__":
    main()