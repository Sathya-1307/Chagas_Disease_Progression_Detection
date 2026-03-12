import pandas as pd

def load_and_preprocess(file_path):
    
    # Load dataset
    data = pd.read_csv(file_path)

    # Remove ID column
    if "exam_id" in data.columns:
        data = data.drop("exam_id", axis=1)

    # Fill missing values
    data = data.fillna(data.mean())

    # Convert boolean to integer
    if "is_male" in data.columns:
        data["is_male"] = data["is_male"].astype(int)

    if "chagas" in data.columns:
        data["chagas"] = data["chagas"].astype(int)

    # Remove duplicates
    data = data.drop_duplicates()

    # Split features and target
    X = data.drop("chagas", axis=1)
    y = data["chagas"]

    return X, y