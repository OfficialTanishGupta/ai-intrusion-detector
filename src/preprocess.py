import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    le_protocol = LabelEncoder()
    le_flag = LabelEncoder()
    
    df['protocol_type'] = le_protocol.fit_transform(df['protocol_type'])
    df['flag'] = le_flag.fit_transform(df['flag'])

    le_label = LabelEncoder()
    df['label'] = le_label.fit_transform(df['label'])

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
