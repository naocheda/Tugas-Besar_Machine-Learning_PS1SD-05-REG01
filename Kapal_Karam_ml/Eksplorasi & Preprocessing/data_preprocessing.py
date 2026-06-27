import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Mengisi missing values dengan nilai median untuk kolom numerik
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Mengubah data kategorikal (teks) menjadi numerik
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Memisahkan fitur (X) dan target klasifikasi (y)
    # Asumsi kolom target bernama 'Diagnosis' (sesuaikan jika berbeda di dataset)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    # Membagi data menjadi data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
