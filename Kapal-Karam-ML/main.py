from data_preprocessing import load_and_preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

def main():
    print("Memulai proses Machine Learning...")
    
    # 1. Load & Preprocess
    file_path = 'Alzheimer_Dataset_Details.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    print("Tahap Preprocessing selesai.")
    
    # 2. Training
    models = train_models(X_train, y_train)
    print("Tahap Training Model selesai.")
    
    # 3. Evaluasi
    print("Hasil Evaluasi:\n")
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
