import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

class ModelTrainer:
    def __init__(self, X_path, y_path, config_path):
        self.X_path = X_path
        self.y_path = y_path
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"{self.config_path} not found.")
        with open(self.config_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        return config

    def load_data(self):
        if not os.path.exists(self.X_path) or not os.path.exists(self.y_path):
            raise FileNotFoundError(f"Data files {self.X_path} or {self.y_path} not found.")
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path).values.ravel()
        return X, y

    def train(self):
        print("Loading data...")
        X, y = self.load_data()

        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['train']['test_size'], 
            random_state=self.config['train']['random_state']
        )

        print("Building neural network model...")
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation=self.config['train']['activation']))
        model.add(Dense(32, activation=self.config['train']['activation']))
        model.add(Dense(3, activation=self.config['train']['activation_2'])) 

        # Compilar el modelo
        model.compile(optimizer=self.config['train']['optimizer'], 
                      loss=self.config['train']['loss'], 
                      metrics=['accuracy'])

        print("Training the model...")
        history = model.fit(
            X_train, y_train, 
            epochs=self.config['train']['epochs'], 
            batch_size=self.config['train']['batch_size'], 
            verbose=self.config['train']['verbose']
        )

        print("Evaluating the model...")
        y_pred = np.argmax(model.predict(X_test), axis=self.config['train']['axis'])
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy del modelo de red neuronal: {accuracy:.4f}')
        return model

    def save_model(self, model):
        print("Saving model...")
        os.makedirs(os.path.dirname(self.config['reports']['model']), exist_ok=True)
        model.save(self.config['reports']['model'])
        print(f'Model saved to {self.config["reports"]["model"]} successfully.')

    def run(self):
        print("Starting the training process...")
        model = self.train()
        self.save_model(model)
        print("Training and saving process completed.")

if __name__ == "__main__":
    trainer = ModelTrainer(r'data/processed/processed_dataset_X.csv', 
                           r'data/processed/processed_dataset_y.csv', 
                           'params.yaml')
    trainer.run()
