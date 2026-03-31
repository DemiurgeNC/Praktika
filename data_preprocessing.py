import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print(f"Данные загружены. Размер: {self.data.shape}")
        return self.data

    def clean_data(self, min_minutes=900):
        if self.data is None:
            self.load_data()

        df = self.data.copy()

        df = df[df['minutes'] >= min_minutes]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        df = df.dropna(subset=['market_value'])
        df = df[df['market_value'] > 0]

        if 'age' in df.columns:
            df = df[df['age'] >= 16]
            df = df[df['age'] <= 40]

        Q1 = df['market_value'].quantile(0.01)
        Q3 = df['market_value'].quantile(0.99)
        df = df[(df['market_value'] >= Q1) & (df['market_value'] <= Q3)]

        self.data = df.reset_index(drop=True)
        print(f"Данные очищены. Новый размер: {self.data.shape}")
        return self.data

    def compute_per_90(self, df, metric_cols):
        for col in metric_cols:
            if col in df.columns:
                per90_name = f"{col}_90"
                df[per90_name] = df[col] / df['minutes'] * 90
            else:
                print(f"Предупреждение: колонка {col} не найдена, пропускаем.")
        return df
