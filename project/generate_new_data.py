import pandas as pd
import numpy as np
import os

# Caminho do dataset original
DATASET_PATH = "data/creditcard.csv"
OUTPUT_PATH = "data/current_data.csv"

# Cria diretório se não existir
os.makedirs("data", exist_ok=True)

# Carregar dados
df = pd.read_csv(DATASET_PATH)

# Seleciona 5000 amostras aleatórias mantendo o balanceamento de classes
fraud_df = df[df["Class"] == 1]
non_fraud_df = df[df["Class"] == 0]

fraud_sample = fraud_df.sample(n=250, random_state=42)
non_fraud_sample = non_fraud_df.sample(n=4750, random_state=42)

current_df = pd.concat([fraud_sample, non_fraud_sample])
current_df = current_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Aplica um pequeno ruído nas colunas numéricas
feature_cols = [col for col in df.columns if col not in ["Class"]]
noise = np.random.normal(0, 0.01, current_df[feature_cols].shape)
current_df[feature_cols] += noise

# Salvar como current_data.csv
current_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ current_data.csv salvo em: {OUTPUT_PATH}")
