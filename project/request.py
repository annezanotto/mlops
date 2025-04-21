import requests
import pandas as pd
import json

# 🔹 Exemplo com um dado do X_test (ajuste os valores conforme suas features)
sample_input = {
    "Time": -1.203,
    "Amount": 0.153,
    "V1": -0.913,
    "V2": 0.234,
    "V3": 1.002,
    "V4": -0.324,
    "V5": 0.431,
    "V6": -1.204,
    "V7": 0.312,
    "V8": 0.204,
    "V9": -0.543,
    "V10": 0.101,
    "V11": 1.333,
    "V12": -0.123,
    "V13": 0.999,
    "V14": -0.432,
    "V15": 0.645,
    "V16": 0.321,
    "V17": -0.983,
    "V18": 0.345,
    "V19": -0.101,
    "V20": 0.189,
    "V21": 0.050,
    "V22": -0.222,
    "V23": 0.003,
    "V24": 0.066,
    "V25": -0.123,
    "V26": 0.014,
    "V27": 0.079,
    "V28": -0.011
}

# 🔸 Enviar como lista de dicionários (pode enviar vários registros também)
payload = {
    "inputs": [sample_input]
}

# 🔸 Endpoint do MLServer exposto via Docker
url = "http://localhost:9000/invocations"

# 🔸 Enviar requisição
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)


if response.status_code == 200:
    result = response.json()
    pred = result['predictions'][0] if 'predictions' in result else result[0]
    label = "FRAUD" if pred == 1 else "NOT FRAUD"
    print(f"✅ Predição: {pred} ({label})")
else:
    print("❌ Erro na requisição:", response.status_code)
    print(response.text)
