import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import openpyxl

df = pd.read_csv('creditcard.csv')

df=df.drop_duplicates()

scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

#Balanceamento

X = df.drop(columns=['Class'])
y = df['Class']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Verificar o balanceamento
print("Distribuição antes do SMOTE:", y.value_counts())
print("Distribuição depois do SMOTE:", pd.Series(y_res).value_counts())

df_resampled = pd.concat([X_res, y_res], axis=1)  
df_resampled.head()
df_resampled.to_csv('dados_normalizados_balanceados.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

classifier = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier(    
            
        criterion='gini',         # Função de divisão
        max_depth=10,             # Profundidade máxima da árvore
        min_samples_split=2,      # Número mínimo de amostras para dividir um nó
        min_samples_leaf=1,       # Número mínimo de amostras em uma folha
        max_features=None,        # Número máximo de features para dividir um nó
        random_state=42           # Garante reprodutibilidade
    )
}

for name, clf in classifier.items():
    print(f"\n=========={name}===========")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n Accuaracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test, y_pred)}")
    print(f"\n Recall: {recall_score(y_test, y_pred)}")
    print(f"\n F1 Score: {f1_score(y_test, y_pred)}")

    from sklearn.model_selection import cross_val_score

# Validação cruzada para o modelo de Regressão Logística
log_reg_scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)  # cv=5 significa 5-fold cross-validation
print(f"Acurácia média da Regressão Logística: {log_reg_scores.mean()}")

# Validação cruzada para o modelo de Árvore de Decisão
dt_scores = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=5)
print(f"Acurácia média da Árvore de Decisão: {dt_scores.mean()}")