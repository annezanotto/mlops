# mlops
Trabalho da disciplina MLOPS

# Projeto de Detecção de Fraude com MLflow e Docker

Este projeto utiliza MLflow e Docker para treinar, servir e monitorar modelos de detecção de fraude.

## Requisitos

- [MLflow](https://mlflow.org/) instalado
- [Docker](https://www.docker.com/) instalado

---

## Como rodar o projeto

### 1. Iniciar o MLflow

Execute o script:

```bash
./mlflow.sh
```

Esse script irá iniciar o MLflow Server na sua máquina na **porta 5050**.

---

### 2. Iniciar a API de previsão

Abra uma **nova janela do terminal** e execute:

```bash
./serv_api.sh
```

Esse script realizará os seguintes passos:

1. Verificará se existe algum container rodando na **porta 9000**.
   - Se houver, ele **derruba o container atual**.
2. Verificará se existe a imagem `fraud_detection` no seu Docker.
   - Se **não existir**, o script **treina um novo modelo**.
   - Se **existir**, ele **executa um container com o último modelo disponível**.

---

### 3. Agendar re-treinamento automático

Execute:

```bash
./create_cron.sh
```

Esse script irá criar uma **cron job** que executa o arquivo `monitor_retrain` a cada **24 horas**.

- Caso seja detectado **drift** no modelo, um novo modelo será treinado e executado automaticamente.

---

## Observações

- Verifique se os scripts possuem permissão de execução:
  ```bash
  chmod +x mlflow.sh serv_api.sh create_cron.sh
  ```

- O monitoramento de drift pode ser feito com ferramentas como **Evidently AI**.
