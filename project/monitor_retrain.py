import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

import mlflow
import os
from datetime import datetime
import logging
import subprocess  # Se quiser chamar um script de retraining externo

# Configuração
REFERENCE_DATA_PATH = "data/creditcard.csv"
CURRENT_DATA_PATH = "data/current_data.csv"
OUTPUT_DIR = "reports"
RETRAIN_SCRIPT = "ml.py"  # substitua se tiver outro nome

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)
    logger.info("✅ Dados carregados")
    return reference, current

def monitor_drift(reference, current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Salva o relatório
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"drift_report_{timestamp}.html")
    report.save_html(report_path)
    logger.info(f"📊 Relatório Evidently salvo: {report_path}")

    # Extrai os dados de drift
    result = report.as_dict()
    drift_info = result['metrics'][0]['result']
    share_drifted = drift_info['share_of_drifted_columns']
    n_drifted = drift_info['number_of_drifted_columns']
    total = drift_info['number_of_columns']

    return share_drifted, n_drifted, total, report_path

def trigger_retrain():
    logger.info("🚀 Iniciando re-treinamento...")
    try:
        subprocess.run(["python3", RETRAIN_SCRIPT], check=True)
        logger.info("✅ Re-treinamento concluído")
    except Exception as e:
        logger.error(f"❌ Erro no re-treinamento: {e}")

def main():
    reference, current = load_data()
    share_drifted, n_drifted, total, report_path = monitor_drift(reference, current)

    # Logando no MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    mlflow.set_experiment(experiment_id="223920342058916763")

                            
    with mlflow.start_run(run_name="Drift Monitoring"):
        mlflow.log_metric("share_drifted_columns", share_drifted)
        mlflow.log_metric("drifted_columns", n_drifted)
        mlflow.log_metric("total_columns", total)
        mlflow.log_artifact(report_path)

        logger.info(f"📈 Share de colunas com drift: {share_drifted:.2%} ({n_drifted}/{total})")

        if share_drifted > 0.5:
            logger.warning("⚠️ Drift detectado! Acionando re-treinamento...")
            mlflow.set_tag("drift_detected", True)
            trigger_retrain()
        else:
            logger.info("✅ Nenhum drift significativo detectado.")
            mlflow.set_tag("drift_detected", False)

if __name__ == "__main__":
    main()
