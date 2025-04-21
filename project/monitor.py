import pandas as pd


from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metric_preset import TargetDriftPreset
from evidently.legacy.metric_preset import ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping


import os
from datetime import datetime
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
REFERENCE_DATA_PATH = "creditcard.csv"
CURRENT_DATA_PATH = "data/predictions_data.csv"
OUTPUT_DIR = "reports"

def load_data():
    try:
        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
        current_data = pd.read_csv(CURRENT_DATA_PATH)
        logger.info("✅ Dados carregados com sucesso.")
        return reference_data, current_data
    except Exception as e:
        logger.error(f"❌ Erro ao carregar os dados: {e}")
        raise

def generate_monitoring_report(reference_data, current_data):
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ClassificationPreset()
    ])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=ColumnMapping(
            target="Class",
            prediction="Class"
        )
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"monitoring_report_{timestamp}.html")

    report.save_html(output_path)
    logger.info(f"📊 Relatório Evidently salvo em: {output_path}")

def main():
    reference_data, current_data = load_data()
    generate_monitoring_report(reference_data, current_data)

if __name__ == "__main__":
    main()
