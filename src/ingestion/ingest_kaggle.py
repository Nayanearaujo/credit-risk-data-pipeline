"""
Bronze Layer — Ingestão de dados de risco de crédito
=====================================================
Baixa e armazena os dados brutos do Kaggle Credit Risk Dataset.
Nenhum tratamento é feito aqui — apenas ingestão fiel à origem.

Arquitetura Medallion:
  [Fonte] ──► [Bronze] ──► Silver ──► Gold

Dataset: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
         (Kaggle Credit Risk Dataset — 32.581 registros)

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# Adiciona o root do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger

setup_logger(log_level="INFO")

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

BRONZE_PATH = Path("data/bronze")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)

# URL pública com espelho do dataset (sem precisar de login no Kaggle)
# Fonte: dataset público no GitHub (cópia autorizada do Kaggle)
DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/credit_risk_dataset.csv"
)
OUTPUT_FILE = BRONZE_PATH / "credit_risk_raw.csv"


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def download_dataset(url: str, output_path: Path) -> pd.DataFrame:
    """
    Baixa o dataset de risco de crédito e salva na camada Bronze.

    Args:
        url:         URL pública do CSV.
        output_path: Caminho de destino para o arquivo bruto.

    Returns:
        DataFrame com os dados brutos.

    Raises:
        requests.HTTPError: Se a URL não estiver acessível.
    """
    logger.info(f"📥 Baixando dataset de: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.success(f"✅ Dataset salvo em: {output_path}")

    df = pd.read_csv(output_path)
    logger.info(f"📊 Shape do dataset bruto: {df.shape}")
    logger.info(f"📋 Colunas: {list(df.columns)}")

    return df


def load_local_dataset(path: Path) -> pd.DataFrame:
    """
    Carrega um dataset local (quando o download não está disponível).
    Use esta função para reprocessar dados já baixados.

    Args:
        path: Caminho para o CSV local.

    Returns:
        DataFrame com os dados brutos.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute primeiro: python src/ingestion/ingest_kaggle.py"
        )

    df = pd.read_csv(path)
    logger.info(f"📂 Dataset local carregado: {path}")
    logger.info(f"📊 Shape: {df.shape}")
    return df


def log_bronze_metadata(df: pd.DataFrame) -> None:
    """
    Registra metadados da camada Bronze para rastreabilidade.

    Args:
        df: DataFrame com os dados brutos.
    """
    logger.info("=" * 60)
    logger.info("📦 BRONZE LAYER — Metadados do Dataset")
    logger.info("=" * 60)
    logger.info(f"Total de registros: {len(df):,}")
    logger.info(f"Total de colunas:   {len(df.columns)}")
    logger.info(f"Uso de memória:     {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    logger.info("Valores nulos por coluna:")
    nulls = df.isnull().sum()
    for col, count in nulls[nulls > 0].items():
        pct = count / len(df) * 100
        logger.warning(f"  ⚠️  {col}: {count:,} nulos ({pct:.1f}%)")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def run_ingestion() -> pd.DataFrame:
    """
    Orquestra a ingestão completa da camada Bronze.

    Returns:
        DataFrame com os dados brutos armazenados.
    """
    logger.info("🚀 Iniciando ingestão — Camada Bronze")

    # Verifica se já existe localmente
    if OUTPUT_FILE.exists():
        logger.info("🔁 Dataset já existe localmente. Carregando...")
        df = load_local_dataset(OUTPUT_FILE)
    else:
        df = download_dataset(DATASET_URL, OUTPUT_FILE)

    log_bronze_metadata(df)
    logger.success("✅ Ingestão Bronze concluída!")
    return df


if __name__ == "__main__":
    df_raw = run_ingestion()
    print(f"\nPrimeiras linhas do dataset bruto:")
    print(df_raw.head())
