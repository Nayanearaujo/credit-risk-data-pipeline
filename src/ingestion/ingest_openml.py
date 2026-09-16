"""
Bronze Layer - Ingestão oficial de dados via OpenML
===================================================
Baixa e armazena os dados brutos do Credit Risk Dataset via OpenML (ID: 43454).
Possui fallback automático para mirror no GitHub caso a API OpenML fique indisponível.

Dataset: https://www.openml.org/search?type=data&id=43454
         (32.581 registros, licença CC0)

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

OPENML_DATASET_ID = 43454
OUTPUT_FILE = BRONZE_PATH / "credit_risk_raw.csv"

# Fallback plano B caso OpenML API esteja inacessível
FALLBACK_URL = (
    "https://raw.githubusercontent.com/dsrscientist/"
    "dataset1/master/credit_risk_dataset.csv"
)


# ---------------------------------------------------------------------------
# Funções de Ingestão
# ---------------------------------------------------------------------------


def download_openml(
    dataset_id: int = OPENML_DATASET_ID, output_path: Path = OUTPUT_FILE
) -> pd.DataFrame:
    """
    Baixa o dataset diretamente da OpenML (fonte oficial citável).

    Args:
        dataset_id: ID do dataset na OpenML (43454).
        output_path: Caminho de destino para o arquivo CSV bruto.

    Returns:
        DataFrame com os dados brutos.
    """
    import openml

    logger.info(f"📥 Baixando dataset oficial da OpenML (ID: {dataset_id})...")
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data(dataset_format="dataframe")

    # Salva em CSV na camada Bronze
    df.to_csv(output_path, index=False)
    logger.success(f"✅ Dataset OpenML salvo com sucesso em: {output_path}")
    logger.info(f"📊 Shape do dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


def download_fallback(
    url: str = FALLBACK_URL, output_path: Path = OUTPUT_FILE
) -> pd.DataFrame:
    """
    Plano B: Baixa o dataset do mirror no GitHub caso a OpenML falhe.
    """
    logger.warning(f"⚠️  Iniciando download via fallback (mirror GitHub): {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.success(f"✅ Dataset de fallback salvo em: {output_path}")

    df = pd.read_csv(output_path)
    logger.info(
        f"📊 Shape do dataset de fallback: {df.shape[0]:,} linhas × {df.shape[1]} colunas"
    )
    return df


def load_local_dataset(path: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Carrega o dataset localmente caso já tenha sido baixado.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute primeiro: python src/ingestion/ingest_openml.py"
        )
    df = pd.read_csv(path)
    logger.info(f"📂 Dataset local carregado: {path} ({df.shape[0]:,} linhas)")
    return df


def log_bronze_metadata(df: pd.DataFrame) -> dict:
    """
    Registra e retorna metadados da camada Bronze para auditoria.
    """
    logger.info("=" * 60)
    logger.info("📦 BRONZE LAYER - Metadados do Dataset")
    logger.info("=" * 60)
    logger.info(f"Total de registros: {len(df):,}")
    logger.info(f"Total de colunas:   {len(df.columns)}")
    nulls = df.isnull().sum()
    for col, count in nulls[nulls > 0].items():
        pct = count / len(df) * 100
        logger.warning(f"  ⚠️  {col}: {count:,} nulos ({pct:.1f}%)")
    logger.info("=" * 60)

    return {
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
        "nulos_por_coluna": nulls.to_dict(),
    }


def run_ingestion() -> pd.DataFrame:
    """
    Orquestra a ingestão da camada Bronze com OpenML e fallback automático.
    """
    logger.info("🚀 Iniciando ingestão - Camada Bronze")

    if OUTPUT_FILE.exists():
        logger.info(f"🔁 Dataset já existe localmente em {OUTPUT_FILE}. Carregando...")
        df = load_local_dataset(OUTPUT_FILE)
    else:
        try:
            df = download_openml(OPENML_DATASET_ID, OUTPUT_FILE)
        except Exception as e:
            logger.error(f"❌ Falha no download via OpenML API: {e}")
            logger.info("🔄 Acionando plano B (fallback mirror)...")
            df = download_fallback(FALLBACK_URL, OUTPUT_FILE)

    log_bronze_metadata(df)
    logger.success("✅ Ingestão Bronze concluída!")
    return df


if __name__ == "__main__":
    df_raw = run_ingestion()
    print("\nPrimeiras linhas do dataset bruto:")
    print(df_raw.head())
