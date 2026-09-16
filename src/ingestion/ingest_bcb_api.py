"""
Bronze Layer - Ingestão da Taxa Selic via API do Banco Central do Brasil
=========================================================================
Coleta dados macroeconômicos públicos para enriquecer a análise de crédito.
A Taxa Selic influencia diretamente o risco de inadimplência.

API: https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados
     (Série 432 = Taxa Selic Over - diária, desde 1986)

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger

setup_logger(log_level="INFO")

BRONZE_PATH = Path("data/bronze")
BRONZE_PATH.mkdir(parents=True, exist_ok=True)

BCB_API_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
OUTPUT_FILE = BRONZE_PATH / "selic_rate_raw.csv"


def fetch_selic_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Consulta a API pública do Banco Central para obter a Taxa Selic histórica.

    Args:
        start_date: Data de início no formato DD/MM/YYYY.
                    Default: 5 anos atrás.
        end_date:   Data de fim no formato DD/MM/YYYY.
                    Default: hoje.

    Returns:
        DataFrame com colunas ['data', 'valor'] representando a Selic diária.

    Raises:
        requests.HTTPError: Se a API não responder corretamente.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%d/%m/%Y")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%d/%m/%Y")

    params = {
        "formato": "json",
        "dataInicial": start_date,
        "dataFinal": end_date,
    }

    logger.info(f"🏦 Consultando API BCB - Selic de {start_date} a {end_date}")
    response = requests.get(BCB_API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["data", "selic_diaria"]
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["selic_diaria"] = df["selic_diaria"].astype(float)

    logger.info(f"✅ {len(df):,} registros de Selic obtidos")
    return df


def save_selic_bronze(df: pd.DataFrame) -> None:
    """
    Salva os dados brutos da Selic na camada Bronze.

    Args:
        df: DataFrame com dados da Selic.
    """
    df.to_csv(OUTPUT_FILE, index=False)
    logger.success(f"💾 Selic salva em: {OUTPUT_FILE}")


def run_bcb_ingestion() -> pd.DataFrame:
    """
    Orquestra a ingestão da Taxa Selic.

    Returns:
        DataFrame com dados brutos da Selic.
    """
    logger.info("🚀 Iniciando ingestão BCB - Taxa Selic")

    try:
        df = fetch_selic_data()
        save_selic_bronze(df)
        logger.success("✅ Ingestão BCB concluída!")
        return df
    except requests.RequestException as e:
        logger.error(f"❌ Erro ao consultar API BCB: {e}")
        logger.info("💡 Verifique sua conexão ou use dados em cache se disponíveis.")
        raise


if __name__ == "__main__":
    df_selic = run_bcb_ingestion()
    print("\nÚltimas taxas Selic:")
    print(df_selic.tail())
