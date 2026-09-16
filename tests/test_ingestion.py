"""
Testes unitários - Lógica de Ingestão (Bronze)
================================================
Testa as funções reais de src.ingestion.ingest_openml.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.ingest_openml import (
    load_local_dataset,
    log_bronze_metadata,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bronze_df():
    """Dataset de exemplo simulando a camada Bronze."""
    return pd.DataFrame(
        {
            "person_age": [25, 30, 45, None],
            "loan_amnt": [5000, 10000, 15000, 8000],
            "loan_status": [0, 1, 0, 1],
            "loan_int_rate": [7.5, 12.0, None, 9.5],
            "person_income": [50000, 80000, 30000, 60000],
        }
    )


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestLoadLocalDataset:
    """Testa carregamento de arquivos CSV locais da camada Bronze."""

    def test_raises_when_file_missing(self, tmp_path):
        """Deve levantar FileNotFoundError para arquivo inexistente."""
        caminho = tmp_path / "nao_existe.csv"
        with pytest.raises(FileNotFoundError):
            load_local_dataset(caminho)

    def test_returns_dataframe_when_exists(self, tmp_path):
        """Deve retornar DataFrame quando o arquivo existir."""
        caminho = tmp_path / "dados.csv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(caminho, index=False)

        resultado = load_local_dataset(caminho)
        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == 2

    def test_preserves_columns(self, tmp_path):
        """Deve preservar os nomes das colunas do CSV."""
        colunas = ["person_age", "loan_amnt", "loan_status"]
        caminho = tmp_path / "credito.csv"
        pd.DataFrame({c: [1] for c in colunas}).to_csv(caminho, index=False)

        resultado = load_local_dataset(caminho)
        assert list(resultado.columns) == colunas


class TestLogBronzeMetadata:
    """Testa geração de metadados do dataset da camada Bronze."""

    def test_returns_correct_shape(self, sample_bronze_df):
        """Metadados devem refletir o shape correto do DataFrame."""
        meta = log_bronze_metadata(sample_bronze_df)
        assert meta["total_linhas"] == 4
        assert meta["total_colunas"] == 5

    def test_detects_nulls(self, sample_bronze_df):
        """Metadados devem detectar colunas com valores nulos."""
        meta = log_bronze_metadata(sample_bronze_df)
        assert meta["nulos_por_coluna"]["person_age"] == 1
        assert meta["nulos_por_coluna"]["loan_int_rate"] == 1

    def test_no_nulls_in_clean_df(self):
        """Dataset sem nulos deve ter zero em todas as contagens."""
        df_limpo = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        meta = log_bronze_metadata(df_limpo)
        assert all(v == 0 for v in meta["nulos_por_coluna"].values())

    def test_empty_dataframe(self):
        """Metadados de DataFrame vazio devem ter zero linhas."""
        meta = log_bronze_metadata(pd.DataFrame())
        assert meta["total_linhas"] == 0
        assert meta["total_colunas"] == 0
