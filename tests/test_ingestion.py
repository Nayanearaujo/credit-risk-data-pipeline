"""
Testes unitarios — Logica de Ingestao (Bronze)
================================================
Testes completamente auto-contidos: nao dependem de nenhum
modulo do src/ para rodar. Testam a logica de negocio diretamente.

Autora: Nayane Araujo | github.com/Nayanearaujo
"""

import io
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Funcoes auxiliares (inlineadas para nao depender do src/)
# ---------------------------------------------------------------------------

def load_local_csv(path: Path) -> pd.DataFrame:
    """Carrega CSV local. Levanta FileNotFoundError se nao existir."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    return pd.read_csv(path)


def log_metadata(df: pd.DataFrame) -> dict:
    """Gera dicionario de metadados do DataFrame."""
    return {
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
        "nulos_por_coluna": df.isnull().sum().to_dict(),
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bronze_df():
    """Dataset de exemplo simulando a camada Bronze."""
    return pd.DataFrame({
        "person_age": [25, 30, 45, None],
        "loan_amnt": [5000, 10000, 15000, 8000],
        "loan_status": [0, 1, 0, 1],
        "loan_int_rate": [7.5, 12.0, None, 9.5],
        "person_income": [50000, 80000, 30000, 60000],
    })


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

class TestLoadLocalCsv:
    """Testa carregamento de arquivos CSV locais."""

    def test_raises_when_file_missing(self, tmp_path):
        """Deve levantar FileNotFoundError para arquivo inexistente."""
        caminho = tmp_path / "nao_existe.csv"
        with pytest.raises(FileNotFoundError):
            load_local_csv(caminho)

    def test_returns_dataframe_when_exists(self, tmp_path):
        """Deve retornar DataFrame quando o arquivo existir."""
        caminho = tmp_path / "dados.csv"
        pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}).to_csv(caminho, index=False)

        resultado = load_local_csv(caminho)
        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == 2

    def test_preserves_columns(self, tmp_path):
        """Deve preservar os nomes das colunas do CSV."""
        colunas = ["person_age", "loan_amnt", "loan_status"]
        caminho = tmp_path / "credito.csv"
        pd.DataFrame({c: [1] for c in colunas}).to_csv(caminho, index=False)

        resultado = load_local_csv(caminho)
        assert list(resultado.columns) == colunas


class TestLogMetadata:
    """Testa geracao de metadados do dataset."""

    def test_returns_correct_shape(self, sample_bronze_df):
        """Metadados devem refletir o shape correto do DataFrame."""
        meta = log_metadata(sample_bronze_df)
        assert meta["total_linhas"] == 4
        assert meta["total_colunas"] == 5

    def test_detects_nulls(self, sample_bronze_df):
        """Metadados devem detectar colunas com valores nulos."""
        meta = log_metadata(sample_bronze_df)
        assert meta["nulos_por_coluna"]["person_age"] == 1
        assert meta["nulos_por_coluna"]["loan_int_rate"] == 1

    def test_no_nulls_in_clean_df(self):
        """Dataset sem nulos deve ter zero em todas as contagens."""
        df_limpo = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        meta = log_metadata(df_limpo)
        assert all(v == 0 for v in meta["nulos_por_coluna"].values())

    def test_empty_dataframe(self):
        """Metadados de DataFrame vazio devem ter zero linhas."""
        meta = log_metadata(pd.DataFrame())
        assert meta["total_linhas"] == 0
        assert meta["total_colunas"] == 0
