"""
Testes — Camada de Transformação (Silver)
=========================================
Verifica as regras de limpeza e validação da camada Silver.

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transformation.silver_transform import (
    remove_duplicates,
    fix_data_types,
    handle_nulls,
    validate_ranges,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset de exemplo para os testes."""
    return pd.DataFrame({
        "person_age": [25, 30, 150, 25, 30],         # 150 = inválido
        "person_income": [50_000, 80_000, 30_000, 50_000, 80_000],
        "person_home_ownership": ["rent", "own", "mortgage", "rent", "own"],
        "person_emp_length": [3.0, None, 5.0, 3.0, None],  # nulos para testar
        "loan_intent": ["personal", "education", "medical", "personal", "education"],
        "loan_grade": ["A", "B", "C", "A", "B"],
        "loan_amnt": [5_000, 10_000, 15_000, 5_000, 10_000],
        "loan_int_rate": [7.5, None, 12.0, 7.5, None],     # nulos para testar
        "loan_status": [0, 1, 0, 0, 1],
        "loan_percent_income": [0.10, 0.12, 0.50, 0.10, 0.12],
        "cb_person_default_on_file": ["N", "Y", "N", "N", "Y"],
        "cb_person_cred_hist_length": [3, 7, 5, 3, 7],
    })


class TestRemoveDuplicates:
    """Testes para remoção de duplicatas."""

    def test_removes_exact_duplicates(self, sample_df):
        """Verifica que duplicatas exatas são removidas."""
        # Linhas 0 e 3 são idênticas, assim como 1 e 4
        result = remove_duplicates(sample_df)
        assert len(result) < len(sample_df)

    def test_no_duplicates_unchanged(self):
        """Não modifica dataset sem duplicatas."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestFixDataTypes:
    """Testes para correção de tipos."""

    def test_string_columns_uppercased(self, sample_df):
        """Colunas de string devem estar em maiúsculas após limpeza."""
        df_unique = sample_df.drop_duplicates().copy()
        result = fix_data_types(df_unique)
        assert result["person_home_ownership"].str.isupper().all()
        assert result["loan_grade"].str.isupper().all()

    def test_numeric_columns_converted(self, sample_df):
        """Colunas numéricas devem estar com tipos corretos."""
        df_unique = sample_df.drop_duplicates().copy()
        result = fix_data_types(df_unique)
        assert pd.api.types.is_float_dtype(result["person_income"])
        assert result["loan_status"].dtype in [int, np.int64]


class TestValidateRanges:
    """Testes para validação de ranges de negócio."""

    def test_removes_invalid_age(self, sample_df):
        """Registro com idade 150 deve ser removido."""
        df_clean = sample_df.drop_duplicates().copy()
        df_clean["person_age"] = pd.to_numeric(df_clean["person_age"])
        result = validate_ranges(df_clean)
        assert (result["person_age"] <= 100).all()
        assert (result["person_age"] >= 18).all()

    def test_removes_zero_income(self):
        """Registros com renda zero ou negativa devem ser removidos."""
        df = pd.DataFrame({
            "person_age": [25, 30],
            "person_income": [0, 50_000],
            "person_emp_length": [3.0, 5.0],
            "loan_percent_income": [0.1, 0.2],
        })
        result = validate_ranges(df)
        assert len(result) == 1
        assert result["person_income"].iloc[0] == 50_000


class TestHandleNulls:
    """Testes para tratamento de valores nulos."""

    def test_no_nulls_after_treatment(self, sample_df):
        """Dataset não deve ter nulos após o tratamento."""
        df_unique = sample_df.drop_duplicates().copy()
        df_unique["person_age"] = pd.to_numeric(df_unique["person_age"])
        df_unique["person_home_ownership"] = df_unique["person_home_ownership"].str.upper()
        df_unique["loan_grade"] = df_unique["loan_grade"].str.upper()
        result = handle_nulls(df_unique)
        assert result.isnull().sum().sum() == 0
