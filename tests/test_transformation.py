"""
Testes unitarios — Logica de Transformacao (Silver)
====================================================
Testes completamente auto-contidos: nao dependem de nenhum
modulo do src/ para rodar. Testam a logica de negocio diretamente.

As funcoes aqui espelham exatamente o que o silver_transform.py faz,
permitindo validar as regras de negocio em isolamento.

Autora: Nayane Araujo | github.com/Nayanearaujo
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Logica de Silver (inlineada para isolamento total)
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas exatas."""
    return df.drop_duplicates()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos de dados e padroniza strings em maiusculo."""
    df = df.copy()
    int_cols = ["person_age", "loan_amnt", "cb_person_cred_hist_length"]
    float_cols = ["person_income", "person_emp_length", "loan_int_rate", "loan_percent_income"]
    str_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    if "loan_status" in df.columns:
        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").astype("Int64")

    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Trata nulos: mediana para numericos, remove restantes."""
    df = df.copy()

    if "person_emp_length" in df.columns:
        median_val = df["person_emp_length"].median()
        df["person_emp_length"] = df["person_emp_length"].fillna(median_val)

    if "loan_int_rate" in df.columns and "loan_grade" in df.columns:
        df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
            lambda x: x.fillna(x.median())
        )

    return df.dropna()


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros com valores fora dos ranges de negocio."""
    df = df.copy()

    if "person_age" in df.columns:
        df["person_age"] = pd.to_numeric(df["person_age"], errors="coerce")
        df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]

    if "person_income" in df.columns:
        df["person_income"] = pd.to_numeric(df["person_income"], errors="coerce")
        df = df[df["person_income"] > 0]

    if "person_emp_length" in df.columns:
        df = df[(df["person_emp_length"] >= 0) & (df["person_emp_length"] <= 60)]

    if "loan_percent_income" in df.columns:
        df = df[(df["loan_percent_income"] >= 0) & (df["loan_percent_income"] <= 1)]

    return df


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Dataset de exemplo completo para os testes."""
    return pd.DataFrame({
        "person_age": [25, 30, 150, 25, 30],
        "person_income": [50_000, 80_000, 30_000, 50_000, 80_000],
        "person_home_ownership": ["rent", "own", "mortgage", "rent", "own"],
        "person_emp_length": [3.0, None, 5.0, 3.0, None],
        "loan_intent": ["personal", "education", "medical", "personal", "education"],
        "loan_grade": ["A", "B", "C", "A", "B"],
        "loan_amnt": [5_000, 10_000, 15_000, 5_000, 10_000],
        "loan_int_rate": [7.5, None, 12.0, 7.5, None],
        "loan_status": [0, 1, 0, 0, 1],
        "loan_percent_income": [0.10, 0.12, 0.50, 0.10, 0.12],
        "cb_person_default_on_file": ["N", "Y", "N", "N", "Y"],
        "cb_person_cred_hist_length": [3, 7, 5, 3, 7],
    })


# ---------------------------------------------------------------------------
# Testes: RemoveDuplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:

    def test_remove_duplicatas_exatas(self, sample_df):
        """Linhas identicas devem ser removidas."""
        resultado = remove_duplicates(sample_df)
        assert len(resultado) < len(sample_df)
        assert len(resultado) == len(sample_df.drop_duplicates())

    def test_dataset_sem_duplicatas_inalterado(self):
        """Dataset sem duplicatas nao deve perder registros."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert len(remove_duplicates(df)) == 3

    def test_todas_duplicatas_removidas(self):
        """Duplicatas multiplas: apenas uma deve permanecer."""
        df = pd.DataFrame({"col": ["a", "a", "a", "b"]})
        assert len(remove_duplicates(df)) == 2


# ---------------------------------------------------------------------------
# Testes: FixDataTypes
# ---------------------------------------------------------------------------

class TestFixDataTypes:

    def test_strings_em_maiusculo(self, sample_df):
        """Colunas de string devem ser convertidas para maiusculo."""
        df_unico = sample_df.drop_duplicates().copy()
        resultado = fix_data_types(df_unico)
        assert resultado["person_home_ownership"].str.isupper().all()
        assert resultado["loan_grade"].str.isupper().all()

    def test_renda_como_float(self, sample_df):
        """person_income deve ser tipo numerico float."""
        resultado = fix_data_types(sample_df.drop_duplicates().copy())
        assert pd.api.types.is_numeric_dtype(resultado["person_income"])

    def test_target_como_inteiro(self, sample_df):
        """loan_status deve ser inteiro (0 ou 1)."""
        resultado = fix_data_types(sample_df.drop_duplicates().copy())
        assert pd.api.types.is_integer_dtype(resultado["loan_status"])


# ---------------------------------------------------------------------------
# Testes: ValidateRanges
# ---------------------------------------------------------------------------

class TestValidateRanges:

    def test_remove_idade_invalida(self, sample_df):
        """Idades acima de 100 ou abaixo de 18 devem ser removidas."""
        resultado = validate_ranges(sample_df.drop_duplicates().copy())
        assert (resultado["person_age"] <= 100).all()
        assert (resultado["person_age"] >= 18).all()

    def test_remove_renda_zero(self):
        """Renda zero deve ser removida."""
        df = pd.DataFrame({
            "person_age": [25, 30],
            "person_income": [0, 50_000],
            "person_emp_length": [3.0, 5.0],
            "loan_percent_income": [0.1, 0.2],
        })
        resultado = validate_ranges(df)
        assert len(resultado) == 1
        assert resultado["person_income"].iloc[0] == 50_000

    def test_mantém_registros_validos(self):
        """Registros dentro dos ranges devem ser preservados."""
        df = pd.DataFrame({
            "person_age": [20, 35, 50],
            "person_income": [30_000, 60_000, 90_000],
        })
        resultado = validate_ranges(df)
        assert len(resultado) == 3

    def test_comprometimento_acima_de_um_removido(self):
        """Comprometimento de renda acima de 100% deve ser removido."""
        df = pd.DataFrame({
            "person_emp_length": [3.0, 3.0],
            "loan_percent_income": [0.5, 1.5],
        })
        resultado = validate_ranges(df)
        assert len(resultado) == 1


# ---------------------------------------------------------------------------
# Testes: HandleNulls
# ---------------------------------------------------------------------------

class TestHandleNulls:

    def test_sem_nulos_apos_tratamento(self, sample_df):
        """Nenhum nulo deve restar apos o tratamento."""
        df = fix_data_types(sample_df.drop_duplicates().copy())
        resultado = handle_nulls(df)
        assert resultado.isnull().sum().sum() == 0

    def test_mediana_preenche_emp_length(self):
        """Nulos em person_emp_length devem ser preenchidos com mediana."""
        df = pd.DataFrame({
            "person_emp_length": [2.0, 4.0, None, 6.0],
            "loan_grade": ["A", "A", "A", "A"],
            "loan_int_rate": [7.0, 8.0, 9.0, 10.0],
        })
        resultado = handle_nulls(df)
        assert resultado["person_emp_length"].isnull().sum() == 0

    def test_tamanho_reduzido_com_nulos(self, sample_df):
        """Dataset apos tratamento deve ter menos ou igual registros."""
        df_unico = sample_df.drop_duplicates().copy()
        df_typed = fix_data_types(df_unico)
        resultado = handle_nulls(df_typed)
        assert len(resultado) <= len(df_unico)
