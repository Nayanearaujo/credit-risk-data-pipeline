"""
Machine Learning - Engenharia de Features
==========================================
Prepara as features para o modelo de Risco de Crédito.

Técnicas utilizadas:
  - Encoding de variáveis categóricas (One-Hot e Ordinal)
  - Criação de features derivadas (razões, interações)
  - Balanceamento de classes com SMOTE (target desbalanceado)
  - Escalonamento numérico com StandardScaler

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger

setup_logger(log_level="INFO")

SILVER_PATH = Path("data/silver")
INPUT_FILE = SILVER_PATH / "credit_risk_clean.csv"


# Mapeamento ordinal da grade de risco (A = mais seguro, G = mais arriscado)
GRADE_ORDER = [["A", "B", "C", "D", "E", "F", "G"]]


def load_silver() -> pd.DataFrame:
    """Carrega o dataset da camada Silver."""
    df = pd.read_csv(INPUT_FILE)
    # Remove colunas de metadados geradas pelo pipeline
    meta_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=meta_cols, errors="ignore")
    logger.info(f"📂 Silver carregada: {df.shape}")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas que agregam valor preditivo.

    Features criadas:
        - debt_to_income_ratio: parcela / renda (risco de comprometimento)
        - income_per_year_employed: renda / tempo de emprego (estabilidade)
        - loan_to_income_ratio: valor do empréstimo / renda anual
        - has_prior_default: variável binária de inadimplência histórica
    """
    df = df.copy()

    # Razão dívida/renda (DTI) - quanto da renda mensal vai para a parcela
    df["debt_to_income_ratio"] = df["loan_percent_income"]

    # Estabilidade financeira - renda por ano de emprego
    df["income_per_year_employed"] = df["person_income"] / (df["person_emp_length"] + 1)

    # Custo total estimado do empréstimo
    df["estimated_total_cost"] = df["loan_amnt"] * (1 + df["loan_int_rate"] / 100)

    # Histórico de inadimplência
    df["has_prior_default"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    logger.info(
        "✅ Features derivadas criadas: debt_to_income_ratio, income_per_year_employed, "
        "estimated_total_cost, has_prior_default"
    )
    return df


INTENT_CATEGORIES = [
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
]
HOME_CATEGORIES = ["MORTGAGE", "OTHER", "OWN", "RENT"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas.

    - loan_grade:          OrdinalEncoder (A < B < ... < G)
    - loan_intent:         One-Hot Encoding
    - person_home_ownership: One-Hot Encoding
    """
    df = df.copy()

    # Ordinal: grade de risco
    enc_grade = OrdinalEncoder(
        categories=GRADE_ORDER,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    df["loan_grade_encoded"] = enc_grade.fit_transform(df[["loan_grade"]])

    # One-Hot: propósito do empréstimo (Categorical garante consistência para 1 linha ou N linhas)
    df["loan_intent"] = pd.Categorical(df["loan_intent"], categories=INTENT_CATEGORIES)
    intent_dummies = pd.get_dummies(df["loan_intent"], prefix="intent", drop_first=True)
    df = pd.concat([df, intent_dummies], axis=1)

    # One-Hot: tipo de moradia (Categorical garante consistência para 1 linha ou N linhas)
    df["person_home_ownership"] = pd.Categorical(
        df["person_home_ownership"], categories=HOME_CATEGORIES
    )
    home_dummies = pd.get_dummies(
        df["person_home_ownership"], prefix="home", drop_first=True
    )
    df = pd.concat([df, home_dummies], axis=1)

    logger.info(
        "✅ Encoding concluído: OrdinalEncoder (grade) + One-Hot (intent, home)"
    )
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona as features finais para o modelo e separa o target.

    Returns:
        X: DataFrame de features
        y: Série com o target (loan_status)
    """
    # Remove colunas que não devem entrar no modelo
    drop_cols = [
        "loan_status",
        "loan_grade",
        "loan_intent",
        "person_home_ownership",
        "cb_person_default_on_file",
        "loan_status_label",
    ]
    # Remove também colunas de metadata e dummies originais
    drop_cols += [c for c in df.columns if c.startswith("_")]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["loan_status"].copy()

    # Remove colunas com todos NaN
    X = X.dropna(axis=1, how="all")

    logger.info(f"✅ Features selecionadas: {len(feature_cols)} variáveis")
    logger.info(f"📊 Distribuição do target: {y.value_counts().to_dict()}")
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42,
) -> tuple:
    """
    Divide em treino/teste, aplica SMOTE e escalonamento.

    Args:
        X:            Features.
        y:            Target.
        test_size:    Proporção do conjunto de teste (padrão: 20%).
        apply_smote:  Se True, aplica SMOTE para balancear as classes.
        random_state: Seed para reprodutibilidade.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Balanceamento com SMOTE (apenas no treino!)
    if apply_smote:
        if SMOTE is None:
            raise ImportError(
                "imbalanced-learn nao esta instalado. Instale com: pip install imbalanced-learn"
            )
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            f"⚖️  SMOTE aplicado - Treino: {len(y_train):,} registros "
            f"({y_train.value_counts().to_dict()})"
        )

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"📊 Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_features() -> tuple:
    """
    Pipeline completo de preparação de features.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    logger.info("🚀 Iniciando engenharia de features...")

    df = load_silver()
    df = create_derived_features(df)
    df = encode_categoricals(df)
    X, y = select_features(df)
    feature_names = list(X.columns)

    # Preenche NaN restantes com mediana (seguro para tree-based models)
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    logger.success(
        f"✅ Features prontas! {len(feature_names)} variáveis para o modelo."
    )
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = prepare_features()
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        print(f"  - {f}")
