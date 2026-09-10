"""
Machine Learning — Treinamento do Modelo de Risco de Crédito
============================================================
Treina e compara múltiplos modelos de classificação binária para
prever a probabilidade de inadimplência (loan_status = 1).

Modelos treinados:
  1. Regressão Logística (baseline interpretável)
  2. Random Forest (ensemble robusto)
  3. XGBoost (gradient boosting — estado da arte)

Métricas: AUC-ROC, F1-Score, Precision, Recall, KS Statistic

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.models.feature_engineering import prepare_features

setup_logger(log_level="INFO")

MODELS_PATH = Path("src/models")
GOLD_PATH = Path("data/gold")
GOLD_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Definição dos modelos
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Retorna os modelos a serem treinados e comparados.

    Returns:
        Dicionário {nome: instância do modelo sklearn/xgboost}
    """
    return {
        "Regressao_Logistica": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="auc",
            use_label_encoder=False,
        ),
    }


# ---------------------------------------------------------------------------
# Treinamento e avaliação
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Treina um modelo e calcula métricas de avaliação.

    Args:
        model:      Instância do modelo sklearn/xgboost.
        model_name: Nome identificador do modelo.
        X_train, X_test, y_train, y_test: Dados de treino e teste.

    Returns:
        Dicionário com métricas de performance.
    """
    logger.info(f"🤖 Treinando: {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # KS Statistic — mede separação entre distribuições de scores
    from scipy.stats import ks_2samp
    ks_stat, _ = ks_2samp(
        y_proba[y_test == 1],
        y_proba[y_test == 0],
    )

    metrics = {
        "model": model_name,
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ks_statistic": round(ks_stat, 4),
    }

    logger.info(
        f"  ✅ {model_name}: AUC={auc:.4f} | F1={f1:.4f} | "
        f"Precision={precision:.4f} | Recall={recall:.4f} | KS={ks_stat:.4f}"
    )
    return metrics, model


def select_best_model(results: list[dict]) -> str:
    """
    Seleciona o melhor modelo com base no AUC-ROC.

    Args:
        results: Lista de dicionários de métricas.

    Returns:
        Nome do melhor modelo.
    """
    best = max(results, key=lambda x: x["auc_roc"])
    logger.info(f"🏆 Melhor modelo: {best['model']} (AUC={best['auc_roc']:.4f})")
    return best["model"]


def save_model(model, model_name: str) -> Path:
    """
    Serializa e salva o modelo treinado.

    Args:
        model:      Modelo treinado.
        model_name: Nome do arquivo de saída.

    Returns:
        Caminho do arquivo salvo.
    """
    path = MODELS_PATH / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"💾 Modelo salvo: {path}")
    return path


def save_metrics_report(results: list[dict]) -> None:
    """Salva o relatório de comparação de modelos na camada Gold."""
    df_metrics = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    path = GOLD_PATH / "model_comparison.csv"
    df_metrics.to_csv(path, index=False)
    logger.success(f"💾 Comparação de modelos salva: {path}")
    print("\n" + "=" * 60)
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 60)
    print(df_metrics.to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def run_training() -> dict:
    """
    Orquestra o treinamento completo: engenharia de features → treino → avaliação.

    Returns:
        Dicionário com modelos treinados e métricas.
    """
    logger.info("🚀 Iniciando treinamento de modelos de Risco de Crédito")

    # Features
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features()

    # Salva scaler para uso no dashboard
    with open(MODELS_PATH / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Salva nomes das features
    pd.Series(feature_names).to_csv(MODELS_PATH / "feature_names.csv", index=False)

    # Treina e avalia todos os modelos
    models = get_models()
    all_results = []
    trained_models = {}

    for name, model in models.items():
        metrics, trained_model = train_and_evaluate(
            model, name, X_train, X_test, y_train, y_test
        )
        all_results.append(metrics)
        trained_models[name] = trained_model
        save_model(trained_model, name)

    # Seleciona melhor e salva como modelo padrão
    best_name = select_best_model(all_results)
    save_model(trained_models[best_name], "best_model")

    save_metrics_report(all_results)
    logger.success("✅ Treinamento concluído!")

    return {"models": trained_models, "metrics": all_results, "best": best_name}


if __name__ == "__main__":
    result = run_training()
    print(f"\n🏆 Melhor modelo: {result['best']}")
