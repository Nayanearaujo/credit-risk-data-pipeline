"""
Machine Learning - Treinamento do Modelo de Risco de Crédito
============================================================
Treina e compara múltiplos modelos de classificação binária para
prever a probabilidade de inadimplência (loan_status = 1).

Recursos implementados:
  1. Validação Cruzada Estratificada (StratifiedKFold, 5 folds)
  2. Tuning de hiperparâmetros para XGBoost via RandomizedSearchCV
  3. Registro dos melhores hiperparâmetros em data/gold/best_params.json
  4. Métricas completas: CV AUC, Test AUC, F1, Precision, Recall, KS, Brier Score
  5. Calibração de probabilidades e curva de confiabilidade

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.calibration import evaluate_calibration
from src.models.feature_engineering import prepare_features
from src.utils.logger import setup_logger

setup_logger(log_level="INFO")

MODELS_PATH = Path("src/models")
GOLD_PATH = Path("data/gold")
GOLD_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Definição dos modelos base
# ---------------------------------------------------------------------------


def get_base_models() -> dict:
    """
    Retorna os modelos base a serem treinados e avaliados com validação cruzada.
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
    }


# ---------------------------------------------------------------------------
# Tuning de Hiperparâmetros
# ---------------------------------------------------------------------------


def tune_xgboost(
    X_train: np.ndarray, y_train: pd.Series, random_state: int = 42
) -> tuple[XGBClassifier, dict]:
    """
    Otimiza hiperparâmetros do XGBoost com RandomizedSearchCV.
    Espaço de busca focado para manter tempo de execução razoável (~15-25 segundos).

    Returns:
        best_estimator: Modelo XGBoost com melhores hiperparâmetros.
        best_params: Dicionário de hiperparâmetros vencedores.
    """
    logger.info("🎯 Executando tuning de hiperparâmetros para XGBoost...")

    param_dist = {
        "n_estimators": [150, 200, 250],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85],
    }

    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_estimator = search.best_estimator_

    # Salva melhores hiperparâmetros em JSON na camada Gold
    params_path = GOLD_PATH / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    logger.success(f"💾 Melhores parâmetros salvos em {params_path}: {best_params}")
    return best_estimator, best_params


# ---------------------------------------------------------------------------
# Treinamento e avaliação com CV
# ---------------------------------------------------------------------------


def evaluate_with_cv(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, object]:
    """
    Avalia o modelo via StratifiedKFold (5 folds) e no conjunto de teste independente.
    """
    logger.info(f"🤖 Avaliando: {model_name} com 5-Fold Stratified CV...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
    )
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # Treina no conjunto de treino completo e avalia no teste
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba)

    ks_stat, _ = ks_2samp(y_proba[y_test == 1], y_proba[y_test == 0])

    metrics = {
        "model": model_name,
        "cv_auc_mean": round(cv_mean, 4),
        "cv_auc_std": round(cv_std, 4),
        "test_auc_roc": round(test_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "ks_statistic": round(ks_stat, 4),
        "brier_score": round(brier, 4),
    }

    logger.info(
        f"  ✅ {model_name}: CV_AUC={cv_mean:.4f}±{cv_std:.4f} | "
        f"Test_AUC={test_auc:.4f} | F1={f1:.4f} | Brier={brier:.4f}"
    )
    return metrics, model


def save_model(model, model_name: str) -> Path:
    """Serializa e salva o modelo treinado."""
    path = MODELS_PATH / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"💾 Modelo salvo: {path}")
    return path


def save_metrics_report(results: list[dict]) -> None:
    """Salva o relatório comparativo na camada Gold."""
    df_metrics = pd.DataFrame(results).sort_values("test_auc_roc", ascending=False)
    path = GOLD_PATH / "model_comparison.csv"
    df_metrics.to_csv(path, index=False)
    logger.success(f"💾 Comparação de modelos salva: {path}")
    print("\n" + "=" * 75)
    print("📊 COMPARAÇÃO DE MODELOS COM VALIDAÇÃO CRUZADA E CALIBRAÇÃO")
    print("=" * 75)
    print(df_metrics.to_string(index=False))
    print("=" * 75)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def run_training() -> dict:
    """
    Orquestra o pipeline completo de Machine Learning:
    Features → CV → Tuning XGBoost → Avaliação → Calibração → Registro.
    """
    logger.info("🚀 Iniciando suite de Machine Learning de Risco de Crédito")

    # Features
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features()

    # Salva scaler para inferência
    with open(MODELS_PATH / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Salva nomes ordenados das features (sem header para leitura limpa)
    pd.Series(feature_names).to_csv(
        MODELS_PATH / "feature_names.csv", index=False, header=False
    )

    all_results = []
    trained_models = {}

    # 1. Modelos base
    base_models = get_base_models()
    for name, model in base_models.items():
        metrics, trained_model = evaluate_with_cv(
            model, name, X_train, X_test, y_train, y_test
        )
        all_results.append(metrics)
        trained_models[name] = trained_model
        save_model(trained_model, name)

    # 2. XGBoost com Hyperparameter Tuning
    tuned_xgb, best_params = tune_xgboost(X_train, y_train)
    xgb_metrics, trained_xgb = evaluate_with_cv(
        tuned_xgb, "XGBoost_Tuned", X_train, X_test, y_train, y_test
    )
    all_results.append(xgb_metrics)
    trained_models["XGBoost_Tuned"] = trained_xgb
    save_model(trained_xgb, "XGBoost")

    # Salva campeão como best_model
    best_name = max(all_results, key=lambda x: x["test_auc_roc"])["model"]
    save_model(trained_models[best_name], "best_model")

    # 3. Calibração do modelo campeão
    evaluate_calibration(
        trained_models[best_name],
        X_test,
        y_test,
        model_name="XGBoost_Tuned",
        save_plot=True,
    )

    save_metrics_report(all_results)
    logger.success("✅ Suite de modelagem concluída!")

    return {"models": trained_models, "metrics": all_results, "best": best_name}


if __name__ == "__main__":
    run_training()
