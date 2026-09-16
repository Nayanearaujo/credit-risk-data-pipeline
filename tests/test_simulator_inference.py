import pickle
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.feature_engineering import create_derived_features, encode_categoricals


def test_simulator_low_risk_profile():
    with open(ROOT / "src/models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "src/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    lines = [
        line.strip()
        for line in open(ROOT / "src/models/feature_names.csv")
        if line.strip()
    ]
    if lines and lines[0] == "0":
        lines = lines[1:]

    raw_input = {
        "person_age": [30.0],
        "person_income": [50000.0],
        "person_home_ownership": ["RENT"],
        "person_emp_length": [5.0],
        "loan_intent": ["PERSONAL"],
        "loan_grade": ["A"],
        "loan_amnt": [10000.0],
        "loan_int_rate": [12.0],
        "loan_percent_income": [10000.0 / 50000.0],
        "cb_person_default_on_file": ["N"],
        "cb_person_cred_hist_length": [5.0],
    }
    df_input = pd.DataFrame(raw_input)
    df_derived = create_derived_features(df_input)
    df_encoded = encode_categoricals(df_derived)
    X_input = df_encoded.reindex(columns=lines, fill_value=0)
    X_eval = scaler.transform(X_input)
    prob = float(model.predict_proba(X_eval)[0][1])

    # Perfil seguro (Grade A, sem default previo, 20% de comprometimento)
    # Deve ser aprovado com baixo risco (< 15%)
    assert prob < 0.15, f"Esperava prob < 0.15 mas obteve {prob}"
