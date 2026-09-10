"""
Dashboard Streamlit — Plataforma de Risco de Crédito
=====================================================
Interface interativa para visualização de insights e
predição de inadimplência em tempo real.

Como executar:
    streamlit run src/dashboard/app.py
    (ou: make run-dashboard)

Autora: Nayane Araújo | github.com/Nayanearaujo
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Risco de Crédito | Nayane Araújo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para visual profissional
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #3498db;
        color: white;
    }
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 10px; }
    h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold"
MODELS_PATH = ROOT / "src" / "models"
DOCS_PATH = ROOT / "docs"


# ---------------------------------------------------------------------------
# Funções de carregamento (com cache para performance)
# ---------------------------------------------------------------------------

@st.cache_data
def load_gold_data(table: str) -> pd.DataFrame:
    """Carrega uma tabela da camada Gold."""
    path = GOLD_PATH / f"{table}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    """Carrega o melhor modelo treinado."""
    path = MODELS_PATH / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Carrega o scaler treinado."""
    path = MODELS_PATH / "scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Renderiza a barra lateral com filtros globais."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=80)
        st.title("💳 Risco de Crédito")
        st.markdown("**Nayane Araújo**")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github)](https://github.com/Nayanearaujo)")
        st.divider()

        page = st.radio(
            "Navegação",
            ["📊 Visão Geral", "🔍 Análise por Segmento", "🤖 Simulador de Crédito"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Pipeline: Bronze → Silver → Gold")
        st.caption("Arquitetura Medallion | Scikit-Learn | XGBoost")
    return page


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

def page_overview():
    """Página 1: Visão geral dos dados e KPIs."""
    st.title("📊 Visão Geral — Portfolio de Crédito")
    st.markdown("Análise end-to-end de **32.000+ empréstimos** com Arquitetura Medallion.")

    fact = load_gold_data("fact_loans")
    if fact.empty:
        st.warning("⚠️ Execute o pipeline primeiro: `make run-pipeline && make train`")
        st.code("make run-pipeline\nmake train", language="bash")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total = len(fact)
    inadimplentes = fact["loan_status"].sum()
    taxa_inadimplencia = inadimplentes / total
    volume_total = fact["loan_amnt"].sum()

    col1.metric("Total de Empréstimos", f"{total:,}")
    col2.metric("Inadimplentes", f"{inadimplentes:,}", delta=f"{taxa_inadimplencia:.1%}", delta_color="inverse")
    col3.metric("Taxa de Inadimplência", f"{taxa_inadimplencia:.1%}")
    col4.metric("Volume Total (R\$)", f"R\$ {volume_total:,.0f}")

    st.divider()

    # Gráficos
    col_left, col_right = st.columns(2)

    with col_left:
        agg_grade = load_gold_data("agg_default_by_grade")
        if not agg_grade.empty:
            fig = px.bar(
                agg_grade,
                x="loan_grade", y="default_rate",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Taxa de Inadimplência por Grade de Risco",
                labels={"loan_grade": "Grade", "default_rate": "Taxa de Inadimplência"},
                text_auto=".1%",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        agg_intent = load_gold_data("agg_default_by_intent")
        if not agg_intent.empty:
            fig = px.bar(
                agg_intent.sort_values("default_rate", ascending=True),
                x="default_rate", y="loan_intent",
                orientation="h",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Taxa de Inadimplência por Finalidade do Empréstimo",
                labels={"loan_intent": "Finalidade", "default_rate": "Taxa"},
                text_auto=".1%",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Distribuição por faixa etária
    agg_age = load_gold_data("agg_default_by_age")
    if not agg_age.empty:
        fig = px.bar(
            agg_age,
            x="age_group", y="default_rate",
            color="default_rate",
            color_continuous_scale="Blues",
            title="Taxa de Inadimplência por Faixa Etária",
            labels={"age_group": "Faixa Etária", "default_rate": "Taxa de Inadimplência"},
            text_auto=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_segment_analysis():
    """Página 2: Análise detalhada por segmento."""
    st.title("🔍 Análise por Segmento")

    fact = load_gold_data("fact_loans")
    if fact.empty:
        st.warning("Execute o pipeline primeiro.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de Valores de Empréstimo")
        fig = px.histogram(
            fact, x="loan_amnt",
            color="loan_status_label",
            color_discrete_map={"Adimplente": "#2ecc71", "Inadimplente": "#e74c3c"},
            nbins=50,
            barmode="overlay",
            opacity=0.7,
            title="Distribuição do Valor Solicitado",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Taxa de Juros vs. Risco")
        fig = px.box(
            fact, x="loan_status_label", y="loan_int_rate",
            color="loan_status_label",
            color_discrete_map={"Adimplente": "#2ecc71", "Inadimplente": "#e74c3c"},
            title="Taxa de Juros por Status do Empréstimo",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Modelo comparison
    model_comp = load_gold_data("model_comparison")
    if not model_comp.empty:
        st.subheader("📈 Comparação de Modelos de ML")
        st.dataframe(
            model_comp.style.highlight_max(subset=["auc_roc", "f1_score"], color="#d4edda"),
            use_container_width=True,
        )

    # Imagens geradas pelo evaluate_model.py
    roc_path = DOCS_PATH / "roc_curve.png"
    cm_path = DOCS_PATH / "confusion_matrix.png"
    fi_path = DOCS_PATH / "feature_importance.png"

    if roc_path.exists() and cm_path.exists():
        col3, col4 = st.columns(2)
        with col3:
            st.image(str(roc_path), caption="Curva ROC — Melhor Modelo")
        with col4:
            st.image(str(cm_path), caption="Matriz de Confusão")

    if fi_path.exists():
        st.image(str(fi_path), caption="Importância das Features", use_container_width=True)


def page_credit_simulator():
    """Página 3: Simulador de concessão de crédito."""
    st.title("🤖 Simulador de Crédito")
    st.markdown(
        "Preencha os dados abaixo para calcular a **probabilidade de inadimplência** "
        "usando o modelo de Machine Learning treinado."
    )

    model = load_model()
    if model is None:
        st.warning("⚠️ Treine o modelo primeiro: `make train`")
        return

    # Formulário de entrada
    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Dados Pessoais")
            age = st.slider("Idade", 18, 80, 30)
            income = st.number_input("Renda Anual (R\$)", min_value=10_000, max_value=500_000,
                                      value=50_000, step=5_000)
            emp_length = st.slider("Anos de Emprego", 0, 40, 5)
            home = st.selectbox("Tipo de Moradia",
                                 ["RENT", "OWN", "MORTGAGE", "OTHER"])

        with col2:
            st.subheader("💰 Dados do Empréstimo")
            loan_amnt = st.number_input("Valor Solicitado (R\$)", min_value=500, max_value=35_000,
                                         value=10_000, step=500)
            loan_int_rate = st.slider("Taxa de Juros Anual (%)", 5.0, 25.0, 12.0, step=0.5)
            loan_intent = st.selectbox("Finalidade", [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])
            loan_grade = st.selectbox("Grade de Risco", ["A", "B", "C", "D", "E", "F", "G"])

        with col3:
            st.subheader("📋 Histórico de Crédito")
            loan_percent_income = loan_amnt / income
            cred_hist = st.slider("Anos de Histórico de Crédito", 0, 30, 5)
            prior_default = st.radio("Histórico de Inadimplência?", ["Não", "Sim"])

            st.metric("% da Renda Comprometida",
                      f"{loan_percent_income:.1%}",
                      help="Parcela estimada / Renda mensal")

        submitted = st.form_submit_button("🔮 Calcular Risco", use_container_width=True)

    if submitted:
        st.divider()

        # Prepara features (simplificado para o simulador)
        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        has_prior = 1 if prior_default == "Sim" else 0

        # Feature vector simplificado (deve corresponder ao treinamento)
        features = np.array([[
            age, income, emp_length, loan_amnt, loan_int_rate,
            loan_percent_income, cred_hist,
            grade_map.get(loan_grade, 3),
            has_prior,
            loan_amnt / income,
            income / (emp_length + 1),
            loan_amnt * (1 + loan_int_rate / 100),
        ]])

        try:
            prob = model.predict_proba(features[:, :model.n_features_in_])[0][1]

            # Resultado visual
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                color = "#e74c3c" if prob > 0.5 else "#2ecc71"
                label = "⚠️ ALTO RISCO" if prob > 0.5 else "✅ BAIXO RISCO"
                st.markdown(f"""
                <div style='background:{color}20; border:2px solid {color};
                            border-radius:12px; padding:20px; text-align:center;'>
                    <h2 style='color:{color}'>{label}</h2>
                    <h1 style='color:{color}'>{prob:.1%}</h1>
                    <p>Probabilidade de Inadimplência</p>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                    },
                    title={"text": "Score de Risco"},
                ))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Simulador simplificado — treine o modelo completo para precisão máxima. ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()

    if page == "📊 Visão Geral":
        page_overview()
    elif page == "🔍 Análise por Segmento":
        page_segment_analysis()
    elif page == "🤖 Simulador de Crédito":
        page_credit_simulator()

    # Footer
    st.divider()
    st.markdown(
        "**Pipeline de Risco de Crédito** · Nayane Araújo · "
        "[GitHub](https://github.com/Nayanearaujo) · "
        "Arquitetura Medallion (Bronze → Silver → Gold)",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
