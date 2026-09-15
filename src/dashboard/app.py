"""
Dashboard Streamlit — Plataforma de Risco de Credito
=====================================================
Interface interativa com 3 paginas:
  1. Visao Geral: KPIs e graficos do portfolio
  2. Analise por Segmento: comparacao de modelos e filtros
  3. Simulador de Credito: predicao em tempo real

Como executar:
    streamlit run src/dashboard/app.py

Autora: Nayane Araujo | github.com/Nayanearaujo
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
# Configuracao da pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Risco de Credito | Nayane Araujo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 10px; }
    .stMetric label { font-size: 13px !important; }
    .badge-ok { background:#d4edda; color:#155724; padding:4px 10px; border-radius:20px; }
    .badge-warn { background:#fff3cd; color:#856404; padding:4px 10px; border-radius:20px; }
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
# Carregamento com cache
# ---------------------------------------------------------------------------

@st.cache_data
def load_gold(table: str) -> pd.DataFrame:
    """Carrega uma tabela da camada Gold. Retorna DataFrame vazio se nao existir."""
    path = GOLD_PATH / f"{table}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    """Carrega o modelo treinado. Retorna None se nao existir."""
    path = MODELS_PATH / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Carrega o StandardScaler treinado. Retorna None se nao existir."""
    path = MODELS_PATH / "scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def pipeline_status_banner():
    """Exibe banner de status do pipeline."""
    fact = load_gold("fact_loans")
    model = load_model()

    if fact.empty:
        st.error(
            "**Pipeline nao rodado ainda.**  \n"
            "Execute no terminal: `python run_pipeline.py`"
        )
        st.code("cd /Users/nayane/Desktop/dio-agent-main/credit-risk-data-pipeline\npython run_pipeline.py", language="bash")
        return False

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Gold Layer: {len(fact):,} registros carregados")
    with col2:
        if model:
            st.success("✅ Modelo treinado e pronto")
        else:
            st.warning("⚠️ Modelo nao treinado — rode `python run_pipeline.py`")
    return True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("💳 Risco de Credito")
        st.markdown("**Nayane Araujo**")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github)](https://github.com/Nayanearaujo)")
        st.divider()

        page = st.radio(
            "Navegacao",
            ["📊 Visao Geral", "🔍 Analise por Segmento", "🤖 Simulador de Credito"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Pipeline: Bronze -> Silver -> Gold")
        st.caption("Arquitetura Medallion | Scikit-Learn")
        st.caption("[Ver no GitHub](https://github.com/Nayanearaujo/credit-risk-data-pipeline)")
    return page


# ---------------------------------------------------------------------------
# Pagina 1: Visao Geral
# ---------------------------------------------------------------------------

def page_overview():
    st.title("📊 Visao Geral — Portfolio de Credito")
    st.markdown("Analise end-to-end de **32.000+ emprestimos** com Arquitetura Medallion.")

    if not pipeline_status_banner():
        return

    fact = load_gold("fact_loans")

    # KPIs
    total = len(fact)
    inad = int(fact["loan_status"].sum())
    taxa = inad / total
    volume = fact["loan_amnt"].sum()
    taxa_juros_media = fact["loan_int_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Emprestimos", f"{total:,}")
    c2.metric("Inadimplentes", f"{inad:,}", delta=f"{taxa:.1%}", delta_color="inverse")
    c3.metric("Taxa de Inadimplencia", f"{taxa:.1%}")
    c4.metric("Taxa de Juros Media", f"{taxa_juros_media:.1f}%")

    st.divider()

    col_l, col_r = st.columns(2)

    # Grafico 1: Inadimplencia por Grade
    with col_l:
        agg_grade = load_gold("agg_default_by_grade")
        if not agg_grade.empty:
            fig = px.bar(
                agg_grade,
                x="loan_grade", y="default_rate",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Taxa de Inadimplencia por Grade de Risco",
                labels={"loan_grade": "Grade", "default_rate": "Taxa"},
                text_auto=".1%",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Grafico 2: Inadimplencia por Finalidade
    with col_r:
        agg_intent = load_gold("agg_default_by_intent")
        if not agg_intent.empty:
            fig = px.bar(
                agg_intent.sort_values("default_rate"),
                x="default_rate", y="loan_intent",
                orientation="h",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Taxa de Inadimplencia por Finalidade",
                labels={"loan_intent": "Finalidade", "default_rate": "Taxa"},
                text_auto=".1%",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Grafico 3: Por faixa etaria
    agg_age = load_gold("agg_default_by_age")
    if not agg_age.empty:
        order = ["18-25", "26-35", "36-45", "46-60", "60+"]
        agg_age["age_group"] = pd.Categorical(agg_age["age_group"], categories=order, ordered=True)
        agg_age = agg_age.sort_values("age_group")
        fig = px.bar(
            agg_age,
            x="age_group", y="default_rate",
            color="default_rate",
            color_continuous_scale="Blues_r",
            title="Taxa de Inadimplencia por Faixa Etaria",
            labels={"age_group": "Faixa Etaria", "default_rate": "Taxa"},
            text_auto=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Distribuicao de moradia
    st.subheader("Distribuicao por Tipo de Moradia")
    home_counts = fact["person_home_ownership"].value_counts().reset_index()
    home_counts.columns = ["Tipo", "Quantidade"]
    fig = px.pie(home_counts, values="Quantidade", names="Tipo",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 2: Analise por Segmento
# ---------------------------------------------------------------------------

def page_segment_analysis():
    st.title("🔍 Analise por Segmento")

    if not pipeline_status_banner():
        return

    fact = load_gold("fact_loans")

    # Filtros
    st.subheader("Filtros")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        grades = sorted(fact["loan_grade"].unique())
        sel_grade = st.multiselect("Grade", grades, default=grades)
    with fc2:
        intents = sorted(fact["loan_intent"].unique())
        sel_intent = st.multiselect("Finalidade", intents, default=intents)
    with fc3:
        renda_min, renda_max = int(fact["person_income"].min()), int(fact["person_income"].max())
        sel_renda = st.slider("Renda Anual (R$)", renda_min, renda_max,
                              (renda_min, min(renda_max, 200_000)))

    fact_f = fact[
        fact["loan_grade"].isin(sel_grade) &
        fact["loan_intent"].isin(sel_intent) &
        fact["person_income"].between(sel_renda[0], sel_renda[1])
    ]

    st.caption(f"{len(fact_f):,} registros apos filtros")

    col1, col2 = st.columns(2)

    with col1:
        # Valor do emprestimo por status
        if "loan_status_label" in fact_f.columns:
            status_col = "loan_status_label"
            color_map = {"Adimplente": "#2ecc71", "Inadimplente": "#e74c3c"}
        else:
            fact_f = fact_f.copy()
            fact_f["Status"] = fact_f["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})
            status_col = "Status"
            color_map = {"Adimplente": "#2ecc71", "Inadimplente": "#e74c3c"}

        fig = px.histogram(
            fact_f, x="loan_amnt", color=status_col,
            color_discrete_map=color_map,
            nbins=40, barmode="overlay", opacity=0.75,
            title="Distribuicao do Valor Solicitado",
            labels={"loan_amnt": "Valor (R$)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            fact_f, x=status_col, y="loan_int_rate",
            color=status_col,
            color_discrete_map=color_map,
            title="Taxa de Juros por Status",
            labels={"loan_int_rate": "Taxa (%)", status_col: ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparacao de modelos ML
    model_comp = load_gold("model_comparison")
    if not model_comp.empty:
        st.subheader("Comparacao de Modelos de ML")
        cols_disponiveis = [c for c in ["auc_roc", "f1_score", "precision", "recall"] if c in model_comp.columns]
        if cols_disponiveis:
            st.dataframe(
                model_comp.style.highlight_max(subset=cols_disponiveis, color="#d4edda"),
                use_container_width=True,
            )

    # Scatterplot renda vs valor
    st.subheader("Renda vs Valor do Emprestimo")
    sample = fact_f.sample(min(2000, len(fact_f)), random_state=42)
    fig = px.scatter(
        sample, x="person_income", y="loan_amnt",
        color=status_col, color_discrete_map=color_map,
        opacity=0.5, size_max=4,
        title="Renda vs Valor Solicitado",
        labels={"person_income": "Renda Anual (R$)", "loan_amnt": "Valor do Emprestimo (R$)"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 3: Simulador de Credito
# ---------------------------------------------------------------------------

def page_credit_simulator():
    st.title("🤖 Simulador de Credito")
    st.markdown(
        "Preencha os dados abaixo para calcular a **probabilidade de inadimplencia** "
        "usando o modelo de Machine Learning treinado."
    )

    model = load_model()
    if model is None:
        st.warning("Modelo nao treinado ainda.")
        st.info("Execute no terminal: `python run_pipeline.py`")
        st.code("python run_pipeline.py", language="bash")
        return

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Dados Pessoais")
            age = st.slider("Idade", 18, 80, 30)
            income = st.number_input("Renda Anual (R$)", 10_000, 500_000, 50_000, 5_000)
            emp_length = st.slider("Anos de Emprego", 0, 40, 5)
            home = st.selectbox("Tipo de Moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        with col2:
            st.subheader("Dados do Emprestimo")
            loan_amnt = st.number_input("Valor Solicitado (R$)", 500, 35_000, 10_000, 500)
            loan_int_rate = st.slider("Taxa de Juros Anual (%)", 5.0, 25.0, 12.0, 0.5)
            loan_intent = st.selectbox("Finalidade", [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])
            loan_grade = st.selectbox("Grade de Risco", ["A", "B", "C", "D", "E", "F", "G"])

        with col3:
            st.subheader("Historico de Credito")
            cred_hist = st.slider("Anos de Historico de Credito", 0, 30, 5)
            prior_default = st.radio("Historico de Inadimplencia?", ["Nao", "Sim"])
            loan_pct_income = loan_amnt / income
            st.metric("Comprometimento de Renda", f"{loan_pct_income:.1%}")

        submitted = st.form_submit_button("Calcular Risco", use_container_width=True)

    if submitted:
        st.divider()

        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        has_prior = 1 if prior_default == "Sim" else 0
        custo_total = loan_amnt * (1 + loan_int_rate / 100)
        renda_por_emp = income / (emp_length + 1)

        # Feature vector — usa n_features_in_ do modelo para truncar/preencher
        feat_base = np.array([[
            age, float(income), emp_length, loan_amnt, loan_int_rate,
            loan_pct_income, cred_hist,
            grade_map.get(loan_grade, 2),
            has_prior,
            custo_total,
            renda_por_emp,
            loan_amnt / (cred_hist + 1),
            loan_pct_income,
        ]])

        n_feat = model.n_features_in_
        if feat_base.shape[1] < n_feat:
            feat_base = np.pad(feat_base, ((0, 0), (0, n_feat - feat_base.shape[1])))
        else:
            feat_base = feat_base[:, :n_feat]

        try:
            prob = model.predict_proba(feat_base)[0][1]

            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                if prob < 0.3:
                    cor = "#27ae60"
                    label = "BAIXO RISCO"
                    recomendacao = "Aprovar"
                elif prob < 0.6:
                    cor = "#f39c12"
                    label = "RISCO MODERADO"
                    recomendacao = "Avaliar com atencao"
                else:
                    cor = "#e74c3c"
                    label = "ALTO RISCO"
                    recomendacao = "Negar ou solicitar garantia"

                st.markdown(f"""
                <div style='background:{cor}20; border:2px solid {cor};
                            border-radius:12px; padding:24px; text-align:center;'>
                    <h3 style='color:{cor}; margin:0'>{label}</h3>
                    <h1 style='color:{cor}; margin:8px 0'>{prob:.1%}</h1>
                    <p style='margin:0'>Probabilidade de Inadimplencia</p>
                    <hr style='border-color:{cor}40'>
                    <b>Recomendacao: {recomendacao}</b>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 44}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": cor},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "value": 50
                        },
                    },
                    title={"text": "Score de Risco de Inadimplencia"},
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Resumo das entradas
            st.subheader("Resumo do Pedido")
            dados = {
                "Idade": f"{age} anos",
                "Renda Anual": f"R$ {income:,.0f}",
                "Anos de Emprego": f"{emp_length} anos",
                "Valor Solicitado": f"R$ {loan_amnt:,.0f}",
                "Taxa de Juros": f"{loan_int_rate:.1f}%",
                "Grade de Risco": loan_grade,
                "Comprometimento de Renda": f"{loan_pct_income:.1%}",
                "Historico de Inadimplencia": prior_default,
            }
            df_resumo = pd.DataFrame(list(dados.items()), columns=["Campo", "Valor"])
            st.dataframe(df_resumo, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erro ao calcular: {e}")
            st.info("Certifique-se de que o pipeline foi executado com `python run_pipeline.py`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()

    if page == "📊 Visao Geral":
        page_overview()
    elif page == "🔍 Analise por Segmento":
        page_segment_analysis()
    elif page == "🤖 Simulador de Credito":
        page_credit_simulator()

    st.divider()
    st.caption(
        "Pipeline de Risco de Credito - Nayane Araujo | "
        "[GitHub](https://github.com/Nayanearaujo/credit-risk-data-pipeline) | "
        "Arquitetura Medallion: Bronze -> Silver -> Gold"
    )


if __name__ == "__main__":
    main()
