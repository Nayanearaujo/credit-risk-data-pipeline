"""
Dashboard Streamlit — Plataforma de Risco de Credito
=====================================================
Autora: Nayane Araujo | github.com/Nayanearaujo
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Risco de Credito | Nayane Araujo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* KPI cards com fundo escuro e texto branco */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #3b82f6;
    color: white !important;
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 13px !important;
}
/* Status badges */
.status-ok {
    background: #065f46;
    color: #6ee7b7;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 14px;
    display: inline-block;
    margin: 4px 0;
}
.status-warn {
    background: #78350f;
    color: #fbbf24;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 14px;
    display: inline-block;
    margin: 4px 0;
}
/* Resultado do simulador */
.risk-card {
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold"
MODELS_PATH = ROOT / "src" / "models"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_data
def load_gold(table: str) -> pd.DataFrame:
    path = GOLD_PATH / f"{table}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Garante coluna de label legivel
    if "loan_status" in df.columns and "status_label" not in df.columns:
        df["status_label"] = df["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})
    return df


@st.cache_resource
def load_model():
    path = MODELS_PATH / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.title("💳 Risco de Credito")
        st.markdown("**Nayane Araujo**  \nEng. de Dados")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github)]"
            "(https://github.com/Nayanearaujo)"
        )
        st.divider()
        page = st.radio(
            "Pagina",
            ["Visao Geral", "Analise por Segmento", "Simulador de Credito"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Bronze -> Silver -> Gold")
        st.caption("Medallion Architecture")
        fact = load_gold("fact_loans")
        model = load_model()
        if not fact.empty:
            st.markdown('<div class="status-ok">Pipeline: OK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warn">Pipeline nao rodou</div>', unsafe_allow_html=True)
        if model:
            st.markdown('<div class="status-ok">Modelo: OK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warn">Modelo nao treinado</div>', unsafe_allow_html=True)
    return page


# ---------------------------------------------------------------------------
# Pagina 1: Visao Geral
# ---------------------------------------------------------------------------

def page_overview():
    st.title("Visao Geral do Portfolio de Credito")
    st.caption("Analise end-to-end com Arquitetura Medallion")

    fact = load_gold("fact_loans")
    if fact.empty:
        st.error("Execute o pipeline primeiro: `python run_pipeline.py`")
        return

    # KPIs
    total = len(fact)
    inad = int(fact["loan_status"].sum())
    taxa = inad / total
    vol = fact["loan_amnt"].sum()
    taxa_juros = fact["loan_int_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Emprestimos", f"{total:,}")
    c2.metric("Inadimplentes", f"{inad:,}", delta=f"{taxa:.1%} do total", delta_color="inverse")
    c3.metric("Taxa de Inadimplencia", f"{taxa:.1%}")
    c4.metric("Taxa de Juros Media", f"{taxa_juros:.1f}% a.a.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graficos linha 1
    c_left, c_right = st.columns(2)

    with c_left:
        agg = load_gold("agg_default_by_grade")
        if not agg.empty:
            fig = px.bar(
                agg, x="loan_grade", y="default_rate",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Inadimplencia por Grade de Risco",
                labels={"loan_grade": "Grade", "default_rate": "Taxa"},
                text_auto=".1%",
            )
            fig.update_traces(textfont_size=12)
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with c_right:
        agg_i = load_gold("agg_default_by_intent")
        if not agg_i.empty:
            fig = px.bar(
                agg_i.sort_values("default_rate"),
                x="default_rate", y="loan_intent",
                orientation="h",
                color="default_rate",
                color_continuous_scale="RdYlGn_r",
                title="Inadimplencia por Finalidade do Emprestimo",
                labels={"loan_intent": "", "default_rate": "Taxa"},
                text_auto=".1%",
            )
            fig.update_traces(textfont_size=12)
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # Grafico linha 2
    agg_a = load_gold("agg_default_by_age")
    if not agg_a.empty:
        order = ["18-25", "26-35", "36-45", "46-60", "60+"]
        agg_a["age_group"] = pd.Categorical(agg_a["age_group"], categories=order, ordered=True)
        agg_a = agg_a.sort_values("age_group")
        fig = px.bar(
            agg_a, x="age_group", y="default_rate",
            color="default_rate",
            color_continuous_scale="Blues_r",
            title="Inadimplencia por Faixa Etaria",
            labels={"age_group": "Faixa Etaria", "default_rate": "Taxa"},
            text_auto=".1%",
        )
        fig.update_traces(textfont_size=13)
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Distribuicao moradia
    home_d = fact["person_home_ownership"].value_counts().reset_index()
    home_d.columns = ["Tipo", "Qtd"]
    fig = px.pie(home_d, values="Qtd", names="Tipo",
                 title="Distribuicao por Tipo de Moradia",
                 color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#6366f1"])
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 2: Analise por Segmento
# ---------------------------------------------------------------------------

def page_segment():
    st.title("Analise por Segmento")
    st.caption("Filtre os dados e explore padroes de inadimplencia")

    fact = load_gold("fact_loans")
    if fact.empty:
        st.error("Execute o pipeline primeiro: `python run_pipeline.py`")
        return

    # Garante coluna status_label sempre presente
    if "status_label" not in fact.columns:
        fact = fact.copy()
        fact["status_label"] = fact["loan_status"].map({0: "Adimplente", 1: "Inadimplente"})

    # Filtros
    with st.expander("Filtros", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            grades = sorted(fact["loan_grade"].dropna().unique())
            sel_grade = st.multiselect("Grade", grades, default=grades)
        with fc2:
            intents = sorted(fact["loan_intent"].dropna().unique())
            sel_intent = st.multiselect("Finalidade", intents, default=intents)
        with fc3:
            rmin = int(fact["person_income"].min())
            rmax = int(fact["person_income"].quantile(0.99))
            sel_renda = st.slider("Renda Anual (R$)", rmin, rmax, (rmin, min(rmax, 200_000)))

    mask = (
        fact["loan_grade"].isin(sel_grade) &
        fact["loan_intent"].isin(sel_intent) &
        fact["person_income"].between(sel_renda[0], sel_renda[1])
    )
    fact_f = fact[mask].copy()

    # Garante valores validos para o campo de status
    fact_f["status_label"] = fact_f["loan_status"].map({0: "Adimplente", 1: "Inadimplente"}).fillna("Outro")
    fact_f = fact_f[fact_f["status_label"].isin(["Adimplente", "Inadimplente"])]

    st.caption(f"{len(fact_f):,} registros com os filtros aplicados")

    if fact_f.empty:
        st.warning("Nenhum registro com esses filtros.")
        return

    COR_MAP = {"Adimplente": "#10b981", "Inadimplente": "#ef4444"}

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            fact_f, x="loan_amnt",
            color="status_label",
            color_discrete_map=COR_MAP,
            category_orders={"status_label": ["Adimplente", "Inadimplente"]},
            nbins=40, barmode="overlay", opacity=0.75,
            title="Distribuicao do Valor Solicitado (R$)",
            labels={"loan_amnt": "Valor (R$)", "status_label": "Status"},
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            fact_f, x="status_label", y="loan_int_rate",
            color="status_label",
            color_discrete_map=COR_MAP,
            category_orders={"status_label": ["Adimplente", "Inadimplente"]},
            title="Taxa de Juros por Status",
            labels={"loan_int_rate": "Taxa (%)", "status_label": ""},
        )
        fig.update_layout(showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter renda x emprestimo
    sample = fact_f.sample(min(2000, len(fact_f)), random_state=42)
    fig = px.scatter(
        sample, x="person_income", y="loan_amnt",
        color="status_label",
        color_discrete_map=COR_MAP,
        opacity=0.55,
        title="Renda vs Valor do Emprestimo",
        labels={
            "person_income": "Renda Anual (R$)",
            "loan_amnt": "Valor do Emprestimo (R$)",
            "status_label": "Status",
        },
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Comparacao de modelos
    mc = load_gold("model_comparison")
    if not mc.empty:
        st.subheader("Comparacao de Modelos de Machine Learning")
        cols_num = [c for c in ["auc_roc", "f1_score", "precision", "recall"] if c in mc.columns]
        styled = mc.style.highlight_max(subset=cols_num, color="#065f46").format({c: "{:.4f}" for c in cols_num})
        st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 3: Simulador
# ---------------------------------------------------------------------------

def page_simulator():
    st.title("Simulador de Concessao de Credito")
    st.caption("Preencha os campos e calcule a probabilidade de inadimplencia em tempo real.")

    model = load_model()
    if model is None:
        st.error("Modelo nao treinado. Execute: `python run_pipeline.py`")
        return

    with st.form("form_credito", clear_on_submit=False):
        st.subheader("Dados do Tomador")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            age = st.number_input("Idade", 18, 80, 30, 1)
        with r1c2:
            income = st.number_input("Renda Anual (R$)", 10_000, 1_000_000, 50_000, 5_000)
        with r1c3:
            emp_length = st.number_input("Anos de Emprego", 0, 40, 5, 1)
        with r1c4:
            home = st.selectbox("Moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        st.subheader("Dados do Emprestimo")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            loan_amnt = st.number_input("Valor Solicitado (R$)", 500, 35_000, 10_000, 500)
        with r2c2:
            loan_int_rate = st.number_input("Taxa de Juros (% a.a.)", 5.0, 30.0, 12.0, 0.5)
        with r2c3:
            loan_intent = st.selectbox("Finalidade", [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])
        with r2c4:
            loan_grade = st.selectbox("Grade de Risco", ["A", "B", "C", "D", "E", "F", "G"])

        st.subheader("Historico de Credito")
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            cred_hist = st.number_input("Anos de Historico de Credito", 0, 30, 5, 1)
        with r3c2:
            prior_default = st.selectbox("Teve inadimplencia anterior?", ["Nao", "Sim"])
        with r3c3:
            pct = loan_amnt / income
            st.markdown(f"""
            <div style='background:#1e3a5f; border-radius:8px; padding:14px; margin-top:4px;'>
                <div style='color:#94a3b8; font-size:12px; text-transform:uppercase;'>
                    Comprometimento de Renda
                </div>
                <div style='color:{"#ef4444" if pct > 0.3 else "#10b981"};
                            font-size:28px; font-weight:700;'>
                    {pct:.1%}
                </div>
                <div style='color:#64748b; font-size:11px;'>
                    {"Acima do limite recomendado" if pct > 0.3 else "Dentro do limite recomendado"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Calcular Probabilidade de Inadimplencia",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        has_prior = 1 if prior_default == "Sim" else 0
        loan_pct = loan_amnt / income
        custo_total = loan_amnt * (1 + loan_int_rate / 100)
        renda_por_emp = income / (emp_length + 1)

        feat = np.array([[
            float(age), float(income), float(emp_length),
            float(loan_amnt), loan_int_rate, loan_pct,
            float(cred_hist), float(grade_map.get(loan_grade, 2)),
            float(has_prior), custo_total, renda_por_emp,
            loan_amnt / (cred_hist + 1), loan_pct,
        ]])

        n = model.n_features_in_
        feat = feat[:, :n] if feat.shape[1] >= n else np.pad(feat, ((0, 0), (0, n - feat.shape[1])))

        try:
            prob = float(model.predict_proba(feat)[0][1])

            if prob < 0.3:
                cor = "#10b981"
                nivel = "BAIXO RISCO"
                decisao = "Aprovado"
                icon = "✅"
                bg = "#064e3b"
            elif prob < 0.55:
                cor = "#f59e0b"
                nivel = "RISCO MODERADO"
                decisao = "Aprovado com restricoes"
                icon = "⚠️"
                bg = "#78350f"
            else:
                cor = "#ef4444"
                nivel = "ALTO RISCO"
                decisao = "Negado"
                icon = "❌"
                bg = "#7f1d1d"

            st.divider()
            col_r, col_g = st.columns([1, 1])

            with col_r:
                st.markdown(f"""
                <div style='background:{bg}; border:2px solid {cor};
                            border-radius:14px; padding:32px; text-align:center;'>
                    <div style='font-size:40px; margin-bottom:8px;'>{icon}</div>
                    <div style='color:{cor}; font-size:16px; font-weight:600;
                                text-transform:uppercase; letter-spacing:0.1em;'>
                        {nivel}
                    </div>
                    <div style='color:white; font-size:52px; font-weight:800; margin:12px 0;'>
                        {prob:.1%}
                    </div>
                    <div style='color:#cbd5e1; font-size:13px; margin-bottom:16px;'>
                        Probabilidade de Inadimplencia
                    </div>
                    <div style='background:{cor}; color:white; border-radius:8px;
                                padding:8px 16px; font-weight:700; font-size:15px;'>
                        {decisao}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_g:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 48, "color": "white"}},
                    delta={"reference": 30, "suffix": "%", "position": "bottom"},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#94a3b8"}},
                        "bar": {"color": cor, "thickness": 0.3},
                        "bgcolor": "#0f172a",
                        "bordercolor": "#1e293b",
                        "steps": [
                            {"range": [0, 30], "color": "#064e3b"},
                            {"range": [30, 55], "color": "#78350f"},
                            {"range": [55, 100], "color": "#7f1d1d"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                    title={"text": "Score de Risco", "font": {"color": "#94a3b8", "size": 14}},
                ))
                fig.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Resumo
            st.subheader("Resumo da Analise")
            dados = [
                ("Solicitante", f"{age} anos", f"Renda: R$ {income:,.0f}"),
                ("Emprestimo", f"R$ {loan_amnt:,.0f}", f"Taxa: {loan_int_rate:.1f}% a.a."),
                ("Grade", loan_grade, f"Comprometimento: {loan_pct:.1%}"),
                ("Historico", f"Hist. Credito: {cred_hist} anos", f"Inad. Anterior: {prior_default}"),
            ]
            for label, val1, val2 in dados:
                c_a, c_b, c_c = st.columns([1, 2, 2])
                c_a.markdown(f"**{label}**")
                c_b.write(val1)
                c_c.write(val2)

        except Exception as e:
            st.error(f"Erro no calculo: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()
    if page == "Visao Geral":
        page_overview()
    elif page == "Analise por Segmento":
        page_segment()
    elif page == "Simulador de Credito":
        page_simulator()

    st.divider()
    st.caption(
        "Pipeline de Risco de Credito - Nayane Araujo | "
        "[GitHub](https://github.com/Nayanearaujo/credit-risk-data-pipeline)"
    )


if __name__ == "__main__":
    main()
