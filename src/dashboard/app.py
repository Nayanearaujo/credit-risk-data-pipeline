"""
Dashboard Streamlit - Plataforma de Risco de Credito
=====================================================
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
# Paleta de Cores Unificada (UI/UX - Alto Contraste Semantico)
# ---------------------------------------------------------------------------
PALETTE = {
    # Status binario: Contraste real entre categorias (Teal vs Laranja Coral)
    "adimplente": "#0d9488",  # Verde-petroleo / Teal (bom pagador, confiavel)
    "inadimplente": "#f97316",  # Coral / Laranja vibrante (alerta claro de risco)
    # Niveis de Decisao do Simulador
    "risk_low": "#0d9488",  # Teal / Aprovado
    "risk_mod": "#f59e0b",  # Ambar / Aprovado com restricoes
    "risk_high": "#ef4444",  # Coral avermelhado / Negado
    # Fundos dos cards de decisao
    "risk_low_bg": "#042f2e",
    "risk_mod_bg": "#451a03",
    "risk_high_bg": "#450a0a",
    # Paleta qualitativa para categorias nominais (ex: Moradia)
    "categorical": ["#0284c7", "#0d9488", "#f59e0b", "#8b5cf6", "#ec4899", "#64748b"],
    # Escala continua para taxas de risco e distribuicoes graduais
    "risk_continuous": ["#0d9488", "#eab308", "#f97316", "#ef4444"],
}

# ---------------------------------------------------------------------------
# Configuracao de Pagina e Estilos Globais
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Risco de Credito | Nayane Araujo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Espacamento superior para eliminar sobreposicao com a barra nativa do Streamlit */
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
}

/* Tipografia e hierarquia de cabecalhos */
h1 {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
}
h2, h3 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

/* KPI metric cards com contraste, sombra suave e acento visual */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 4px solid #0d9488;
    box-shadow: 0 4px 8px -1px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 12px -1px rgba(0, 0, 0, 0.35);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 13px !important;
}

/* Sidebar Radio Navigation com affordance de clique e estado ativo */
div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 8px;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out;
    background: rgba(255, 255, 255, 0.02);
}
div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(13, 148, 136, 0.5);
    transform: translateX(2px);
}
div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(13, 148, 136, 0.18) !important;
    border: 1px solid #0d9488 !important;
    border-left: 5px solid #0d9488 !important;
}
div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Status badges na sidebar */
.status-ok {
    background: rgba(13, 148, 136, 0.15);
    color: #2dd4bf;
    border: 1px solid #0d9488;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    display: inline-block;
    margin: 3px 0;
}
.status-warn {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid #f59e0b;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    display: inline-block;
    margin: 3px 0;
}

/* Card de drivers explicativos no simulador */
.driver-card {
    background: #1e293b;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 14px;
}
.driver-title {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}
.driver-item {
    font-size: 13px;
    color: #e2e8f0;
    margin-bottom: 6px;
    line-height: 1.5;
}
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold"
MODELS_PATH = ROOT / "src" / "models"


sys.path.insert(0, str(ROOT))
from src.models.feature_engineering import create_derived_features, encode_categoricals

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


@st.cache_resource
def load_scaler():
    path = MODELS_PATH / "scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_names():
    path = MODELS_PATH / "feature_names.csv"
    if not path.exists():
        return []
    lines = [line.strip() for line in open(path) if line.strip()]
    if lines and lines[0] == "0":
        lines = lines[1:]
    return lines


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        st.title("💳 Risco de Credito")
        st.markdown("**Nayane Araujo**  \nEngenharia de Dados e Machine Learning")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Nayanearaujo-181717?logo=github)]"
            "(https://github.com/Nayanearaujo/credit-risk-data-pipeline)"
        )
        st.divider()
        nav_options = [
            "📊 Visao Geral",
            "🔍 Analise por Segmento",
            "⚡ Simulador de Credito",
        ]
        selected = st.radio(
            "Navegacao",
            nav_options,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Arquitetura Medallion: Bronze -> Silver -> Gold")
        fact = load_gold("fact_loans")
        model = load_model()
        if not fact.empty:
            st.markdown(
                '<div class="status-ok">Pipeline: OK</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-warn">Pipeline nao rodou</div>',
                unsafe_allow_html=True,
            )
        if model:
            st.markdown(
                '<div class="status-ok">Modelo: OK</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-warn">Modelo nao treinado</div>',
                unsafe_allow_html=True,
            )

    if "Visao Geral" in selected:
        return "Visao Geral"
    if "Analise por Segmento" in selected:
        return "Analise por Segmento"
    return "Simulador de Credito"


# ---------------------------------------------------------------------------
# Pagina 1: Visao Geral
# ---------------------------------------------------------------------------


def page_overview():
    st.title("Visao Geral do Portfolio de Credito")
    st.caption("Monitoramento executivo e indicadores de risco da carteira de credito")

    fact = load_gold("fact_loans")
    if fact.empty:
        st.error("Execute o pipeline primeiro: `python run_pipeline.py`")
        return

    # KPIs agrupados no topo
    total = len(fact)
    inad = int(fact["loan_status"].sum())
    taxa = inad / total
    vol = fact["loan_amnt"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Emprestimos", f"{total:,}")
    c2.metric(
        "Inadimplentes",
        f"{inad:,}",
        delta=f"{taxa:.1%} do portfolio",
        delta_color="inverse",
    )
    c3.metric("Taxa de Inadimplencia", f"{taxa:.1%}")
    c4.metric("Volume Total (R$)", f"R$ {vol:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graficos linha 1
    c_left, c_right = st.columns(2)

    with c_left:
        agg = load_gold("agg_default_by_grade")
        if not agg.empty:
            fig = px.bar(
                agg,
                x="loan_grade",
                y="default_rate",
                color="default_rate",
                color_continuous_scale=PALETTE["risk_continuous"],
                title="Inadimplencia por Grade de Risco",
                labels={"loan_grade": "Grade", "default_rate": "Taxa de Inadimplencia"},
                text_auto=".1%",
            )
            fig.update_traces(textfont_size=12)
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c_right:
        agg_i = load_gold("agg_default_by_intent")
        if not agg_i.empty:
            fig = px.bar(
                agg_i.sort_values("default_rate"),
                x="default_rate",
                y="loan_intent",
                orientation="h",
                color="default_rate",
                color_continuous_scale=PALETTE["risk_continuous"],
                title="Inadimplencia por Finalidade do Emprestimo",
                labels={"loan_intent": "", "default_rate": "Taxa de Inadimplencia"},
                text_auto=".1%",
            )
            fig.update_traces(textfont_size=12)
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Graficos linha 2
    c_age, c_home = st.columns(2)

    with c_age:
        agg_a = load_gold("agg_default_by_age")
        if not agg_a.empty:
            order = ["18-25", "26-35", "36-45", "46-60", "60+"]
            agg_a["age_group"] = pd.Categorical(
                agg_a["age_group"], categories=order, ordered=True
            )
            agg_a = agg_a.sort_values("age_group")
            fig = px.bar(
                agg_a,
                x="age_group",
                y="default_rate",
                color="default_rate",
                color_continuous_scale=PALETTE["risk_continuous"],
                title="Inadimplencia por Faixa Etaria",
                labels={
                    "age_group": "Faixa Etaria",
                    "default_rate": "Taxa de Inadimplencia",
                },
                text_auto=".1%",
            )
            fig.update_traces(textfont_size=12)
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c_home:
        # Distribuicao de moradia em grafico de rosca com alto contraste
        home_d = fact["person_home_ownership"].value_counts().reset_index()
        home_d.columns = ["Tipo", "Qtd"]
        fig = px.pie(
            home_d,
            values="Qtd",
            names="Tipo",
            hole=0.45,
            title="Distribuicao por Tipo de Moradia",
            color_discrete_sequence=PALETTE["categorical"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 2: Analise por Segmento
# ---------------------------------------------------------------------------


def page_segment():
    st.title("Analise por Segmento")
    st.caption(
        "Filtre as caracteristicas da carteira e explore padroes de inadimplencia"
    )

    fact = load_gold("fact_loans")
    if fact.empty:
        st.error("Execute o pipeline primeiro: `python run_pipeline.py`")
        return

    # Garante coluna status_label sempre presente
    if "status_label" not in fact.columns:
        fact = fact.copy()
        fact["status_label"] = fact["loan_status"].map(
            {0: "Adimplente", 1: "Inadimplente"}
        )

    # Filtros
    with st.expander("Filtros de Segmentacao", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            grades = sorted(fact["loan_grade"].dropna().unique())
            sel_grade = st.multiselect("Grade de Risco", grades, default=grades)
        with fc2:
            intents = sorted(fact["loan_intent"].dropna().unique())
            sel_intent = st.multiselect("Finalidade", intents, default=intents)
        with fc3:
            rmin = int(fact["person_income"].min())
            rmax = int(fact["person_income"].quantile(0.99))
            sel_renda = st.slider(
                "Renda Anual (R$)", rmin, rmax, (rmin, min(rmax, 200_000))
            )

    mask = (
        fact["loan_grade"].isin(sel_grade)
        & fact["loan_intent"].isin(sel_intent)
        & fact["person_income"].between(sel_renda[0], sel_renda[1])
    )
    fact_f = fact[mask].copy()

    # Reconstroi status_label direto do loan_status
    fact_f["status_label"] = fact_f["loan_status"].map(
        {0: "Adimplente", 1: "Inadimplente"}
    )
    fact_f = fact_f.dropna(subset=["status_label"])

    st.caption(f"{len(fact_f):,} registros com os filtros aplicados")

    if fact_f.empty:
        st.warning("Nenhum registro encontrado com os filtros selecionados.")
        return

    adim = fact_f[fact_f["loan_status"] == 0]
    inad = fact_f[fact_f["loan_status"] == 1]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=adim["loan_amnt"],
                name="Adimplente",
                marker_color=PALETTE["adimplente"],
                opacity=0.8,
                nbinsx=40,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=inad["loan_amnt"],
                name="Inadimplente",
                marker_color=PALETTE["inadimplente"],
                opacity=0.8,
                nbinsx=40,
            )
        )
        fig.update_layout(
            barmode="overlay",
            title="Distribuicao do Valor Solicitado (R$)",
            xaxis_title="Valor (R$)",
            yaxis_title="Quantidade de Contratos",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=adim["loan_int_rate"].dropna(),
                name="Adimplente",
                marker_color=PALETTE["adimplente"],
                boxmean=True,
            )
        )
        fig.add_trace(
            go.Box(
                y=inad["loan_int_rate"].dropna(),
                name="Inadimplente",
                marker_color=PALETTE["inadimplente"],
                boxmean=True,
            )
        )
        fig.update_layout(
            title="Taxa de Juros por Status",
            yaxis_title="Taxa de Juros (% a.a.)",
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter renda x valor - amostra para performance fluida
    sample = fact_f.sample(min(2000, len(fact_f)), random_state=42)
    s_adim = sample[sample["loan_status"] == 0]
    s_inad = sample[sample["loan_status"] == 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s_adim["person_income"],
            y=s_adim["loan_amnt"],
            mode="markers",
            name="Adimplente",
            marker=dict(color=PALETTE["adimplente"], opacity=0.5, size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s_inad["person_income"],
            y=s_inad["loan_amnt"],
            mode="markers",
            name="Inadimplente",
            marker=dict(color=PALETTE["inadimplente"], opacity=0.5, size=5),
        )
    )
    fig.update_layout(
        title="Renda vs Valor do Emprestimo",
        xaxis_title="Renda Anual (R$)",
        yaxis_title="Valor Solicitado (R$)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparacao de modelos
    mc = load_gold("model_comparison")
    if not mc.empty:
        st.subheader("Comparacao de Modelos de Machine Learning")
        cols_num = [
            c for c in ["auc_roc", "f1_score", "precision", "recall"] if c in mc.columns
        ]
        styled = mc.style.highlight_max(subset=cols_num, color="#065f46").format(
            {c: "{:.4f}" for c in cols_num}
        )
        st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# Pagina 3: Simulador
# ---------------------------------------------------------------------------


def page_simulator():
    st.title("Simulador de Concessao de Credito")
    st.caption(
        "Insira as informacoes cadastrais e financeiras para estimar o risco em tempo real."
    )

    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    if model is None:
        st.error("Modelo nao treinado. Execute o pipeline: `python run_pipeline.py`")
        return

    with st.form("form_credito", clear_on_submit=False):
        st.subheader("Dados do Tomador")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            age = st.number_input("Idade", 18, 80, 30, 1)
        with r1c2:
            income = st.number_input(
                "Renda Anual (R$)", 10_000, 1_000_000, 50_000, 5_000
            )
        with r1c3:
            emp_length = st.number_input("Anos de Emprego", 0, 40, 5, 1)
        with r1c4:
            home = st.selectbox("Moradia", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        st.subheader("Dados do Emprestimo")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            loan_amnt = st.number_input(
                "Valor Solicitado (R$)", 500, 35_000, 10_000, 500
            )
        with r2c2:
            loan_int_rate = st.number_input(
                "Taxa de Juros (% a.a.)", 5.0, 30.0, 12.0, 0.5
            )
        with r2c3:
            loan_intent = st.selectbox(
                "Finalidade",
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ],
            )
        with r2c4:
            loan_grade = st.selectbox(
                "Grade de Risco", ["A", "B", "C", "D", "E", "F", "G"]
            )

        st.subheader("Historico de Credito")
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            cred_hist = st.number_input("Anos de Historico de Credito", 0, 30, 5, 1)
        with r3c2:
            prior_default = st.selectbox("Teve inadimplencia anterior?", ["Nao", "Sim"])
        with r3c3:
            pct = loan_amnt / income
            st.markdown(
                f"""
            <div style='background:#1e293b; border-radius:10px; padding:14px;
                        margin-top:4px; border:1px solid rgba(255,255,255,0.08);'>
                <div style='color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:0.05em;'>
                    Comprometimento de Renda
                </div>
                <div style='color:{"#ef4444" if pct > 0.3 else "#0d9488"};
                            font-size:26px; font-weight:700;'>
                    {pct:.1%}
                </div>
                <div style='color:#94a3b8; font-size:11px;'>
                    {"Acima do limite prudencial (30%)" if pct > 0.3 else "Dentro do limite prudencial (30%)"}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Calcular Probabilidade de Inadimplencia",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        raw_input = {
            "person_age": [float(age)],
            "person_income": [float(income)],
            "person_home_ownership": [home],
            "person_emp_length": [float(emp_length)],
            "loan_intent": [loan_intent],
            "loan_grade": [loan_grade],
            "loan_amnt": [float(loan_amnt)],
            "loan_int_rate": [float(loan_int_rate)],
            "loan_percent_income": [float(loan_amnt / income)],
            "cb_person_default_on_file": ["Y" if prior_default == "Sim" else "N"],
            "cb_person_cred_hist_length": [float(cred_hist)],
        }
        df_input = pd.DataFrame(raw_input)

        # Feature engineering identico ao pipeline de treinamento
        df_derived = create_derived_features(df_input)
        df_encoded = encode_categoricals(df_derived)

        # Reindexacao pelas features oficiais
        X_input = df_encoded.reindex(columns=feature_names, fill_value=0)

        # Normalizacao com StandardScaler
        if scaler is not None:
            X_eval = scaler.transform(X_input)
        else:
            X_eval = X_input.values

        try:
            prob = float(model.predict_proba(X_eval)[0][1])

            # Decisao e faixas de risco
            if prob < 0.30:
                cor = PALETTE["risk_low"]
                nivel = "BAIXO RISCO"
                decisao = "Aprovado"
                icon = "✅"
                bg = PALETTE["risk_low_bg"]
                explicacao = "Perfil compativel com politicas convencionais de concessao de credito."
            elif prob < 0.55:
                cor = PALETTE["risk_mod"]
                nivel = "RISCO MODERADO"
                decisao = "Aprovado com Restricoes / Revisao Manual"
                icon = "⚠️"
                bg = PALETTE["risk_mod_bg"]
                explicacao = "Indicadores intermediarios. Recomenda-se analise documental detalhada."
            else:
                cor = PALETTE["risk_high"]
                nivel = "ALTO RISCO"
                decisao = "Negado"
                icon = "❌"
                bg = PALETTE["risk_high_bg"]
                explicacao = "Probabilidade de inadimplencia acima do limite maximo tolerado pela politica."

            st.divider()
            col_r, col_g = st.columns([1, 1])

            with col_r:
                st.markdown(
                    f"""
                <div style='background:{bg}; border:2px solid {cor};
                            border-radius:14px; padding:28px; text-align:center;'>
                    <div style='font-size:38px; margin-bottom:6px;'>{icon}</div>
                    <div style='color:{cor}; font-size:15px; font-weight:700;
                                text-transform:uppercase; letter-spacing:0.08em;'>
                        {nivel}
                    </div>
                    <div style='color:white; font-size:50px; font-weight:800; margin:10px 0;'>
                        {prob:.1%}
                    </div>
                    <div style='color:#cbd5e1; font-size:13px; margin-bottom:14px;'>
                        Probabilidade Estimada de Inadimplencia
                    </div>
                    <div style='background:{cor}; color:white; border-radius:8px;
                                padding:8px 16px; font-weight:700; font-size:14px;'>
                        {decisao}
                    </div>
                    <div style='color:#cbd5e1; font-size:12px; margin-top:12px;'>
                        {explicacao}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col_g:
                # Gauge com zonas nitidas (Aprovado, Revisao, Negado)
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 44, "color": "white"}},
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "tickfont": {"color": "#94a3b8", "size": 11},
                            },
                            "bar": {"color": cor, "thickness": 0.28},
                            "bgcolor": "#0f172a",
                            "bordercolor": "#334155",
                            "steps": [
                                {"range": [0, 30], "color": "rgba(13, 148, 136, 0.35)"},
                                {
                                    "range": [30, 55],
                                    "color": "rgba(245, 158, 11, 0.35)",
                                },
                                {
                                    "range": [55, 100],
                                    "color": "rgba(239, 68, 68, 0.35)",
                                },
                            ],
                            "threshold": {
                                "line": {"color": "#f8fafc", "width": 3},
                                "thickness": 0.8,
                                "value": prob * 100,
                            },
                        },
                        title={
                            "text": "Score de Risco (0 - 100%)",
                            "font": {"color": "#94a3b8", "size": 13},
                        },
                    )
                )
                fig.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    margin=dict(l=25, r=25, t=40, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Analise dos Principais Fatores do Perfil (Drivers de Risco)
            st.subheader("Fatores Determinantes da Decisao")
            drivers = []

            # 1. Comprometimento
            if pct > 0.30:
                drivers.append(
                    f"⚠️ <b>Comprometimento de Renda:</b> Solicita {pct:.1%} da renda anual "
                    "(acima do teto prudencial de 30%), elevando a exposicao ao risco."
                )
            else:
                drivers.append(
                    f"✅ <b>Comprometimento de Renda:</b> {pct:.1%} da renda anual solicitado "
                    "esta dentro da margem conservadora."
                )

            # 2. Historico anterior
            if prior_default == "Sim":
                drivers.append(
                    "❌ <b>Historico de Inadimplencia:</b> Registro previo de inadimplencia no bureau "
                    "constitui um dos principais preditores negativos no modelo."
                )
            else:
                drivers.append(
                    "✅ <b>Historico Limpo:</b> Sem registros previos de inadimplencia identificados."
                )

            # 3. Grade e Taxa
            if loan_grade in ["A", "B"]:
                drivers.append(
                    f"✅ <b>Grade de Risco:</b> Classificacao favoravel (Grade {loan_grade}) "
                    f"com taxa de juros de {loan_int_rate:.1f}% a.a."
                )
            elif loan_grade in ["C", "D"]:
                drivers.append(
                    f"⚠️ <b>Grade de Risco Intermediaria:</b> Classificacao {loan_grade} "
                    f"com taxa de juros de {loan_int_rate:.1f}% a.a."
                )
            else:
                drivers.append(
                    f"❌ <b>Grade de Risco Elevada:</b> Classificacao {loan_grade} "
                    f"com taxa de juros de {loan_int_rate:.1f}% a.a."
                )

            # 4. Estabilidade de emprego
            if emp_length >= 3:
                drivers.append(
                    f"✅ <b>Estabilidade Empregaticia:</b> {emp_length:.0f} anos de emprego comprovados."
                )
            else:
                drivers.append(
                    f"⚠️ <b>Tempo de Emprego Recente:</b> Apenas {emp_length:.0f} anos no emprego atual."
                )

            drivers_html = "".join(
                [f"<div class='driver-item'>{d}</div>" for d in drivers]
            )
            st.markdown(
                f"""
                <div class='driver-card'>
                    <div class='driver-title'>Impacto dos Fatores neste Perfil</div>
                    {drivers_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Resumo tabular
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Resumo dos Dados Submetidos")
            dados = [
                ("Solicitante", f"{age} anos", f"Renda: R$ {income:,.0f}"),
                (
                    "Emprestimo",
                    f"R$ {loan_amnt:,.0f}",
                    f"Taxa: {loan_int_rate:.1f}% a.a.",
                ),
                ("Grade", loan_grade, f"Comprometimento: {pct:.1%}"),
                (
                    "Historico",
                    f"Hist. Credito: {cred_hist} anos",
                    f"Inad. Anterior: {prior_default}",
                ),
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
