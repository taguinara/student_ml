"""
Dashboard Streamlit — Projeto Integrador UC 10 | Machine Learning
Previsão de Desempenho Acadêmico de Estudantes (A, B, C, D, F)
Repositório: https://github.com/taguinara/student_ml
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="UC10 · ML — Desempenho Estudantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOMIZADO
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #e0e7ff; margin: 0.3rem 0 0 0; font-size: 1.1rem; }

    .metric-card {
        background: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card h4 { margin: 0; color: #475569; font-size: 0.85rem; font-weight: 500; }
    .metric-card p  { margin: 0.3rem 0 0 0; color: #1e293b; font-size: 1.8rem; font-weight: 700; }

    .section-title {
        color: #1e3a8a;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CAMINHOS DOS ARQUIVOS (com fallback)
# ============================================================
POSSIVEIS_CSV = [
    "desempenho_estudantil/tratamento_dados_binario.csv",
    "desempenho_estudantil/traducao_dados.csv",
    "desempenho_estudantil/student_performance_dataset.csv",
    "tratamento_dados_binario.csv",
    "traducao_dados.csv",
    "student_performance_dataset.csv",
]

POSSIVEIS_RESULTADOS = [
    "reports/comparacao_modelos.csv",
    "comparacao_modelos.csv",
]

POSSIVEIS_IMGS = {
    "dist_antes":   ["reports/dist_classes_antes.png", "dist_classes_antes.png"],
    "dist_depois":  ["reports/dist_classes_depois.png", "dist_classes_depois.png"],
    "pca_2d":       ["reports/pca_2d.png", "pca_2d.png"],
    "pca_var":      ["reports/pca_variancia.png", "pca_variancia.png"],
    "knn":          ["reports/comparacao_knn.png", "comparacao_knn.png"],
    "knn_vs_rf":    ["reports/comparacao_knn_vs_rf.png", "comparacao_knn_vs_rf.png"],
    "comparacao":   ["reports/comparacao_modelos.png", "comparacao_modelos.png"],
    "cm_melhor":    ["reports/cm_melhor_modelo.png", "cm_melhor_modelo.png"],
}


def encontrar_arquivo(lista_caminhos):
    for c in lista_caminhos:
        if os.path.exists(c):
            return c
    return None


@st.cache_data(show_spinner=False)
def carregar_dados(caminho: str) -> pd.DataFrame:
    return pd.read_csv(caminho)


@st.cache_data(show_spinner=False)
def carregar_resultados(caminho: str) -> pd.DataFrame:
    return pd.read_csv(caminho)


# ============================================================
# PIPELINE ML (cacheado)
# ============================================================
@st.cache_resource(show_spinner=False)
def executar_pipeline(caminho_csv: str, n_pca: int = 4, random_state: int = 42):
    """Executa o pipeline completo e devolve tudo que o dashboard precisa."""

    df = pd.read_csv(caminho_csv)
    df = df.drop_duplicates().dropna()

    # Normaliza target
    if "nota_final" in df.columns:
        df["nota_final"] = df["nota_final"].astype(str).str.strip().str.upper()

    # Features / Target
    feature_cols = [c for c in df.columns if c != "nota_final"]
    X = df[feature_cols].copy()
    y = df["nota_final"].copy()

    # Converte possíveis strings remanescentes
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].map({"Sim": 1, "Não": 0, "Masculino": 1, "Feminino": 0}).fillna(X[col])
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna()

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # SMOTE (apenas no treino)
    min_class = y_train.value_counts().min()
    k_neighbors = max(1, min(5, min_class - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Padronização
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc  = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)

    # Modelos
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train_bal)

    rf = RandomForestClassifier(n_estimators=300, random_state=random_state,
                                class_weight="balanced")
    rf.fit(X_train_pca, y_train_bal)

    return {
        "df_original": df,
        "feature_cols": feature_cols,
        "classes": sorted(y.unique()),
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_bal": X_train_bal, "y_train_bal": y_train_bal,
        "X_train_pca": X_train_pca, "X_test_pca": X_test_pca,
        "variancia": pca.explained_variance_ratio_,
        "knn": knn, "rf": rf,
        "scaler": scaler, "pca": pca,
        "y_pred_knn": knn.predict(X_test_pca),
        "y_pred_rf":  rf.predict(X_test_pca),
    }


def calcular_metricas(y_true, y_pred):
    return {
        "Acurácia":  accuracy_score(y_true, y_pred),
        "Precisão":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 Previsão de Desempenho Acadêmico</h1>
    <p>Projeto Integrador · UC 10 · Machine Learning · Ciência de Dados</p>
    <p style="font-size:0.95rem; opacity:0.9;">
        Napoleão Júnior · Tainara Almeida · Júlia Stefany
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.title("Navegação")
    st.markdown("---")

    caminho_csv = encontrar_arquivo(POSSIVEIS_CSV)
    caminho_res = encontrar_arquivo(POSSIVEIS_RESULTADOS)

    if caminho_csv is None:
        st.error("❌ Nenhum CSV encontrado. Faça upload:")
        upload = st.file_uploader("Dataset (.csv)", type=["csv"])
        if upload:
            os.makedirs("desempenho_estudantil", exist_ok=True)
            caminho_csv = "desempenho_estudantil/upload.csv"
            with open(caminho_csv, "wb") as f:
                f.write(upload.getbuffer())
            st.rerun()
        st.stop()

    st.success(f"📁 Dataset: `{os.path.basename(caminho_csv)}`")

    st.markdown("### ⚙️ Parâmetros do Pipeline")
    n_pca = st.slider("Componentes PCA", 2, 5, 4)
    random_state = st.number_input("Random State", 0, 9999, 42)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("- [📦 Repositório GitHub](https://github.com/taguinara/student_ml)")
    st.markdown("- [📊 Dataset (Kaggle)](https://www.kaggle.com/datasets)")

    st.markdown("---")
    st.caption("Desenvolvido para o PI da UC 10 · 2026")


# ============================================================
# EXECUTA PIPELINE
# ============================================================
with st.spinner("🔄 Executando pipeline de Machine Learning..."):
    resultado = executar_pipeline(caminho_csv, n_pca=n_pca, random_state=random_state)

df_original = resultado["df_original"]
metricas_knn = calcular_metricas(resultado["y_test"], resultado["y_pred_knn"])
metricas_rf  = calcular_metricas(resultado["y_test"], resultado["y_pred_rf"])


# ============================================================
# MÉTRICAS NO TOPO
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card"><h4>AMOSTRAS</h4>
        <p>{len(df_original)}</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><h4>FEATURES</h4>
        <p>{len(resultado['feature_cols'])}</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><h4>CLASSES</h4>
        <p>{len(resultado['classes'])}</p></div>""", unsafe_allow_html=True)
with c4:
    melhor_f1 = max(metricas_knn["F1-macro"], metricas_rf["F1-macro"])
    st.markdown(f"""<div class="metric-card"><h4>MELHOR F1-MACRO</h4>
        <p>{melhor_f1:.3f}</p></div>""", unsafe_allow_html=True)
with c5:
    var_acum = resultado["variancia"].sum() * 100
    st.markdown(f"""<div class="metric-card"><h4>VAR. PCA ({n_pca} comp.)</h4>
        <p>{var_acum:.1f}%</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "🧹 Dados & Limpeza",
    "📉 PCA",
    "🤖 Modelos",
    "🎯 Previsão",
    "ℹ️ Sobre"
])


# ---------- TAB 1 · VISÃO GERAL ----------
with tab1:
    st.markdown('<h2 class="section-title">📊 Visão Geral do Projeto</h2>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("""
        ### 🎯 Objetivo
        Prever a **nota final** de estudantes (A, B, C, D, F) a partir de hábitos
        de estudo e características socioeconômicas, permitindo **intervenções
        pedagógicas precoces**.

        ### 🧭 Pipeline aplicado
        1. **Aquisição** — Dataset Student Performance (Kaggle)
        2. **Limpeza** — Tradução, remoção de duplicados e dados omissos
        3. **Tratamento** — Conversão de variáveis categóricas
        4. **Split** — 70/30 estratificado
        5. **Balanceamento** — SMOTE (apenas no treino)
        6. **Padronização** — StandardScaler
        7. **Redução de dimensionalidade** — PCA (2 a 5 componentes)
        8. **Treinamento** — KNN e Random Forest
        9. **Avaliação** — Acurácia, Precisão, Recall, F1-macro
        10. **Previsão** — Classificação para novos estudantes
        """)

    with col_b:
        st.markdown("### 🥇 Melhor Modelo (nesta execução)")
        melhor_nome = "Random Forest" if metricas_rf["F1-macro"] >= metricas_knn["F1-macro"] else "KNN (k=5)"
        melhor = metricas_rf if metricas_rf["F1-macro"] >= metricas_knn["F1-macro"] else metricas_knn

        st.success(f"**{melhor_nome}** com PCA = {n_pca} componentes")

        df_m = pd.DataFrame({
            "Métrica": list(melhor.keys()),
            "Valor":   [f"{v:.4f}" for v in melhor.values()]
        })
        st.dataframe(df_m, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Amostra do Dataset")
    st.dataframe(df_original.head(15), use_container_width=True, height=350)


# ---------- TAB 2 · DADOS & LIMPEZA ----------
with tab2:
    st.markdown('<h2 class="section-title">🧹 Dados & Limpeza</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📥 Linhas originais (após limpeza)", len(df_original))
    with col2:
        st.metric("🔁 Duplicados removidos",
                  df_original.duplicated().sum())
    with col3:
        st.metric("❓ Dados omissos restantes",
                  int(df_original.isnull().sum().sum()))

    st.markdown("---")

    # Distribuição do target
    st.markdown("### 🎯 Distribuição das Classes")
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        contagem_antes = resultado["y_train"].value_counts().sort_index()
        fig1 = px.bar(
            x=contagem_antes.index, y=contagem_antes.values,
            labels={"x": "Nota Final", "y": "Quantidade"},
            title="Antes do balanceamento (treino)",
            color=contagem_antes.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col_dist2:
        contagem_depois = resultado["y_train_bal"].value_counts().sort_index()
        fig2 = px.bar(
            x=contagem_depois.index, y=contagem_depois.values,
            labels={"x": "Nota Final", "y": "Quantidade"},
            title="Depois do balanceamento (SMOTE)",
            color=contagem_depois.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 O SMOTE foi aplicado **apenas no conjunto de treino** para evitar "
            "vazamento de dados (*data leakage*). O conjunto de teste permanece "
            "com a distribuição original.")

    # Estatísticas descritivas
    st.markdown("### 📈 Estatísticas Descritivas")
    st.dataframe(df_original.describe().T, use_container_width=True)


# ---------- TAB 3 · PCA ----------
with tab3:
    st.markdown('<h2 class="section-title">📉 Análise de Componentes Principais (PCA)</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        var = resultado["variancia"]
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(var))],
            y=var * 100, name="Variância individual",
            marker_color="#3b82f6"
        ))
        fig_var.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(var))],
            y=np.cumsum(var) * 100, name="Acumulada",
            mode="lines+markers", line=dict(color="#dc2626", width=3)
        ))
        fig_var.update_layout(
            title="Variância explicada por componente",
            yaxis_title="% Variância", height=420
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with col2:
        X_pca = resultado["X_train_pca"]
        y_tr = resultado["y_train_bal"]

        if X_pca.shape[1] >= 2:
            df_pca = pd.DataFrame({
                "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Classe": y_tr.values
            })
            fig_pca = px.scatter(
                df_pca, x="PC1", y="PC2", color="Classe",
                title=f"Projeção 2D — {X_pca.shape[1]} componentes",
                color_discrete_sequence=px.colors.qualitative.Set1,
                opacity=0.6
            )
            fig_pca.update_layout(height=420)
            st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown(f"""
    ### 🧠 Interpretação
    - Com **{n_pca} componentes**, o PCA explica **{resultado['variancia'].sum()*100:.2f}%** da variância total.
    - A projeção 2D mostra que as classes **apresentam sobreposição considerável**,
      o que explica a dificuldade dos modelos em separá-las com alta acurácia.
    - A redução dimensional ajudou a **acelerar o treinamento** e a **reduzir ruído**.
    """)


# ---------- TAB 4 · MODELOS ----------
with tab4:
    st.markdown('<h2 class="section-title">🤖 Avaliação dos Modelos</h2>', unsafe_allow_html=True)

    # Tabela comparativa
    df_comp = pd.DataFrame({
        "Modelo": ["KNN (k=5)", "Random Forest"],
        "Acurácia": [metricas_knn["Acurácia"], metricas_rf["Acurácia"]],
        "Precisão": [metricas_knn["Precisão"], metricas_rf["Precisão"]],
        "Recall":   [metricas_knn["Recall"],   metricas_rf["Recall"]],
        "F1-macro": [metricas_knn["F1-macro"], metricas_rf["F1-macro"]],
    })
    st.dataframe(df_comp.style.format({
        "Acurácia": "{:.4f}", "Precisão": "{:.4f}",
        "Recall": "{:.4f}", "F1-macro": "{:.4f}"
    }).highlight_max(axis=0, color="#bbf7d0"), use_container_width=True)

    # Gráfico comparativo
    fig_comp = go.Figure()
    metricas = ["Acurácia", "Precisão", "Recall", "F1-macro"]
    fig_comp.add_trace(go.Bar(
        name="KNN (k=5)", x=metricas,
        y=[metricas_knn[m] for m in metricas],
        marker_color="#3b82f6",
        text=[f"{metricas_knn[m]:.3f}" for m in metricas],
        textposition="outside"
    ))
    fig_comp.add_trace(go.Bar(
        name="Random Forest", x=metricas,
        y=[metricas_rf[m] for m in metricas],
        marker_color="#10b981",
        text=[f"{metricas_rf[m]:.3f}" for m in metricas],
        textposition="outside"
    ))
    fig_comp.add_hline(y=0.5, line_dash="dash", line_color="red",
                       annotation_text="Baseline (50%)")
    fig_comp.update_layout(
        barmode="group", title="Comparação de métricas — KNN vs Random Forest",
        yaxis_title="Valor", yaxis_range=[0, 1.1], height=450
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Matrizes de Confusão")

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        cm_knn = confusion_matrix(resultado["y_test"], resultado["y_pred_knn"],
                                  labels=resultado["classes"])
        fig_cm_knn = px.imshow(
            cm_knn, text_auto=True, aspect="auto",
            x=resultado["classes"], y=resultado["classes"],
            color_continuous_scale="Blues",
            title="KNN (k=5)"
        )
        fig_cm_knn.update_layout(height=450)
        st.plotly_chart(fig_cm_knn, use_container_width=True)

    with col_cm2:
        cm_rf = confusion_matrix(resultado["y_test"], resultado["y_pred_rf"],
                                 labels=resultado["classes"])
        fig_cm_rf = px.imshow(
            cm_rf, text_auto=True, aspect="auto",
            x=resultado["classes"], y=resultado["classes"],
            color_continuous_scale="Greens",
            title="Random Forest"
        )
        fig_cm_rf.update_layout(height=450)
        st.plotly_chart(fig_cm_rf, use_container_width=True)

    # Classification report
    st.markdown("---")
    st.markdown("### 📋 Relatório de Classificação Detalhado")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**KNN (k=5)**")
        st.text(classification_report(resultado["y_test"], resultado["y_pred_knn"],
                                       zero_division=0))
    with col_r2:
        st.markdown("**Random Forest**")
        st.text(classification_report(resultado["y_test"], resultado["y_pred_rf"],
                                       zero_division=0))

    # Importância das features (RF)
    st.markdown("---")
    st.markdown("### 🌟 Importância das Features (Random Forest)")
    importancias = resultado["rf"].feature_importances_
    df_imp = pd.DataFrame({
        "Feature": resultado["feature_cols"][:len(importancias)],
        "Importância": importancias
    }).sort_values("Importância", ascending=True)

    fig_imp = px.bar(
        df_imp, x="Importância", y="Feature", orientation="h",
        color="Importância", color_continuous_scale="Viridis",
        title="Importância das variáveis no modelo Random Forest"
    )
    fig_imp.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)


# ---------- TAB 5 · PREVISÃO ----------
with tab5:
    st.markdown('<h2 class="section-title">🎯 Previsão para Novo Estudante</h2>',
                unsafe_allow_html=True)
    st.markdown("Preencha os dados do estudante e veja a previsão do modelo treinado.")

    modelo_escolhido = st.radio(
        "Escolha o modelo:",
        ["Random Forest", "KNN (k=5)"],
        horizontal=True
    )

    with st.form("form_previsao"):
        col1, col2, col3 = st.columns(3)

        with col1:
            genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
            horas_estudo = st.slider("Horas de estudo/semana", 0.0, 20.0, 4.5, 0.5)
            frequencia = st.slider("Frequência (%)", 0, 100, 90)

        with col2:
            horas_sono = st.slider("Horas de sono/noite", 0.0, 12.0, 7.0, 0.5)
            escolaridade = st.selectbox(
                "Escolaridade dos pais",
                ["Ensino Médio", "Bacharelado", "Mestrado", "Doutorado"]
            )
            nota_anterior = st.slider("Nota anterior", 0.0, 20.0, 15.0, 0.5)

        with col3:
            internet = st.selectbox("Acesso à internet", ["Sim", "Não"])
            extracurr = st.selectbox("Atividades extracurriculares", ["Sim", "Não"])
            trabalho  = st.selectbox("Trabalho meio período", ["Sim", "Não"])

        submitted = st.form_submit_button("🔮 Prever Desempenho", use_container_width=True)

    if submitted:
        mapa_genero = {"Masculino": 1, "Feminino": 0}
        mapa_esc = {"Ensino Médio": 1, "Bacharelado": 2, "Mestrado": 3, "Doutorado": 4}
        mapa_sn = {"Sim": 1, "Não": 0}

        entrada_dict = {
            "genero": mapa_genero[genero],
            "horas_estudo": horas_estudo,
            "percentual_frequencia": frequencia,
            "horas_sono": horas_sono,
            "escolaridade_pais": mapa_esc[escolaridade],
            "acesso_internet": mapa_sn[internet],
            "atividades_extracurriculares": mapa_sn[extracurr],
            "trabalho_meio_periodo": mapa_sn[trabalho],
            "nota_anterior": nota_anterior,
        }

        # Alinha com as features usadas no treino
        X_novo = pd.DataFrame([entrada_dict])
        for col in resultado["feature_cols"]:
            if col not in X_novo.columns:
                X_novo[col] = 0
        X_novo = X_novo[resultado["feature_cols"]]

        X_novo_sc  = resultado["scaler"].transform(X_novo)
        X_novo_pca = resultado["pca"].transform(X_novo_sc)

        if modelo_escolhido == "Random Forest":
            modelo = resultado["rf"]
        else:
            modelo = resultado["knn"]

        pred = modelo.predict(X_novo_pca)[0]
        proba = modelo.predict_proba(X_novo_pca)[0]
        classes = modelo.classes_

        st.markdown("---")
        st.markdown("### 🎉 Resultado da Previsão")

        col_r1, col_r2 = st.columns([1, 1.3])
        with col_r1:
            cor = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🟠", "F": "🔴"}.get(pred, "⚪")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg,#1e3a8a,#3b82f6);
                        padding:2rem; border-radius:12px; text-align:center; color:white;">
                <p style="margin:0; font-size:1.1rem;">Nota prevista</p>
                <p style="margin:0.3rem 0; font-size:4rem; font-weight:800;">{cor} {pred}</p>
                <p style="margin:0; font-size:0.9rem;">Modelo: {modelo_escolhido}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            df_prob = pd.DataFrame({"Classe": classes, "Probabilidade": proba})
            fig_prob = px.bar(
                df_prob, x="Classe", y="Probabilidade",
                color="Probabilidade", color_continuous_scale="Blues",
                title="Probabilidade por classe",
                text=df_prob["Probabilidade"].apply(lambda v: f"{v*100:.1f}%")
            )
            fig_prob.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

        st.info(f"💡 **Interpretação:** Com base nos dados informados, o modelo "
                f"prevê que este estudante terá nota final **{pred}**. "
                f"A confiança do modelo nessa classe é de "
                f"**{max(proba)*100:.1f}%**.")


# ---------- TAB 6 · SOBRE ----------
with tab6:
    st.markdown('<h2 class="section-title">ℹ️ Sobre o Projeto</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎓 Projeto Integrador — UC 10 · Machine Learning
    **Curso:** Ciência de Dados  
    **Integrantes:** Napoleão Júnior · Tainara Almeida · Júlia Stefany

    ### 🎯 Objetivo
    Aplicar um pipeline completo de Machine Learning para **prever o desempenho
    acadêmico de estudantes** (notas A, B, C, D, F) com base em hábitos de estudo
    e características socioeconômicas.

    ### 🧰 Tecnologias Utilizadas
    - **Python 3.10+**
    - **Pandas / NumPy** — Manipulação de dados
    - **Scikit-learn** — Modelos e métricas
    - **Imbalanced-learn** — SMOTE
    - **Matplotlib / Seaborn / Plotly** — Visualizações
    - **Streamlit** — Dashboard interativo
    - **Git / GitHub** — Versionamento

    ### 📁 Estrutura do Pipeline
    - Aquisição → Limpeza → Tratamento → Split 70/30 → SMOTE → Padronização → PCA → Treinamento (KNN + RF) → Avaliação → Previsão
   
   
### 📊 Dataset
**Student Performance & Study Habits Dataset** — disponível no
[Kaggle](https://www.kaggle.com/datasets).

### 🔗 Links
- **Repositório:** [https://github.com/taguinara/student_ml](https://github.com/taguinara/student_ml)
- **Dataset:** [Kaggle Datasets](https://www.kaggle.com/datasets)

### ✅ Requisitos Atendidos
| Requisito | Status |
|---|---|
| Versionamento com GIT | ✅ |
| Dataset escolhido e carregado | ✅ |
| Split treino/teste | ✅ |
| Limpeza e tratamento | ✅ |
| Balanceamento (SMOTE) | ✅ |
| PCA com diferentes componentes | ✅ |
| Múltiplos classificadores (KNN, RF) | ✅ |
| Múltiplos parâmetros | ✅ |
| Matriz de confusão | ✅ |
| Previsão para novo padrão | ✅ |
| Análise visual | ✅ |

### ⚠️ Limitações e Trabalhos Futuros
- Acurácia máxima obtida ficou em torno de 53% (desafio multiclasse).
- **Sugestões:** testar classificação binária (Aprovado/Reprovado),
  incluir mais features, testar XGBoost / SVM / Gradient Boosting,
  aplicar tuning de hiperparâmetros com GridSearchCV.
""")

st.markdown("---")
st.caption("Dashboard desenvolvido com 💙 para o Projeto Integrador UC 10 · 2026") 