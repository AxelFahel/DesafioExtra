import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import io
from PIL import Image
import unicodedata
import os
from groq import Groq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore

REMOVE_OUTLIERS = True
STANDARDIZE_DATA = True

state = {
    "df_raw": None,
    "df_processed": None,
    "client": None,
}

GROQ_API_KEY = os.environ.get("API_KEY_GROQ", "")

def init_groq():
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception:
        return None

def load_file(file):
    if file is None:
        return "❌ Nenhum arquivo carregado."
    try:
        if file.name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(file.name,"r") as z:
                csv_name = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_name:
                    return "❌ Nenhum arquivo CSV encontrado no ZIP."
                with z.open(csv_name[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(file.name)
        state["df_raw"] = df
        df_cleaned = preprocess_data(df)
        state["df_processed"] = df_cleaned
        return "✅ Arquivo carregado e pré-processado com sucesso."
    except Exception as e:
        return f"❌ Erro ao ler arquivo: {str(e)}"

def preprocess_data(df):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=np.number).columns
    if REMOVE_OUTLIERS:
        z_scores = np.abs(zscore(df_clean[num_cols], nan_policy='omit'))
        filter_mask = (z_scores < 3).all(axis=1)
        df_clean = df_clean.loc[filter_mask]
    if STANDARDIZE_DATA:
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    return df_clean

def sanitize_text(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return nfkd_form.encode('ASCII', 'ignore').decode('ASCII')

def create_prompt(question):
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    cols_info = "\n".join([f"- {col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)])
    sample = df.head(5).to_string()
    stats = df.describe(include='all').to_string()
    prompt = (f"Você é um analista de dados competente. "
              f"Colunas e tipos:\n{cols_info}\n\n"
              f"Amostra dos dados:\n{sample}\n\n"
              f"Estatísticas:\n{stats}\n\nPergunta: {question}\n")
    return sanitize_text(prompt)

def analyze_frequency_and_outliers(df):
    text_summary = ""
    imgs = []

    for col in df.columns:
        if df[col].dtype == 'object' or len(df[col].unique()) < 20:
            counts = df[col].value_counts()
            most_freq = counts.idxmax()
            least_freq = counts.idxmin()
            text_summary += f"Coluna {col}: valor mais frequente '{most_freq}' com {counts[most_freq]} ocorrências; menos frequente '{least_freq}' com {counts[least_freq]} ocorrências.\n"

            fig, ax = plt.subplots(figsize=(6,3))
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_title(f'Frequência variável {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(np.array(img))
        else:
            text_summary += f"Coluna {col}: variável contínua.\n"
            fig, ax = plt.subplots(figsize=(6,3))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Boxplot variável {col}')
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            imgs.append(np.array(img))
    return text_summary, imgs

def detect_cols_and_tasks(question):
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    lower_q = question.lower()
    cols_mentioned = [col for col in df.columns if col.lower() in lower_q]

    graphic_terms = ["distribuicao", "distribuição", "histograma", "histogram", "grafico", "gráfico", "plot", "frequencia", "visualizacao", "visualização"]
    scatter_terms = ["scatter", "dispersao", "dispersão"]
    heatmap_terms = ["heatmap", "correlacao", "correlação", "matriz"]
    cluster_terms = ["cluster", "clusters", "clusterizacao", "clusterização", "k-means", "pca", "agrupamento"]
    freq_terms = ["frequente", "frequência", "pouco frequente", "valor mais comum", "valor menos comum", "outlier", "valores extremos", "boxplot"]

    wants_plots = any(term in lower_q for term in graphic_terms)
    wants_scatter = any(term in lower_q for term in scatter_terms)
    wants_heatmap = any(term in lower_q for term in heatmap_terms)
    wants_cluster = any(term in lower_q for term in cluster_terms)
    wants_freq_analysis = any(term in lower_q for term in freq_terms)

    if wants_plots and not cols_mentioned:
        num_cols = list(df.select_dtypes(include=np.number).columns)
        cols_mentioned = num_cols[:5]

    return cols_mentioned, wants_scatter, wants_heatmap, wants_cluster, wants_freq_analysis

def generate_distribution_plots(cols):
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    imgs = []
    for col in cols:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6,4))
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=True, bins=30, ax=ax)
        else:
            sns.countplot(x=df[col], ax=ax)
            ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Distribuicao de {col}")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        imgs.append(np.array(img))
    return imgs

def generate_scatter_plot():
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    num_cols = list(df.select_dtypes(include=np.number).columns)
    imgs = []
    for i in range(min(len(num_cols)-1, 3)):
        col_x, col_y = num_cols[i], num_cols[i+1]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df[col_x], df[col_y], alpha=0.5)
        ax.set_title(f'Dispersao entre {col_x} e {col_y}')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        imgs.append(np.array(img))
    return imgs

def generate_correlation_heatmap():
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    num_cols = list(df.select_dtypes(include=np.number).columns)
    if not num_cols:
        return []
    corr_df = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr_df, annot=True, ax=ax, cmap='coolwarm')
    ax.set_title('Heatmap de Correlacao')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return [np.array(img)]

def generate_cluster_plot():
    df = state["df_processed"] if state["df_processed"] is not None else state["df_raw"]
    num_cols = list(df.select_dtypes(include=np.number).columns)
    if len(num_cols) < 2:
        return []
    X = df[num_cols].dropna()
    if X.shape[0] < 10:
        return []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6,4))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title("Clusterizacao K-Means com PCA")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return [np.array(img)]

def ask_groq(question):
    client = state["client"]
    if client is None:
        client = init_groq()
        state["client"] = client
        if client is None:
            return "❌ API Groq não configurada.", []
    if state["df_raw"] is None:
        return "❌ Nenhum arquivo carregado.", []
    cols, wants_scatter, wants_heatmap, wants_cluster, wants_freq = detect_cols_and_tasks(question)
    prompt = create_prompt(question)
    messages = [
        {"role": "system", "content": "Voce e um analista de dados muito detalhado."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.2
        )
        text = response.choices[0].message.content
        imgs = []
        if wants_freq:
            freq_text, freq_imgs = analyze_frequency_and_outliers(state["df_processed"] if state["df_processed"] is not None else state["df_raw"])
            text += "\n\n" + freq_text
            imgs += freq_imgs
        if cols:
            imgs += generate_distribution_plots(cols)
        if wants_scatter:
            imgs += generate_scatter_plot()
        if wants_heatmap:
            imgs += generate_correlation_heatmap()
        if wants_cluster:
            imgs += generate_cluster_plot()
        return text, imgs
    except Exception as e:
        return f"❌ Erro Groq API: {str(e)}", []

def process(file, question):
    load_msg = load_file(file)
    if "❌" in load_msg:
        return load_msg, []
    return ask_groq(question)

demo = gr.Blocks()
with demo:
    gr.Markdown("# 🤖 Agente Autônomo Inteligente para Análise de Dados")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload CSV/ZIP", file_types=['.csv','.zip'])
            question_in = gr.Textbox(label="Faça sua pergunta", placeholder="Ex: Analisar distribuições, frequência, outliers, scatter plot, heatmap ou cluster")
            btn = gr.Button("Analisar e Responder")
        with gr.Column(scale=2):
            answer_out = gr.Markdown(label="Resposta do Agente")
            gallery_out = gr.Gallery(label="Gráficos Gerados")
    btn.click(process, inputs=[file_in, question_in], outputs=[answer_out, gallery_out])

if __name__ == "__main__":
    demo.launch()
