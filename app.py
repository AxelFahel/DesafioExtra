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

# Usar variável de ambiente segura para a chave
GROQ_API_KEY = os.environ.get("API_KEY_GROQ", "")

state = {
    "df": None,
    "client": None,
}

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
        state["df"] = df
        return "✅ Arquivo carregado com sucesso."
    except Exception as e:
        return f"❌ Erro ao ler arquivo: {str(e)}"

def sanitize_text(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    return nfkd_form.encode('ASCII', 'ignore').decode('ASCII')

def create_prompt(question):
    df = state["df"]
    cols_info = "\n".join([f"- {col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)])
    sample = df.head(5).to_string()
    stats = df.describe(include='all').to_string()
    prompt = (f"Você é um analista de dados competente. "
              f"Colunas e tipos:\n{cols_info}\n\n"
              f"Amostra dos dados:\n{sample}\n\n"
              f"Estatísticas:\n{stats}\n\nPergunta: {question}\n")
    return sanitize_text(prompt)

def detect_tasks(question):
    df = state["df"]
    q = question.lower()
    # Colunas citadas
    cols = [col for col in df.columns if col.lower() in q]
    # Termos autônomos para funções
    wants_plots = any(t in q for t in ["distribuicao", "distribuição", "histograma", "histogram", "grafico", "gráfico", "plot", "frequencia", "visualizacao", "visualização"])
    wants_scatter = any(t in q for t in ["scatter", "dispersao", "dispersão"])
    wants_heatmap = any(t in q for t in ["heatmap", "correlacao", "correlação", "matriz"])
    wants_cluster = any(t in q for t in ["cluster", "clusters", "clusterizacao", "k-means", "pca", "agrupamento"])
    if wants_plots and not cols:
        num_cols = list(df.select_dtypes(include=np.number).columns)
        cols = num_cols[:5]
    return cols, wants_scatter, wants_heatmap, wants_cluster

def generate_distribution_plots(cols):
    df = state["df"]
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
    df = state["df"]
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
    df = state["df"]
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
    df = state["df"]
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
        if client is None:
            return "❌ API Groq não configurada.", []
    if state["df"] is None:
        return "❌ Nenhum arquivo carregado.", []
    cols, wants_scatter, wants_heatmap, wants_cluster = detect_tasks(question)
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
    gr.Markdown("# 🤖 Agente Autonomo Genérico para Análise de Dados com Groq")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload CSV/ZIP", file_types=['.csv','.zip'])
            question_in = gr.Textbox(label="Faça sua pergunta", placeholder="Ex: Analisar distribuições, scatter plot, heatmap ou cluster")
            btn = gr.Button("Analisar e Responder")
        with gr.Column(scale=2):
            answer_out = gr.Markdown(label="Resposta do Agente")
            gallery_out = gr.Gallery(label="Gráficos Gerados")
    btn.click(process, inputs=[file_in, question_in], outputs=[answer_out, gallery_out])

if __name__ == "__main__":
    demo.launch()
