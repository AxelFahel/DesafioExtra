---
title: Agente Autônomo de Análise de Dados
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Sistema inteligente para análise, clusterização e visualização automática de dados CSV.
---

# 🤖 Agente Autônomo de Análise de Dados

Este Space apresenta um agente inteligente para exploração e análise de arquivos CSV, capaz de responder perguntas em linguagem natural sobre o dataset, gerar gráficos automaticamente e executar análises de clusterização e correlação. Basta fazer upload de seu arquivo CSV (ou ZIP com CSV), perguntar, e visualizar resultados textuais e gráficos dinâmicos!

## Funcionalidades

- **Análise textual via Groq API**: respostas automáticas e contextualizadas sobre o seu dataset.
- **Visualização gráfica**: histogramas, scatter plots, heatmaps de correlação e clusters K-Means com PCA.
- **Clusterização e análise estatística**: identifica agrupamentos, outliers e tendências centrais.
- **100% genérico**: funciona com qualquer CSV tabular carregado.

## Como usar

### Usando no Hugging Face Spaces

1. Crie um Space e faça upload dos arquivos `app.py` e `requirements.txt`.
2. Configure o segredo (secret) `API_KEY_GROQ` no painel Settings com sua chave Groq API.
3. O Space irá automaticamente construir e disponibilizar sua aplicação.
4. Acesse o link público para usar a interface.

### Executando localmente

1. Clone este repositório.
2. Instale as dependências com:  
