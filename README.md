---
title: Agente Autônomo de Análise de Dados com Groq
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Sistema inteligente para análise, clusterização e visualização automática de dados CSV.
---

# 🤖 Agente Autônomo de Análise de Dados com Groq

Este Space apresenta um agente inteligente para exploração e análise de arquivos CSV, capaz de responder perguntas em linguagem natural sobre o dataset, gerar gráficos automaticamente e executar análises de clusterização e correlação. Basta fazer upload de seu arquivo CSV (ou ZIP com CSV), perguntar, e visualizar resultados textuais e gráficos dinâmicos!

## Funcionalidades

- **Análise textual via Groq API**: respostas automáticas e contextualizadas sobre o seu dataset.
- **Visualização gráfica**: histogramas, scatter plots, heatmaps de correlação e clusters K-Means com PCA.
- **Clusterização e análise estatística**: identifica agrupamentos, outliers e tendências centrais.
- **100% genérico**: funciona com qualquer CSV tabular carregado.

## Como usar

1. Faça upload do seu arquivo no campo `Upload CSV/ZIP`.
2. Digite qualquer pergunta sobre os dados (exemplos abaixo).
3. Visualize a resposta textual e os gráficos produzidos.

> **Importante:** Para usar os recursos da Groq API, configure a chave via Secrets pelo painel Settings (`API_KEY_GROQ`). Não exponha sua chave no código!

## Exemplos de perguntas

- Qual a distribuição da variável Amount?
- Existem clusters claros neste dataset? (Gera gráfico PCA + KMeans)
- Mostre um heatmap de correlação.
- Quais são as médias e medianas das variáveis principais?
- Quais são as conclusões gerais sobre os dados?

## Segurança

- Nenhuma chave API é armazenada ou exposta no frontend.
- Para configuração segura da chave Groq, use o botão **Secrets** do Space.

## Deploy & Créditos

Este app foi implementado em Python usando [Gradio](https://gradio.app/), [Groq](https://groq.com/) e [Hugging Face Spaces](https://huggingface.co/spaces).

Para deploy próprio, basta clonar este repositório e configurar seu segredo `API_KEY_GROQ`.

## Licença

MIT

