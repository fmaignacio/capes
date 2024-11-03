import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import torch
import re
import unicodedata
import sys
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# Configuração da página
st.set_page_config(
    page_title="Busca Semântica - Teses e Dissertações",
    layout="wide"
)


# # Carregar os dados logo no início
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder')
#
#
# @st.cache_data
# def carregar_dados():
#     df = pd.read_pickle('df_termos_30k.pkl')
#     embeddings = np.load('embeddings_serafim-100m.npy')
#     return df, embeddings

FILES = {
    "df_termos_30k.pkl": "1BmidzItS-7vsSAuWxzq4RnfUKrqR6gkC",
    "embeddings_serafim-100m.npy": "1Y3QSdraLqyDUWi7H1MhOeczYd5JBxhlX"
}
@st.cache_resource
def carregar_dados():
    os.makedirs("data", exist_ok=True)
    dados = {}

    for filename, file_id in FILES.items():
        local_path = f"data/{filename}"

        # Verifica se o arquivo já existe
        if not os.path.exists(local_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)

        # Carrega o arquivo baseado na extensão
        if filename.endswith('.pkl'):
            dados[filename] = pd.read_pickle(local_path)
        elif filename.endswith('.npy'):
            dados[filename] = np.load(local_path)

    return dados


# Uso dos dados no app
dados = carregar_dados()
df_termos = dados["df_termos_30k.pkl"]
embeddings = dados["embeddings_serafim-100m.npy"]

st.write("Dados e embeddings carregados com sucesso!")

# Carrega os dados globalmente
try:
    modelo = carregar_modelo()
    df = dados["df_termos_30k.pkl"]
    embeddings = dados["embeddings_serafim-100m.npy"]
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()
st.write("Dados e embeddings carregados com sucesso!")

def limpar_texto_robusto(texto):
    if pd.isna(texto):
        return ''
    remove_stopwords = False
    remove_punctuation = True
    texto = unicodedata.normalize('NFKD', str(texto))
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.lower()

    if remove_punctuation:
        texto = re.sub(r'[^a-z\s]', '', texto)

    palavras = texto.split()

    if remove_stopwords:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('portuguese'))
        palavras = [p for p in palavras if p not in stop_words]

    return ' '.join(palavras)


def busca_semantica_faiss(query, modelo, df_processado, top_k=5, filtros_metadado=None):
    """
    Realiza busca semântica usando FAISS com suporte a filtros e mapeamento de IDs
    """
    CAMPOS_FILTRAVEIS = ['AN_BASE', 'NM_GRAU_ACADEMICO', 'SG_ENTIDADE_ENSINO', 'SG_UF_IES']

    query_processada = limpar_texto_robusto(query)
    query_embedding = modelo.encode([query_processada])
    query_embedding = normalize(query_embedding, norm='l2').astype('float32')

    mapeamento_embeddings = pd.DataFrame({
        'posicao_embedding': range(len(embeddings)),
        'ID_ADD_PRODUCAO_INTELECTUAL': df_processado['ID_ADD_PRODUCAO_INTELECTUAL'].values
    })

    df_filtrado = df_processado.copy()
    if filtros_metadado:
        for campo, valor in filtros_metadado.items():
            if campo in CAMPOS_FILTRAVEIS:
                df_filtrado = df_filtrado[df_filtrado[campo] == valor]

    if len(df_filtrado) == 0:
        return pd.DataFrame()

    posicoes_validas = mapeamento_embeddings[
        mapeamento_embeddings['ID_ADD_PRODUCAO_INTELECTUAL'].isin(
            df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL']
        )
    ]['posicao_embedding'].values

    embeddings_filtrados = embeddings[posicoes_validas].astype('float32')

    d = embeddings_filtrados.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings_filtrados)
    index.add(embeddings_filtrados)

    scores, indices = index.search(query_embedding, min(top_k, len(df_filtrado)))

    ids_selecionados = df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL'].values[indices[0]]

    resultados = df_filtrado[
        df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL'].isin(ids_selecionados)
    ].copy()

    resultados = resultados.set_index('ID_ADD_PRODUCAO_INTELECTUAL')
    resultados = resultados.reindex(ids_selecionados)
    resultados['similaridade'] = scores[0]
    resultados = resultados.reset_index()

    return resultados


def busca_semantica_cosine(query, modelo, embeddings, df_processado, top_k=5, filtros_metadado=None):
    """
    Realiza busca semântica mantendo o mapeamento correto entre embeddings e IDs originais.
    """
    CAMPOS_FILTRAVEIS = ['AN_BASE', 'NM_GRAU_ACADEMICO', 'SG_ENTIDADE_ENSINO', 'SG_UF_IES']

    query_processada = limpar_texto_robusto(query)
    query_embedding = modelo.encode([query_processada])
    query_embedding = normalize(query_embedding, norm='l2')

    mapeamento_embeddings = pd.DataFrame({
        'posicao_embedding': range(len(embeddings)),
        'ID_ADD_PRODUCAO_INTELECTUAL': df_processado['ID_ADD_PRODUCAO_INTELECTUAL'].values
    })

    df_filtrado = df_processado.copy()
    if filtros_metadado:
        for campo, valor in filtros_metadado.items():
            if campo in CAMPOS_FILTRAVEIS:
                df_filtrado = df_filtrado[df_filtrado[campo] == valor]

    if len(df_filtrado) == 0:
        return pd.DataFrame()

    posicoes_validas = mapeamento_embeddings[
        mapeamento_embeddings['ID_ADD_PRODUCAO_INTELECTUAL'].isin(
            df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL']
        )
    ]['posicao_embedding'].values

    embeddings_filtrados = embeddings[posicoes_validas]

    similaridades = cosine_similarity(query_embedding, embeddings_filtrados)[0]

    top_k = min(top_k, len(df_filtrado))

    top_indices_local = np.argsort(similaridades)[-top_k:][::-1]

    ids_selecionados = df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL'].values[top_indices_local]

    resultados = df_filtrado[
        df_filtrado['ID_ADD_PRODUCAO_INTELECTUAL'].isin(ids_selecionados)
    ].copy()

    similaridades_ordenadas = similaridades[top_indices_local]
    resultados = resultados.set_index('ID_ADD_PRODUCAO_INTELECTUAL')
    resultados = resultados.reindex(ids_selecionados)
    resultados['similaridade'] = similaridades_ordenadas
    resultados = resultados.reset_index()

    return resultados


def gerar_insights(resultados):
    """
    Gera insights estatísticos a partir dos resultados da busca semântica.
    """
    insights = {}

    # Total de estudos encontrados
    insights['total_estudos'] = len(resultados)

    # Distribuição por ano
    distribuicao_ano = resultados['AN_BASE'].value_counts().sort_index()
    insights['distribuicao_ano'] = distribuicao_ano.head(10)

    # Distribuição por grau acadêmico
    distribuicao_grau = resultados['NM_GRAU_ACADEMICO'].value_counts()
    insights['distribuicao_grau'] = distribuicao_grau

    # Distribuição por área de conhecimento
    distribuicao_area = resultados['NM_AREA_CONHECIMENTO'].value_counts()
    insights['distribuicao_area'] = distribuicao_area.head(10)

    # Distribuição por instituição
    distribuicao_ies = resultados['SG_ENTIDADE_ENSINO'].value_counts()
    insights['distribuicao_ies'] = distribuicao_ies.head(10)

    # Média de similaridade
    insights['media_similaridade'] = resultados['similaridade'].mean()

    # Palavras-chave mais frequentes
    todas_palavras = ' '.join(resultados['DS_PALAVRA_CHAVE'].dropna()).lower()
    palavras_chave = pd.Series(todas_palavras.split(';')).str.strip()
    top_palavras = palavras_chave.value_counts()
    insights['top_palavras_chave'] = top_palavras.head(10)

    return insights


def main():
    st.title("🔍 Busca Semântica em Teses e Dissertações")

    try:
        st.write(f"Versão do Python: {sys.version}")
        st.write(f"Versão do PyTorch: {torch.__version__}")
        st.write(f"Versão do NumPy: {np.__version__}")

        st.success(f"Base de dados carregada com {len(df):,} documentos")

        # Sidebar com filtros
        with st.sidebar:
            st.title("Filtros de Busca")
            filtros = {}

            ano = st.selectbox(
                "Ano Base",
                options=[None] + sorted(df['AN_BASE'].unique().tolist()),
            )
            if ano:
                filtros['AN_BASE'] = ano

            grau = st.selectbox(
                "Grau Acadêmico",
                options=[None] + sorted(df['NM_GRAU_ACADEMICO'].unique().tolist()),
            )
            if grau:
                filtros['NM_GRAU_ACADEMICO'] = grau

            ies = st.selectbox(
                "Instituição de Ensino",
                options=[None] + sorted(df['SG_ENTIDADE_ENSINO'].unique().tolist()),
            )
            if ies:
                filtros['SG_ENTIDADE_ENSINO'] = ies

            uf = st.selectbox(
                "UF",
                options=[None] + sorted(df['SG_UF_IES'].unique().tolist()),
            )
            if uf:
                filtros['SG_UF_IES'] = uf

        # Área principal
        query = st.text_input("Digite sua busca:", placeholder="Ex: inteligência artificial na educação")
        col1, col2 = st.columns([2, 1])
        with col1:
            num_resultados = st.slider("Número de resultados:", min_value=5, max_value=50, value=10)
        with col2:
            # Mantém FAISS como uma opção comentada
            metodo_busca = st.radio(
                "Método de busca:",
                options=['Cosine Similarity'],  # 'FAISS' desabilitado temporariamente
                horizontal=True
            )

        if query:
            with st.spinner('Realizando busca...'):
                if metodo_busca == 'Cosine Similarity':
                    resultados = busca_semantica_cosine(
                        query=query,
                        modelo=modelo,
                        embeddings=embeddings,
                        df_processado=df,
                        top_k=num_resultados,
                        filtros_metadado=filtros
                    )
                # Desabilitado temporariamente o FAISS
                # else:
                #     resultados = busca_semantica_faiss(
                #         query=query,
                #         modelo=modelo,
                #         df_processado=df,
                #         top_k=num_resultados,
                #         filtros_metadado=filtros
                #     )

                if resultados.empty:
                    st.warning("Nenhum resultado encontrado para os filtros selecionados.")
                else:
                    st.success(f"Encontrados {len(resultados)} resultados!")

                    # Dividir a tela em duas colunas
                    col_resultados, col_insights = st.columns([2, 1])

                    # Coluna de resultados
                    with col_resultados:
                        st.subheader("Resultados da Busca")
                        for idx, row in resultados.iterrows():
                            with st.expander(f"{row['NM_PRODUCAO'][:100]}...", expanded=idx == 0):
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.markdown(f"**Instituição:** {row['SG_ENTIDADE_ENSINO']} ({row['SG_UF_IES']})")
                                    st.markdown(f"**Ano:** {row['AN_BASE']}")
                                    st.markdown(f"**Palavras-chave:** {row['DS_PALAVRA_CHAVE']}")
                                    st.markdown("**Resumo:**")
                                    st.markdown(row['DS_RESUMO'])

                                with col2:
                                    st.markdown("**Detalhes:**")
                                    st.markdown(f"- **Programa:** {row['NM_PROGRAMA']}")
                                    st.markdown(f"- **Área:** {row['NM_AREA_CONHECIMENTO']}")
                                    st.markdown(f"- **Orientador:** {row['NM_ORIENTADOR']}")
                                    st.markdown(f"- **Similaridade:** {row['similaridade']:.2%}")

                                    if pd.notna(row.get('DS_URL_TEXTO_COMPLETO')):
                                        st.markdown(f"[Link para texto completo]({row['DS_URL_TEXTO_COMPLETO']})")

                    # Coluna de insights
                    with col_insights:
                        st.subheader("Insights dos Resultados")
                        insights = gerar_insights(resultados)

                        # Total de estudos
                        st.metric("Total de Estudos", insights['total_estudos'])

                        # Gráficos
                        st.markdown("### Distribuição por Ano")
                        df_ano = pd.DataFrame(insights['distribuicao_ano']).reset_index()
                        df_ano.columns = ['Ano', 'Quantidade']
                        st.bar_chart(df_ano.set_index('Ano'))

                        st.markdown("### Distribuição por Grau")
                        df_grau = pd.DataFrame(insights['distribuicao_grau']).reset_index()
                        df_grau.columns = ['Grau', 'Quantidade']
                        st.bar_chart(df_grau.set_index('Grau'))

                        st.markdown("### Top 10 Áreas")
                        df_area = pd.DataFrame(insights['distribuicao_area']).reset_index()
                        df_area.columns = ['Área', 'Quantidade']
                        st.bar_chart(df_area.set_index('Área'))

    except Exception as e:
        st.error(f"Erro ao executar a aplicação: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
