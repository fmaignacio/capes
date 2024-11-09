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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os  
import gdown  

nltk.download('punkt')

# Configuração inicial da página
st.set_page_config(
    page_title="Busca Semântica - Teses e Dissertações",
    page_icon = "🔎",
    layout="wide"
)

# Carregar os dados logo no início
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder')


# @st.cache_data
# def carregar_dados():
#     df = pd.read_csv('df_termos_30k_retom.csv', encoding='utf-8')
#     embeddings = np.load('embeddings_serafim-100m_30_retom.npy')
#     return df, embeddings

FILES = {
    "df_termos_30k_retom.pkl": "1CJ_ApY8wZ77lwE7ZnHwJCgJNsBHCEOIe",
    "embeddings_serafim-100m_30_retom.npy": "1qI2WqqIRC0S_YiV9sCJkvYh8EKlGtig9"
}
1qI2WqqIRC0S_YiV9sCJkvYh8EKlGtig9
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
df_termos = dados["df_termos_30k_retom.pkl"]
embeddings = dados["embeddings_serafim-100m_30_retom.npy"]

st.write("Dados e embeddings carregados com sucesso!")

# Carrega os dados globalmente
try:
    modelo = carregar_modelo()
    df = dados["df_termos_30k_retom.pkl"]
    embeddings = dados["embeddings_serafim-100m_30_retom.npy"]
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

#Função extrair_palavras_chave usando spaCy e TF-IDF
def extrair_palavras_chave(texto, num_palavras=5):
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(texto)

    stop_words = list(spacy.lang.pt.stop_words.STOP_WORDS)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([token.text for token in doc])

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    top_indices = scores.argsort()[-num_palavras:][::-1]
    return [feature_names[i] for i in top_indices]

# #Função extrair_palavras_chave usando YAKE
# def extrair_palavras_chave(texto, num_palavras=5):
#    kw_extractor = yake.KeywordExtractor(n=1, top=num_palavras)
#    keywords = kw_extractor.extract_keywords(texto)
#    return [kw[0] for kw in keywords]
#
# #Função extrair_palavras_chave usando Gensim
# def extrair_palavras_chave(texto, num_palavras=5):
#     return keywords(texto, words=num_palavras, split=True)
#
#
# def gerar_resumo_topicos(resultados, prompt, num_estudos=5):
#     # Selecionar os primeiros 'num_estudos' estudos
#     estudos_selecionados = resultados.head(num_estudos)
#
#     # Primeiro passo: Concatenar todos os termos em uma única string
#     todos_termos = []
#     for termos_str, palavras_chave in zip(estudos_selecionados['TERMOS_STR'], estudos_selecionados['DS_PALAVRA_CHAVE']):
#         # Garantir que os valores sejam strings e fazer o split
#         termos_str = str(termos_str)
#         palavras_chave = str(palavras_chave)
#
#         # Concatenar os termos e palavras-chave
#         termos_concatenados = termos_str + ";" + palavras_chave
#         todos_termos.append(termos_concatenados)
#
#     # Juntar todos os termos em uma única string
#     string_completa = ";".join(todos_termos)
#
#     # Segundo passo: Separar em lista e limpar
#     lista_termos = string_completa.split(";")
#
#     # Terceiro passo: Limpar espaços em branco
#     lista_termos = [termo.strip() for termo in lista_termos if termo.strip()]
#
#     # Quarto passo: Remover duplicados mantendo a ordem
#     termos_unicos = list(dict.fromkeys(lista_termos))
#
#     # Gerar o resumo dos tópicos
#     resumo_topicos = f"Com base na busca '{prompt}', os {num_estudos} primeiros artigos mostram os seguintes tópicos:\n"
#     resumo_topicos += ", ".join(termos_unicos)
#
#     return resumo_topicos


from collections import defaultdict
from heapq import nlargest


from collections import defaultdict
from heapq import nlargest

from collections import defaultdict
from heapq import nlargest

from collections import defaultdict
from heapq import nlargest


def gerar_resumo_topicos(resultados, prompt, num_estudos=10):
    """
    Gera resumo dos tópicos principais usando a coluna SINGLE_KEYS.

    Args:
        resultados: DataFrame com os resultados da busca
        prompt: String com a consulta original
        num_estudos: Número de estudos a considerar
    """
    # Selecionar os primeiros estudos
    estudos_selecionados = resultados.head(num_estudos)

    # Inicializar dicionário de pontuação com defaultdict
    termos_pontuacao = defaultdict(float)

    # Iterar sobre os estudos selecionados
    for i, (_, estudo) in enumerate(estudos_selecionados.iterrows()):
        # Usar SINGLE_KEYS que já contém termos únicos
        termos = str(estudo['SINGLE_KEYS']).split(';')

        # Calcular peso baseado na posição e similaridade
        posicao_peso = num_estudos - i
        similaridade_peso = float(estudo.get('similaridade', 1.0))  # Usa 1.0 se não houver similaridade
        peso_composto = posicao_peso * similaridade_peso

        # Atribuir pontuação para cada termo
        for termo in termos:
            termo = termo.strip()
            if termo:  # Ignora termos vazios
                termos_pontuacao[termo] += peso_composto

    # Selecionar top termos considerando pontuação
    top_termos = nlargest(10, termos_pontuacao.items(), key=lambda x: x[1])

    # Gerar resumo estruturado
    resumo_topicos = [
        f"Com base na busca '{prompt}', os {num_estudos} estudos mais similares abordam os seguintes tópicos principais:",
        "",  # Linha em branco para melhor formatação
        "Tópicos por relevância:"
    ]

    # Adicionar termos com suas pontuações relativas
    max_pontuacao = top_termos[0][1] if top_termos else 1
    for termo, pontuacao in top_termos:
        relevancia_relativa = (pontuacao / max_pontuacao) * 100
        resumo_topicos.append(f"• {termo} ({relevancia_relativa:.1f}%)")

    return "\n".join(resumo_topicos)


def gerar_grafo_similaridades(resultados, num_estudos=10, threshold=0.7, prompt="Prompt"):
    estudos_selecionados = resultados.head(num_estudos)
    num_docs = len(estudos_selecionados)

    # Criar o grafo
    G = nx.Graph()

    # Adicionar prompt como nó central
    G.add_node(prompt)

    # Adicionar nós dos estudos e conexões com o prompt
    for i, row in enumerate(estudos_selecionados.itertuples()):
        node_name = f"Estudo{i + 1}"
        G.add_node(node_name)
        if row.similaridade >= threshold:
            G.add_edge(prompt, node_name, weight=row.similaridade)

    # Adicionar conexões entre estudos
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            sim = estudos_selecionados.iloc[i]['similaridade']
            if sim >= threshold:
                G.add_edge(f"Estudo{i + 1}", f"Estudo{j + 1}", weight=sim)

    # Desenhar o grafo
    plt.figure(figsize=(10, 8))

    # Definir posição central para o prompt
    center_pos = {prompt: (0, 0)}

    # Calcular layout mantendo o prompt no centro
    pos = nx.spring_layout(
        G,
        k=1,  # Aumentar distância entre nós
        iterations=50,
        seed=42,
        pos=center_pos,
        fixed=[prompt]  # Fixar o prompt na posição central
    )

    # Desenhar nós
    node_colors = ['red' if node == prompt else 'lightblue' for node in G.nodes()]
    node_sizes = [1200 if node == prompt else 800 for node in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes
    )

    # Desenhar arestas com transparência
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.6,
        width=2,
        edge_color='gray'
    )

    # Adicionar labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_family='sans-serif',
        font_weight='bold'
    )

    # Adicionar pesos nas arestas
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
        font_size=8
    )

    plt.title("") #Grafo de Similaridade de Estudos
    plt.axis('off')

    return plt

def main():
    st.title("🔍 Busca Semântica - Teses e Dissertações")

    try:
        # st.write(f"Versão do Python: {sys.version}")
        # st.write(f"Versão do PyTorch: {torch.__version__}")
        # st.write(f"Versão do NumPy: {np.__version__}")
        #
        # st.success(f"Base de dados carregada com {len(df):,} documentos")

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

                    # Gerar insights antes de usar
                    insights = gerar_insights(resultados)  # Adicione esta linha!

                    # Gerar um único resumo com os principais tópicos dos 10 primeiros estudos
                    resumo_topicos = gerar_resumo_topicos(resultados, query)

                    # Primeiro, exibir o resumo dos tópicos em largura total
                    st.subheader("Resumo dos Principais Tópicos")
                    st.write(resumo_topicos)

                    # Resto do código continua igual...

                    # Agora criar as duas colunas principais para o resto do conteúdo
                    col_resultados, col_insights = st.columns([2, 1])

                    # Coluna de resultados
                    with col_resultados:
                        st.subheader("Resultados da Busca")
                        for idx, row in resultados.iterrows():
                            with st.expander(f"{row['NM_PRODUCAO'][:300]}...", expanded=idx == 0):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**Título:** {row['NM_PRODUCAO']}")
                                    st.markdown(f"**Instituição:** {row['SG_ENTIDADE_ENSINO']} ({row['SG_UF_IES']})")
                                    st.markdown(f"**Ano:** {row['AN_BASE']}")
                                    st.markdown(f"**Palavras-chave:** {row['DS_PALAVRA_CHAVE']}")
                                    st.markdown("**Resumo:**")
                                    st.markdown(row['DS_RESUMO_CAP'])

                                with col2:
                                    st.markdown("**Detalhes:**")
                                    st.markdown(f"- **Programa:** {row['NM_PROGRAMA']}")
                                    st.markdown(f"- **Área:** {row['NM_AREA_CONHECIMENTO']}")
                                    st.markdown(f"- **Orientador:** {row['NM_ORIENTADOR']}")
                                    st.markdown(f"- **Similaridade:** {row['similaridade']:.2%}")
                                    if pd.notna(row.get('URL')):
                                        st.markdown(f"[Link para texto completo]({row['URL']})")

                    # Coluna de insights
                    with col_insights:
                        st.subheader("Grafo de Similaridades") #"Insights dos Resultados"

                        # Primeiro o grafo de similaridades
                        # st.write("### Grafo de Similaridades")
                        grafo_similaridades = gerar_grafo_similaridades(resultados)
                        st.pyplot(grafo_similaridades)

                        # Depois os outros gráficos em sequência
                        st.write("### Distribuição por Ano")
                        df_ano = pd.DataFrame(insights['distribuicao_ano']).reset_index()
                        df_ano.columns = ['Ano', 'Quantidade']
                        st.bar_chart(df_ano.set_index('Ano'))

                        st.write("### Distribuição por Grau")
                        df_grau = pd.DataFrame(insights['distribuicao_grau']).reset_index()
                        df_grau.columns = ['Grau', 'Quantidade']
                        st.bar_chart(df_grau.set_index('Grau'))

                        st.write("### Top 10 Áreas")
                        df_area = pd.DataFrame(insights['distribuicao_area']).reset_index()
                        df_area.columns = ['Área', 'Quantidade']
                        st.bar_chart(df_area.set_index('Área'))

    except Exception as e:
        st.error(f"Erro ao executar a aplicação: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
