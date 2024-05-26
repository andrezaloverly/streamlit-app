import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configurar a página para largura total
st.set_page_config(layout="wide")

# Criar duas colunas com proporções diferentes
col0, col01 = st.columns([1, 6])  #

with col0:
    st.image("image.png", use_column_width=True)

with col01:
    # Título e descrição
    st.markdown("<h1 style='text-align: center; padding-top: 80px;'>Regressão Linear Monovariada/Simples</h1>", unsafe_allow_html=True)

# Adicionar uma linha divisória
st.markdown("<hr>", unsafe_allow_html=True)

# Dividir a página em duas colunas
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        ### O que é:
        A regressão linear monovariada é uma técnica estatística utilizada para modelar a relação entre duas variáveis: uma variável dependente 𝑌 e uma variável independente 𝑋. 
        O objetivo é prever o valor de 𝑌 com base no valor de 𝑋.
                
        ### Utilização:
        A regressão linear monovariada é utilizada para prever tendências, analisar dados e fazer inferências.
        """)


with col2:
    st.markdown("""
        ### Modelo:
        O modelo de regressão linear simples é representado pela equação:
        Yi = α + β Xi + ϵi
        Onde:
        - Yi = é o valor da variável dependente para a observação 𝑖;
        - α = é o intercepto da reta;
        - β = é a inclinação da reta;
        - Xi = é o valor da variável independente para a observação 𝑖;
        - ϵi = é o erro aleatório.
        """)

# Adicionar uma linha divisória
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Analisando os resultados:</h3>", unsafe_allow_html=True)

# Dividir a página em duas colunas
col3, col4 = st.columns(2)

with col3:
    # Upload do arquivo de dados
    uploaded_file = st.file_uploader("Carregar arquivo Excel", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        # Adicionar uma caixa de seleção para exibir os dados
    if st.checkbox("Mostrar dados carregados"):
        st.write(data)
    # Mostrar nomes das colunas e permitir seleção de X e Y
    columns = data.columns.tolist()
    X_column = st.selectbox("Selecione a coluna X (independente)", columns)
    Y_column = st.selectbox("Selecione a coluna Y (dependente)", columns)

    if X_column and Y_column:
        # Formatação dos dados
        data[X_column] = data[X_column].astype(str).str.replace(',', '.').astype(float)
        data[Y_column] = data[Y_column].astype(str).str.replace(',', '.').astype(float)
        data = data.dropna(subset=[X_column, Y_column])

        # Definir as variáveis X e Y
        X = data[X_column].values
        Y = data[Y_column].values

        # Calcular as médias de X e Y
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        # Calcular os coeficientes da regressão
        SXY = np.sum((X - X_mean) * (Y - Y_mean))
        SXX = np.sum((X - X_mean) ** 2)
        beta = SXY / SXX
        alpha = Y_mean - beta * X_mean

        # Fazer as previsões
        Y_pred = alpha + beta * X


with col4:
    if uploaded_file is not None:
        # Plotar os dados e a reta de regressão
        fig, ax = plt.subplots()
        ax.scatter(X, Y, color='blue', label='Dados observados')
        ax.plot(X, Y_pred, color='red', label='Reta de Regressão')
        ax.set_xlabel(X_column)
        ax.set_ylabel(Y_column)
        ax.legend()
        ax.set_title(f'Regressão Linear: {X_column} vs {Y_column}')
        st.pyplot(fig)

        # Exibir os coeficientes
        st.write(f'Intercepto (alpha): {alpha}')
        st.write(f'Coeficiente (beta): {beta}')

with col3:
    if uploaded_file is not None:
        # Analisando os resultados
        st.markdown(f"""              
        **Intercepto (alpha):**
        O intercepto 𝛼 representa o valor de {Y_column} quando {X_column} é zero.

        **Coeficiente (beta):**
        O coeficiente 𝛽 indica a inclinação da reta de regressão.

        **Gráfico de Dispersão e Reta de Regressão:**
        O gráfico de dispersão com a reta de regressão ilustra visualmente a relação entre as duas variáveis. Os pontos azuis representam os dados observados, enquanto a linha vermelha representa a reta de regressão ajustada.

        **Conclusão:**
        Os resultados da regressão linear monovariada indicam a relação entre {X_column} e {Y_column}.
        """)

