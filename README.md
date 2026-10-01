# Student Performance ML 🎓

 ## Projeto Integrador — UC 10 | Ciência de Dados

 Projeto de Machine Learning para prever o desempenho acadêmico de estudantes, classificando-os como **Aprovado** ou **Reprovado**, com base em hábitos de estudo e características de estilo de vida.
 Dashboard Streamlit: https://studentml-dashboard.streamlit.app/

 ## 📊 Dataset

 **Student Performance & Study Habits Dataset — Kaggle**

 Coloque o arquivo CSV em:

```
desempenho_estudantil/student_performance_dataset.csv
```

 ## 🛠️ Tecnologias

 - Python
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn

 ## 🚀 Como rodar

 ### 1\. Criar o ambiente virtual

 **Windows:**

```
python -m venv venv
venv\Scripts\activate
```

 **Linux/macOS:**

```
python3 -m venv venv
source venv/bin/activate
```

 ### 2\. Instalar as dependências

```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

 ### 3\. Colocar o dataset

 Salve o arquivo baixado do Kaggle em:

```
desempenho_estudantil/student_performance_dataset.csv
```

 ### 4\. Executar os scripts

 Execute na seguinte ordem:

```
python traducao_dos_dados.py
python dados_tratados.py
python aprendizado_machine.py
```

 ### 5\. Conferir os resultados

 Os gráficos e tabelas gerados ficam na pasta:

```
reports/
```

 **Principais arquivos:**

 - `comparacao_modelos.csv` — resultados dos modelos;
- `comparacao_modelos.png` — comparação dos modelos;
- `pca_2d.png` — visualização do PCA;
- `pca_variancia.png` — variância explicada;
- `cm_pca*_k*.png` — matrizes de confusão.

 O terminal também apresenta o **melhor modelo**, seu desempenho e uma **previsão para um novo estudante**.

 ## 🤖 Modelos utilizados

 O projeto testa combinações de:

 - **PCA:** 2, 3, 4 e 5 componentes;
- **KNN:** `k = 3`, `5` e `7`;
- **SMOTE:** balanceamento aplicado somente aos dados de treinamento;
- **F1-macro:** utilizado para selecionar o modelo final.

 ## 👥 Autores

 **Projeto Integrador — UC 10**\
 **Curso de Ciência de Dados**

- Napoleão Júnior
- Tainara Almeida
- Júlia Stefany 
