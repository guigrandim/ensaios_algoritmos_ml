# 🧪 Ensaios de Algoritmos de Machine Learning — comparação sistemática de tuning

Repositório de ensaios experimentais comparando algoritmos de **Classificação**, **Regressão** e **Clusterização**, medindo como cada um reage a variações nos principais parâmetros de controle de overfitting e underfitting.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E?logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

<p align="center">
<img src="./assets/img/fluxo.png" alt="Ensaios de Algoritmos de Machine Learning" width="800px">
</p>

### 🎯 Destaques
- Ensaiei **17 algoritmos de ML** (4 de classificação, 11 de regressão, 2 de clusterização), consolidados em **7 tabelas comparativas** de performance sobre treino, validação e teste.
- **Random Forest** atingiu **F1-Score de 0.9591** na classificação, superando o KNN (0.5647) em quase o dobro — evidência de fronteiras de decisão não-lineares no dataset.
- Apliquei tuning sistemático de hiperparâmetros (grid manual + observação de overfitting/underfitting) em cada algoritmo, documentando o melhor conjunto de parâmetros por caso.
- Stack: Python, scikit-learn, XGBoost, LightGBM, Jupyter Notebooks.

---

## 🚨 Problema de Negócio

**Contexto:** A Data Money é uma consultoria de Ciência de Dados cuja proposta de valor central é a **expertise no treinamento e ajuste fino de algoritmos**. Para sustentar esse diferencial competitivo com os clientes, a equipe precisa de evidências práticas — não apenas teóricas — sobre como cada algoritmo se comporta em diferentes configurações de parâmetros.

**Pergunta central:** Como a performance de cada algoritmo muda à medida que os principais parâmetros de controle de overfitting e underfitting são ajustados, e qual configuração generaliza melhor para dados nunca vistos?

**Minha tarefa:** Conduzi os ensaios experimentais ponta a ponta — treinar cada algoritmo com parâmetros default, medir performance em treino e validação, buscar a melhor configuração de hiperparâmetros, retreinar com o conjunto completo e validar a performance final em teste, documentando tudo em notebooks reprodutíveis.

---

## 🗺️ Planejamento da Solução

**Produto final:** 7 tabelas comparativas mostrando a performance dos algoritmos, avaliadas com múltiplas métricas, sobre 3 conjuntos de dados (Treinamento, Validação e Teste).

1. **Treinar** os algoritmos com os dados de treinamento utilizando parâmetros default.
2. **Medir a performance** no conjunto de treinamento (parâmetros default).
3. **Medir a performance** no conjunto de validação (parâmetros default).
4. **Ajustar os hiperparâmetros** para encontrar o conjunto que minimize overfitting/underfitting.
5. **Unir** os dados de treinamento e validação.
6. **Retreinar** o algoritmo com a união dos dados e os melhores parâmetros encontrados.
7. **Medir a performance final** no conjunto de teste (dados nunca vistos).
8. **Consolidar** um quadro comparativo — análise diagnóstica completa do ensaio.

**Ferramentas:** Python 3.10+, scikit-learn, XGBoost, LightGBM, Jupyter Notebooks.

---

## 🛠️ Desenvolvimento

### Datasets

| Categoria | Dataset | Amostras (treino/val/teste) | Features | Target |
|-----------|---------|------------------------------|----------|--------|
| **Classificação** | Airline Passenger Satisfaction | 129.487 (72.515 / 31.079 / 25.893) | 24 (dados demográficos, voo e avaliações de serviço) | `satisfaction` — binário |
| **Regressão** | Spotify Song Popularity | 18.835 (10.547 / 4.521 / 3.767) | 13 (atributos de áudio da API do Spotify) | `song_popularity` — contínuo (0–100) |
| **Clusterização** | Wine Dataset | 178 (sem split — não supervisionado) | 13 (propriedades químicas do vinho) | Nenhum |

<details>
<summary>Features — Classificação (Airline Passenger Satisfaction)</summary>

`customer_type`, `age`, `class`, `flight_distance`, `inflight_wifi_service`, `departure_arrival_time_convenient`, `ease_of_online_booking`, `gate_location`, `food_and_drink`, `online_boarding`, `seat_comfort`, `inflight_entertainment`, `on_board_service`, `leg_room_service`, `baggage_handling`, `checkin_service`, `inflight_service`, `cleanliness`, `departure_delay_in_minutes`, `arrival_delay_in_minutes`, `gender_Female`, `gender_Male`, `type_of_travel_business_travel`, `type_of_travel_personal_travel`

</details>

<details>
<summary>Features — Regressão (Spotify Song Popularity)</summary>

`song_duration_ms`, `acousticness`, `danceability`, `energy`, `instrumentalness`, `key`, `liveness`, `loudness`, `audio_mode`, `speechiness`, `tempo`, `time_signature`, `audio_valence`

</details>

<details>
<summary>Features — Clusterização (Wine Dataset)</summary>

`alcohol`, `malic_acid`, `ash`, `ash_alcanity`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280`, `proline`

</details>

### Algoritmos Avaliados

| Categoria | Algoritmos | Métricas de Performance |
|-----------|-----------|------------------------|
| **Classificação** | KNN, Decision Tree, Random Forest, Logistic Regression | Accuracy, Precision, Recall, F1-Score |
| **Regressão** | Linear Regression (Lasso, Ridge, ElasticNet), Polynomial (Lasso, Ridge, ElasticNet), Decision Tree, Random Forest, XGBoost, LightGBM | R², MSE, RMSE, MAE, MAPE |
| **Clusterização** | K-Means, Affinity Propagation | Silhouette Score |

### Estrutura do Projeto

```
ml_trials_algorithm/
├── notebooks/
│   ├── classificacao/       # KNN, Decision Tree, Logistic Regression, Random Forest
│   ├── regressao/           # Linear, Lasso, Ridge, ElasticNet, Polynomial, Decision Tree, Random Forest, XGBoost, LightGBM
│   └── clusterizacao/       # KMeans, Affinity Propagation
├── dataset/
│   ├── classification_datasets/
│   │   ├── a_traninig/      # X_training.csv, y_training.csv
│   │   ├── b_validation/    # X_validation.csv, y_validation.csv
│   │   └── c_test/          # X_test.csv, y_test.csv
│   ├── regression_datasets/
│   │   ├── a_traninig/
│   │   ├── b_validation/
│   │   └── c_test/
│   └── clusters_datasets/
│       └── a_traning/       # X_dataset.csv (sem split — unsupervised)
└── requirements.txt
```

### Como Executar Localmente

```bash
git clone https://github.com/guigrandim/ensaios_algoritimos_ml.git
cd ensaios_algoritimos_ml
pip install -r requirements.txt
jupyter notebook
```

Ou abra qualquer notebook diretamente no Colab:

<details>
<summary>Notebooks — Classificação</summary>

| Notebook | Abrir |
|----------|-------|
| KNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_knn.ipynb) |
| Decision Tree | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_decisiontree.ipynb) |
| Random Forest | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_randomforest.ipynb) |
| Logistic Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_logistc_regression.ipynb) |

</details>

<details>
<summary>Notebooks — Regressão</summary>

| Notebook | Abrir |
|----------|-------|
| Linear Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression.ipynb) |
| Linear Regression — Lasso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_lasso.ipynb) |
| Linear Regression — Ridge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_ridge.ipynb) |
| Linear Regression — ElasticNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_elastic_net.ipynb) |
| Polynomial Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor.ipynb) |
| Polynomial Regressor — Lasso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_lasso.ipynb) |
| Polynomial Regressor — Ridge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_ridge.ipynb) |
| Polynomial Regressor — ElasticNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_elastic_net.ipynb) |
| Decision Tree Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_decision_tree_regression.ipynb) |
| Random Forest Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_random_forest_regression.ipynb) |
| XGBoost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_xgboost.ipynb) |
| LightGBM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_lightgbm.ipynb) |

</details>

<details>
<summary>Notebooks — Clusterização</summary>

| Notebook | Abrir |
|----------|-------|
| KMeans | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/clusterizacao/ensaios_clustering_kmeans.ipynb) |
| Affinity Propagation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/clusterizacao/ensaios_clustering_ap.ipynb) |

</details>

---

## 💡 Top Insights

### 1. 🌳 Modelos baseados em árvore dominam a classificação
Esperava-se que Regressão Logística, por ser mais simples, tivesse desempenho competitivo. Na prática, **Random Forest** (F1 = 0.9591) e **Decision Tree** (F1 = 0.9489) superaram a Logistic Regression (F1 = 0.8501) por uma margem larga — sinal de que o dataset tem fronteiras de decisão não-lineares bem capturadas por métodos ensemble.

### 2. 📉 KNN sofreu com a maldição da dimensionalidade
Com 24 features, o KNN teve o pior desempenho entre os classificadores (F1 = 0.5647, menos da metade do Random Forest), reforçando que distância euclidiana perde poder discriminativo em espaços de alta dimensão sem redução prévia.

### 3. 🔍 R² baixo na regressão revela baixo poder preditivo das features atuais
Mesmo o melhor modelo de regressão (XGBoost) explicou apenas **36.78%** da variância do target (R² = 0.3678). Isso não é falha do algoritmo — modelos lineares e não-lineares convergiram para a mesma limitação, indicando que as 13 features de áudio do Spotify não capturam sozinhas os fatores que determinam a popularidade de uma música.

### 4. ⚖️ Regularização (Lasso/Ridge/ElasticNet) não fez diferença nos modelos lineares
Linear Regression, Lasso, Ridge e ElasticNet convergiram todos para R² ≈ 0.051 no teste — evidência de que o gargalo é a linearidade da relação entre features e target, não overfitting, já que penalizar os coeficientes não alterou o resultado.

### 5. ⚡ KMeans supera Affinity Propagation em qualidade e custo
KMeans com K=3 (Silhouette = 0.2330) superou a melhor configuração de Affinity Propagation (7 clusters, Silhouette = 0.2023) — e é computacionalmente mais barato, já que AP escala O(n²) enquanto KMeans escala linearmente com o número de amostras.

---

## 📊 Resultados

### Resultado da Entrega
O ensaio produziu um **guia interno reprodutível de tuning** para 17 algoritmos, cobrindo as 3 famílias de problemas (classificação, regressão, clusterização) mais comuns nas consultorias da Data Money. Cada notebook documenta o passo a passo default → validação → tuning → teste, permitindo que a equipe reaproveite a metodologia e os melhores hiperparâmetros encontrados em projetos futuros com dados semelhantes, sem repetir o processo de busca do zero.

### Classificação — métricas no conjunto de teste

| Algoritmo | Melhores Parâmetros | Acurácia | Precisão | Recall | F1-Score |
|-----------|---------------------|----------|----------|--------|----------|
| **Random Forest** | `max_depth=19`, `n_estimators=100` | **0.9646** | **0.9730** | **0.9456** | **0.9591** |
| **Decision Tree** | `max_depth=14` | 0.9554 | 0.9551 | 0.9427 | 0.9489 |
| **Logistic Regression** | `C=1.0`, `solver=lbfgs`, `max_iter=100` | 0.8711 | 0.8685 | 0.8324 | 0.8501 |
| KNN | `k=4` | 0.6742 | 0.6828 | 0.4815 | 0.5647 |

### Regressão — métricas no conjunto de teste

| Algoritmo | Melhores Parâmetros | R² | RMSE | MAE | MAPE |
|-----------|---------------------|----|------|-----|------|
| **XGBoost** | `n_estimators=500`, `max_depth=7`, `lr=0.1` | **0.3678** | **17.54** | **12.68** | 616.91% |
| **LightGBM** | `n_estimators=500`, `num_leaves=63`, `lr=0.1`, `min_child=10` | 0.3579 | 17.68 | 12.75 | 632.43% |
| Random Forest Regressor | `n_estimators=300`, `max_depth=10` | 0.2462 | 19.16 | 15.26 | 715.39% |
| Polynomial | `degree=2` | 0.0909 | 21.04 | 16.74 | 827.70% |
| Decision Tree Regressor | `max_depth=5` | 0.0896 | 21.05 | 16.83 | 788.62% |
| Polynomial ElasticNet | `degree=2`, `alpha=0.001`, `l1_ratio=0.8` | 0.0886 | 21.07 | 16.76 | 833.11% |
| Polynomial Lasso | `degree=2`, `alpha=0.01` | 0.0854 | 21.10 | 16.79 | 834.21% |
| Polynomial Ridge | `degree=2`, `alpha=1.0` | 0.0854 | 21.10 | 16.79 | 834.21% |
| Linear Regression | sem hiperparâmetros | 0.0512 | 21.49 | 17.14 | 853.14% |
| Lasso | `alpha=0.001` | 0.0511 | 21.49 | 17.14 | 853.31% |
| ElasticNet | `alpha=0.001`, `l1_ratio=0.5` | 0.0511 | 21.49 | 17.14 | 853.71% |
| Ridge | `alpha=10.0` | 0.0511 | 21.49 | 17.14 | 853.78% |

### Clusterização — avaliação por Silhouette Score

**KMeans — Grid Search por K**

| K | WCSS (Inércia) | Silhouette Score |
|---|----------------|-----------------|
| 2 | 1017.83 | 0.2132 |
| **3** | **829.04** | **0.2330** |
| 4 | 750.15 | 0.2166 |
| 5 | 681.87 | 0.1868 |
| 6 | 622.37 | 0.2203 |
| 7 | 568.97 | 0.2108 |
| 8 | 533.33 | 0.1865 |
| 9 | 505.89 | 0.1867 |
| 10 | 477.39 | 0.1746 |

**Affinity Propagation — Variação de Preferência**

| Preferência | Clusters Encontrados | Silhouette Score |
|-------------|---------------------|-----------------|
| **-50** | **7** | **0.2023** |
| -100 | 4 | 0.1588 |
| -200 | 3 | 0.1957 |
| -300 | 2 | 0.1762 |
| -500 | 1 | N/A |
| -700 | 1 | N/A |
| -1000 | 1 | N/A |

### Comparativo Final por Categoria

| Categoria | Melhor Algoritmo | Métrica Principal |
|-----------|-----------------|-------------------|
| Classificação | **Random Forest** | F1-Score = 0.9591 |
| Regressão | **XGBoost** | R² = 0.3678 / RMSE = 17.54 |
| Clusterização | **KMeans (K=3)** | Silhouette = 0.2330 |

---

## ✅ Conclusões

Os ensaios confirmam a hipótese de partida da Data Money: o ajuste fino de hiperparâmetros muda materialmente o resultado, mas seu impacto depende do quanto o algoritmo já é adequado ao formato dos dados — tuning não compensa a escolha de um algoritmo mal alinhado com a estrutura do problema (caso do KNN e dos modelos lineares na regressão). O guia produzido documenta empiricamente onde tuning agrega valor e onde o gargalo é outro (features, algoritmo, dados).

**Próximos passos:**
- [ ] 🔧 Testar **XGBoost e LightGBM** para classificação — potencial de superar Random Forest com tuning adequado.
- [ ] 🔎 Análise de importância de features para identificar variáveis mais relevantes em cada dataset.
- [ ] ⚖️ Investigar desbalanceamento de classes na classificação — avaliar se SMOTE ou `class_weight` melhoram o recall.
- [ ] 🧪 Explorar **feature engineering** na regressão — criação de features derivadas para capturar relações não-lineares.
- [ ] 🌐 Testar **DBSCAN** e **Hierarchical Clustering** como alternativas ao KMeans/AP.

**Limitações:** os datasets usados são fixos e públicos (não são dados reais de cliente da Data Money), o split treino/validação/teste é único (sem cross-validation), e o dataset de regressão tem poder preditivo limitado nas features disponíveis — o R² baixo reflete essa limitação dos dados, não necessariamente dos algoritmos.

---

*Fonte de dados: Airline Passenger Satisfaction, Spotify Song Popularity e Wine Dataset (públicos) · Metodologia: treino → validação → tuning → reunião treino+validação → teste · Stack: Python, scikit-learn, XGBoost, LightGBM*

## 🧰 Skills Demonstradas

- Diagnóstico de overfitting/underfitting via comparação treino × validação
- Tuning sistemático de hiperparâmetros por família de algoritmo (árvores, lineares regularizados, ensembles, boosting, clustering)
- Avaliação com métricas apropriadas por tipo de problema (Accuracy/Precision/Recall/F1, R²/RMSE/MAE/MAPE, Silhouette Score)
- Leitura crítica de resultados — reconhecer quando o gargalo é o algoritmo vs. quando é o dado

## 👩‍💻 Autor

Desenvolvido por Guilherme Grandim como um projeto de portfólio em Ciência de Dados / Machine Learning.

## 📄 Licença

Este projeto está sob a licença MIT — veja [LICENSE](./LICENSE) para detalhes.
