"""
Transform 18 Jupyter notebooks by adding Passo 2-10 structure.
"""

import json
import uuid
import copy

BASE = 'D:/repos/projetos/ml_trials_algorithm/notebooks/'


def new_id():
    return str(uuid.uuid4())[:8]


def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }


PASSO_9_MD = """## Passo 9 — Avaliar e registrar 3 insights

**Insight 1 — [Título do Insight]**
> [Descreva aqui o insight mais relevante observado no comportamento deste algoritmo com estes dados.]

**Insight 2 — [Título do Insight]**
> [Descreva aqui o segundo insight: diferença treino vs validação, comportamento dos hiperparâmetros, etc.]

**Insight 3 — [Título do Insight]**
> [Descreva aqui o terceiro insight: comparação com outras abordagens, limitações, ou pontos de atenção.]"""

PASSO_10_CLASSIF = """data_comparativo = {
    'Conjunto': ['Treino (Default)', 'Validação (Default)', 'Treino (Tunado)', 'Validação (Tunado)', 'Teste (Final)'],
    'Acurácia': [acuracia_train_def, acuracia_val_def, '-', '-', accuracia_test],
    'Precisão': [precision_train_def, precision_val_def, '-', '-', precisao_test],
    'Recall':   [recall_train_def, recall_val_def, '-', '-', recall_test],
    'F1-Score': [f1score_train_def, f1score_val_def, '-', '-', f1score_test],
}
df_comparativo = pd.DataFrame(data_comparativo)
print("\\n--- Quadro Comparativo — Diagnóstico Completo ---")
print(df_comparativo.to_string(index=False))"""

PASSO_10_REGRESS = """data_comparativo = {
    'Conjunto': ['Treino (Default)', 'Validação (Default)', 'Treino (Tunado)', 'Validação (Tunado)', 'Teste (Final)'],
    'R²':    [r2_train_def,   r2_val_def,   '-', '-', r2_test],
    'RMSE':  [rmse_train_def, rmse_val_def, '-', '-', rmse_test],
    'MAE':   [mae_train_def,  mae_val_def,  '-', '-', mae_test],
    'MAPE':  [f'{mape_train_def*100:.2f}%', f'{mape_val_def*100:.2f}%', '-', '-', f'{mape_test*100:.2f}%'],
}
df_comparativo = pd.DataFrame(data_comparativo)
print("\\n--- Quadro Comparativo — Diagnóstico Completo ---")
print(df_comparativo.to_string(index=False))"""


# ─── PASSO 2 code for each notebook ────────────────────────────────────────

PASSO2_KNN = """# Instanciar o modelo com parâmetros default
model_def = KNeighborsClassifier()

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_DT_CLASSIF = """# Instanciar o modelo com parâmetros default
model_def = DecisionTreeClassifier(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_LR_CLASSIF = """# Instanciar o modelo com parâmetros default
model_def = LogisticRegression(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_RF_CLASSIF = """# Instanciar o modelo com parâmetros default
model_def = RandomForestClassifier(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_LIN_REG = """# Instanciar o modelo com parâmetros default
model_def = LinearRegression()

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_LASSO = """# Instanciar o modelo com parâmetros default
model_def = Lasso()

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_RIDGE = """# Instanciar o modelo com parâmetros default
model_def = Ridge()

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_ELASTICNET = """# Instanciar o modelo com parâmetros default
model_def = ElasticNet()

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_DT_REG = """# Instanciar o modelo com parâmetros default
model_def = DecisionTreeRegressor(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_RF_REG = """# Instanciar o modelo com parâmetros default
model_def = RandomForestRegressor(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_POLY = """# Instanciar o modelo com parâmetros default (grau 1)
model_def = Pipeline([
    ('features', PolynomialFeatures(degree=1)),
    ('model', LinearRegression())
])

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_POLY_LASSO = """# Instanciar o modelo com parâmetros default (grau 1)
model_def = Pipeline([
    ('features', PolynomialFeatures(degree=1)),
    ('model', Lasso())
])

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_POLY_RIDGE = """# Instanciar o modelo com parâmetros default (grau 1)
model_def = Pipeline([
    ('features', PolynomialFeatures(degree=1)),
    ('model', Ridge())
])

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_POLY_EN = """# Instanciar o modelo com parâmetros default (grau 1)
model_def = Pipeline([
    ('features', PolynomialFeatures(degree=1)),
    ('model', ElasticNet())
])

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_XGBOOST = """# Instanciar o modelo com parâmetros default
model_def = XGBRegressor(random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

PASSO2_LGBM = """# Instanciar o modelo com parâmetros default
model_def = LGBMRegressor(verbosity=-1, random_state=42)

# Treinar com dados de treino
model_def.fit(X_train, y_train.values.ravel())

# Predições no treino e na validação
yhat_train_def = model_def.predict(X_train)
yhat_val_def   = model_def.predict(X_val)"""

# ─── PASSO 3 (train metrics) ────────────────────────────────────────────────

PASSO3_CLASSIF = """# Métricas no conjunto de TREINO com parâmetros default
acuracia_train_def  = mt.accuracy_score(y_train, yhat_train_def)
precision_train_def = mt.precision_score(y_train, yhat_train_def)
recall_train_def    = mt.recall_score(y_train, yhat_train_def)
f1score_train_def   = mt.f1_score(y_train, yhat_train_def)

print("--- Performance no Treino (Default) ---")
print(f"Acurácia:  {acuracia_train_def:.4f}")
print(f"Precisão:  {precision_train_def:.4f}")
print(f"Recall:    {recall_train_def:.4f}")
print(f"F1-Score:  {f1score_train_def:.4f}")"""

PASSO3_REGRESS = """# Métricas no conjunto de TREINO com parâmetros default
r2_train_def   = mt.r2_score(y_train, yhat_train_def)
mse_train_def  = mt.mean_squared_error(y_train, yhat_train_def)
rmse_train_def = np.sqrt(mse_train_def)
mae_train_def  = mt.mean_absolute_error(y_train, yhat_train_def)
mape_train_def = mt.mean_absolute_percentage_error(y_train, yhat_train_def)

print("--- Performance no Treino (Default) ---")
print(f"R²:   {r2_train_def:.4f}")
print(f"MSE:  {mse_train_def:.2f}")
print(f"RMSE: {rmse_train_def:.2f}")
print(f"MAE:  {mae_train_def:.2f}")
print(f"MAPE: {mape_train_def * 100:.2f}%")"""

# ─── PASSO 4 (val metrics) ──────────────────────────────────────────────────

PASSO4_CLASSIF = """# Métricas no conjunto de VALIDAÇÃO com parâmetros default
acuracia_val_def  = mt.accuracy_score(y_val, yhat_val_def)
precision_val_def = mt.precision_score(y_val, yhat_val_def)
recall_val_def    = mt.recall_score(y_val, yhat_val_def)
f1score_val_def   = mt.f1_score(y_val, yhat_val_def)

print("--- Performance na Validação (Default) ---")
print(f"Acurácia:  {acuracia_val_def:.4f}")
print(f"Precisão:  {precision_val_def:.4f}")
print(f"Recall:    {recall_val_def:.4f}")
print(f"F1-Score:  {f1score_val_def:.4f}")"""

PASSO4_REGRESS = """# Métricas no conjunto de VALIDAÇÃO com parâmetros default
r2_val_def   = mt.r2_score(y_val, yhat_val_def)
mse_val_def  = mt.mean_squared_error(y_val, yhat_val_def)
rmse_val_def = np.sqrt(mse_val_def)
mae_val_def  = mt.mean_absolute_error(y_val, yhat_val_def)
mape_val_def = mt.mean_absolute_percentage_error(y_val, yhat_val_def)

print("--- Performance na Validação (Default) ---")
print(f"R²:   {r2_val_def:.4f}")
print(f"MSE:  {mse_val_def:.2f}")
print(f"RMSE: {rmse_val_def:.2f}")
print(f"MAE:  {mae_val_def:.2f}")
print(f"MAPE: {mape_val_def * 100:.2f}%")"""

# ─── PASSO 5 (linear regression special - no hyperparams) ───────────────────

PASSO5_NO_HYPERPARAMS = """# Este algoritmo (Regressão Linear) não possui hiperparâmetros livres para ajuste.
# O modelo já foi treinado com seus parâmetros default no Passo 2.
print("Regressão Linear: sem hiperparâmetros para ajustar.")
print("Os resultados do Passo 2–4 já representam o modelo final antes do reteste.")"""

# ─── PASSO 6 ────────────────────────────────────────────────────────────────

PASSO6_SUPERVISED = """# Unir treino + validação para formar o conjunto final de treinamento
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

print(f"X_train_final shape: {X_train_final.shape}")
print(f"y_train_final shape: {y_train_final.shape}")"""

# ─── PASSO 2/3/4/6/7/8 for clustering ────────────────────────────────────

PASSO2_KMEANS = """# Instanciar o KMeans com parâmetros default
kmeans_def = KMeans(random_state=42)

# Treinar com todos os dados (clusterização não tem conjunto de validação separado)
labels_def = kmeans_def.fit_predict(df.values)
n_clusters_def = len(set(labels_def))

print("--- Treino com Parâmetros Default ---")
print(f"Número de clusters encontrados: {n_clusters_def}")"""

PASSO3_KMEANS = """# Performance do modelo default
sil_def = mt.silhouette_score(df.values, labels_def) if len(set(labels_def)) > 1 else float('nan')

print("--- Performance no Treino (Default) ---")
print(f"Clusters encontrados: {n_clusters_def}")
print(f"Silhouette Score: {sil_def:.4f}")"""

PASSO4_KMEANS = """# Dataset de clusterização não possui conjunto de validação separado
print("Dataset de clusterização não possui conjunto de validação separado.")
print("A avaliação é feita sobre o conjunto de treinamento completo.")"""

PASSO6_KMEANS = """# Dataset de clusterização não possui validação separada - usando todos os dados
X_cluster = df.values
print("Usando todos os dados para o retreino final.")
print(f"Shape: {X_cluster.shape}")"""

PASSO7_KMEANS = """# Retreinar com os melhores parâmetros encontrados no Passo 5 (K=3 como exemplo)
melhor_k = 3  # ajustar conforme resultado do Passo 5

kmeans_final = KMeans(n_clusters=melhor_k, init='k-means++', n_init=10, random_state=42)
df['cluster'] = kmeans_final.fit_predict(X_cluster)

print(f"--- Retreino com melhor K={melhor_k} ---")
print(df['cluster'].value_counts().sort_index())"""

PASSO8_KMEANS = """# Performance final com o modelo tunado
labels_final = kmeans_final.labels_
sil_final = mt.silhouette_score(X_cluster, labels_final)

print("--- Performance Final (Modelo Tunado) ---")
print(f"Clusters: {melhor_k}")
print(f"Silhouette Score: {sil_final:.4f}")
print(f"Inércia (WCSS): {kmeans_final.inertia_:.2f}")"""

PASSO10_KMEANS = """data_comparativo = {
    'Configuração':       ['Default', 'Tunado (K=' + str(melhor_k) + ')'],
    'N° de Clusters':     [n_clusters_def, melhor_k],
    'Silhouette Score':   [sil_def, sil_final],
}
df_comparativo = pd.DataFrame(data_comparativo)
print("\\n--- Quadro Comparativo — Diagnóstico Completo ---")
print(df_comparativo.to_string(index=False))"""

PASSO2_AP = """# Instanciar o AffinityPropagation com parâmetros default
ap_def = AffinityPropagation(random_state=42)

# Treinar com todos os dados (clusterização não tem conjunto de validação separado)
labels_def = ap_def.fit_predict(df.values)
n_clusters_def = len(ap_def.cluster_centers_indices_)

print("--- Treino com Parâmetros Default ---")
print(f"Número de clusters encontrados: {n_clusters_def}")"""

PASSO3_AP = """# Performance do modelo default
if 1 < n_clusters_def < len(df):
    sil_def = mt.silhouette_score(df.values, labels_def)
else:
    sil_def = float('nan')

print("--- Performance no Treino (Default) ---")
print(f"Clusters encontrados: {n_clusters_def}")
print(f"Silhouette Score: {sil_def:.4f}" if not (sil_def != sil_def) else "Silhouette Score: N/A (clusters insuficientes)")"""

PASSO4_AP = """# Dataset de clusterização não possui conjunto de validação separado
print("Dataset de clusterização não possui conjunto de validação separado.")
print("A avaliação é feita sobre o conjunto de treinamento completo.")"""

PASSO6_AP = """# Dataset de clusterização não possui validação separada - usando todos os dados
X_cluster = df.values
print("Usando todos os dados para o retreino final.")
print(f"Shape: {X_cluster.shape}")"""

PASSO7_AP = """# Retreinar com a melhor preferência encontrada no Passo 5
mediana_sim = np.median(mt.pairwise.euclidean_distances(X_cluster) * -1)
melhor_pref = mediana_sim  # ajustar conforme resultado do Passo 5

ap_final = AffinityPropagation(preference=melhor_pref, damping=0.6, max_iter=1000, random_state=42)
labels_final = ap_final.fit_predict(X_cluster)
n_clusters_final = len(ap_final.cluster_centers_indices_)

print(f"--- Retreino com preferência={melhor_pref:.2f} ---")
print(f"Clusters encontrados: {n_clusters_final}")"""

PASSO8_AP = """# Performance final com o modelo tunado
if 1 < n_clusters_final < len(X_cluster):
    sil_final = mt.silhouette_score(X_cluster, labels_final)
else:
    sil_final = float('nan')

print("--- Performance Final (Modelo Tunado) ---")
print(f"Clusters: {n_clusters_final}")
if sil_final == sil_final:  # not NaN
    print(f"Silhouette Score: {sil_final:.4f}")
else:
    print("Silhouette Score: N/A (clusters insuficientes)")"""

PASSO10_AP = """sil_def_str   = f"{sil_def:.4f}" if sil_def == sil_def else "N/A"
sil_final_str = f"{sil_final:.4f}" if sil_final == sil_final else "N/A"

data_comparativo = {
    'Configuração':     ['Default', 'Tunado'],
    'N° de Clusters':   [n_clusters_def, n_clusters_final],
    'Silhouette Score': [sil_def_str, sil_final_str],
}
df_comparativo = pd.DataFrame(data_comparativo)
print("\\n--- Quadro Comparativo — Diagnóstico Completo ---")
print(df_comparativo.to_string(index=False))"""


def transform_supervised_classif(nb_data, passo2_code, has_retrain_cell=False, existing_cells_after5=None):
    """
    Build new cell list for a supervised classification notebook.
    existing_cells_after5: list of cells from index 6 onwards (original)
    has_retrain_cell: whether there's an explicit retrain cell before final test
    """
    original = existing_cells_after5
    # For classification:
    # - cell[0] = grid search (Passo 5)
    # - If has_retrain_cell: cell[1] = retrain+val metrics (Passo 7), cell[2] = final test (Passo 8)
    # - Else: cell[1] = final test (Passo 8), Passo 7 needs to be added

    new_cells = []
    # Passo 2
    new_cells.append(md_cell("## Passo 2 — Treino com parâmetros default"))
    new_cells.append(code_cell(passo2_code))
    # Passo 3
    new_cells.append(md_cell("## Passo 3 — Performance no treino (default)"))
    new_cells.append(code_cell(PASSO3_CLASSIF))
    # Passo 4
    new_cells.append(md_cell("## Passo 4 — Performance na validação (default)"))
    new_cells.append(code_cell(PASSO4_CLASSIF))
    # Passo 5 (grid search - existing cell[0])
    new_cells.append(md_cell("## Passo 5 — Ajuste de hiperparâmetros"))
    new_cells.append(original[0])
    # Passo 6
    new_cells.append(md_cell("## Passo 6 — União treino + validação"))
    new_cells.append(code_cell(PASSO6_SUPERVISED))

    if has_retrain_cell:
        # existing[1] = retrain cell (Passo 7), existing[2] = final test (Passo 8)
        new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
        new_cells.append(original[1])
        new_cells.append(md_cell("## Passo 8 — Performance no teste"))
        new_cells.append(original[2])
    else:
        # existing[1] = final test (Passo 8), no separate retrain cell
        # Passo 7 header + existing final test has concat already, so we skip adding Passo 6 code duplicate
        # The final test cell already has concat in it, so Passo 7 just needs a header
        new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
        # For KNN and DT that don't have an explicit retrain cell,
        # the final test cell already contains concat + retrain + test
        # We'll keep the final test cell as Passo 8 and add a note for Passo 7
        passo7_note = "# Retreino realizado diretamente no Passo 8 com melhores parâmetros encontrados no Passo 5."
        new_cells.append(code_cell(passo7_note))
        new_cells.append(md_cell("## Passo 8 — Performance no teste"))
        new_cells.append(original[1])

    # Passo 9
    new_cells.append(md_cell(PASSO_9_MD))
    # Passo 10
    new_cells.append(md_cell("## Passo 10 — Quadro Comparativo — Diagnóstico Completo"))
    new_cells.append(code_cell(PASSO_10_CLASSIF))

    return new_cells


def transform_supervised_regress(nb_data, passo2_code, has_retrain_cell=False, existing_cells_after5=None, no_hyperparams=False):
    """
    Build new cell list for a supervised regression notebook.
    """
    original = existing_cells_after5

    new_cells = []
    # Passo 2
    new_cells.append(md_cell("## Passo 2 — Treino com parâmetros default"))
    new_cells.append(code_cell(passo2_code))
    # Passo 3
    new_cells.append(md_cell("## Passo 3 — Performance no treino (default)"))
    new_cells.append(code_cell(PASSO3_REGRESS))
    # Passo 4
    new_cells.append(md_cell("## Passo 4 — Performance na validação (default)"))
    new_cells.append(code_cell(PASSO4_REGRESS))
    # Passo 5
    new_cells.append(md_cell("## Passo 5 — Ajuste de hiperparâmetros"))
    if no_hyperparams:
        new_cells.append(code_cell(PASSO5_NO_HYPERPARAMS))
    else:
        new_cells.append(original[0])
    # Passo 6
    new_cells.append(md_cell("## Passo 6 — União treino + validação"))
    new_cells.append(code_cell(PASSO6_SUPERVISED))

    if no_hyperparams:
        # Linear regression: no grid search, the only "extra" cell is the retrain + test
        # original[0] = existing cell that already does baseline (but we use it as Passo 8 test)
        # Actually for linear regression, original[0] is the baseline cell (now replaced by Passo 2)
        # original[1] is the final test cell
        new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
        passo7_note = "# Regressão Linear não possui hiperparâmetros. Retreino com mesmos parâmetros default."
        new_cells.append(code_cell(passo7_note))
        new_cells.append(md_cell("## Passo 8 — Performance no teste"))
        new_cells.append(original[1])
    elif has_retrain_cell:
        # existing[1] = retrain cell (Passo 7), existing[2] = final test (Passo 8)
        new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
        new_cells.append(original[1])
        new_cells.append(md_cell("## Passo 8 — Performance no teste"))
        new_cells.append(original[2])
    else:
        new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
        new_cells.append(code_cell("# Retreino realizado diretamente no Passo 8 com melhores parâmetros encontrados no Passo 5."))
        new_cells.append(md_cell("## Passo 8 — Performance no teste"))
        new_cells.append(original[1])

    # Passo 9
    new_cells.append(md_cell(PASSO_9_MD))
    # Passo 10
    new_cells.append(md_cell("## Passo 10 — Quadro Comparativo — Diagnóstico Completo"))
    new_cells.append(code_cell(PASSO_10_REGRESS))

    return new_cells


def transform_clustering(nb_data, passo2_code, passo3_code, passo4_code, passo6_code,
                          passo7_code, passo8_code, passo10_code, existing_cells_after5=None):
    """Build new cell list for a clustering notebook."""
    original = existing_cells_after5

    new_cells = []
    new_cells.append(md_cell("## Passo 2 — Treino com parâmetros default"))
    new_cells.append(code_cell(passo2_code))
    new_cells.append(md_cell("## Passo 3 — Performance (default)"))
    new_cells.append(code_cell(passo3_code))
    new_cells.append(md_cell("## Passo 4 — Performance na validação (default)"))
    new_cells.append(code_cell(passo4_code))
    new_cells.append(md_cell("## Passo 5 — Ajuste de hiperparâmetros"))
    # existing[0] is the search cell
    new_cells.append(original[0])
    new_cells.append(md_cell("## Passo 6 — União treino + validação"))
    new_cells.append(code_cell(passo6_code))
    new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
    new_cells.append(code_cell(passo7_code))
    new_cells.append(md_cell("## Passo 8 — Performance no teste"))
    new_cells.append(code_cell(passo8_code))
    new_cells.append(md_cell(PASSO_9_MD))
    new_cells.append(md_cell("## Passo 10 — Quadro Comparativo — Diagnóstico Completo"))
    new_cells.append(code_cell(passo10_code))

    return new_cells


def process_notebook(filename, new_cells_after5):
    path = BASE + filename
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Keep cells 0-5 unchanged
    original_first6 = nb['cells'][:6]

    # Build new cell list
    nb['cells'] = original_first6 + new_cells_after5

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  Processed: {filename} ({len(nb['cells'])} total cells)")


def main():
    print("=== Transforming notebooks ===\n")

    # ── 1. KNN (classification) ──────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_knn.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid_search, final_test]
    new_cells = transform_supervised_classif(nb, PASSO2_KNN, has_retrain_cell=False, existing_cells_after5=original)
    process_notebook('ensaios_ml_knn.ipynb', new_cells)

    # ── 2. DecisionTree (classification) ────────────────────────────────────
    with open(BASE + 'ensaios_ml_decisiontree.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid_search, final_test]
    new_cells = transform_supervised_classif(nb, PASSO2_DT_CLASSIF, has_retrain_cell=False, existing_cells_after5=original)
    process_notebook('ensaios_ml_decisiontree.ipynb', new_cells)

    # ── 3. LogisticRegression (classification) ──────────────────────────────
    with open(BASE + 'ensaios_ml_logistc_regression.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid_search, retrain_cell, final_test]
    new_cells = transform_supervised_classif(nb, PASSO2_LR_CLASSIF, has_retrain_cell=True, existing_cells_after5=original)
    process_notebook('ensaios_ml_logistc_regression.ipynb', new_cells)

    # ── 4. RandomForest (classification) ────────────────────────────────────
    with open(BASE + 'ensaios_ml_randomforest.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    # original[6..9]: [grid1, grid2, retrain, final_test]
    # Passo 5 should have both grid cells; we'll merge them or keep first as main search
    # Actually the spec says "KEPT AS-IS for Passo 5 (the grid search cell/s)"
    # For randomforest there are 2 grid search cells - we'll keep both under Passo 5
    original = nb['cells'][6:]  # [grid1, grid2, retrain, final_test]
    # Custom handling: 2 grid search cells
    cells6 = original[0]  # first grid
    cells7 = original[1]  # second grid
    cells8 = original[2]  # retrain
    cells9 = original[3]  # final_test

    new_cells = []
    new_cells.append(md_cell("## Passo 2 — Treino com parâmetros default"))
    new_cells.append(code_cell(PASSO2_RF_CLASSIF))
    new_cells.append(md_cell("## Passo 3 — Performance no treino (default)"))
    new_cells.append(code_cell(PASSO3_CLASSIF))
    new_cells.append(md_cell("## Passo 4 — Performance na validação (default)"))
    new_cells.append(code_cell(PASSO4_CLASSIF))
    new_cells.append(md_cell("## Passo 5 — Ajuste de hiperparâmetros"))
    new_cells.append(cells6)
    new_cells.append(cells7)
    new_cells.append(md_cell("## Passo 6 — União treino + validação"))
    new_cells.append(code_cell(PASSO6_SUPERVISED))
    new_cells.append(md_cell("## Passo 7 — Retreino com melhores parâmetros"))
    new_cells.append(cells8)
    new_cells.append(md_cell("## Passo 8 — Performance no teste"))
    new_cells.append(cells9)
    new_cells.append(md_cell(PASSO_9_MD))
    new_cells.append(md_cell("## Passo 10 — Quadro Comparativo — Diagnóstico Completo"))
    new_cells.append(code_cell(PASSO_10_CLASSIF))
    process_notebook('ensaios_ml_randomforest.ipynb', new_cells)

    # ── 5. LinearRegression (regression, no hyperparams) ────────────────────
    with open(BASE + 'ensaios_ml_linear_regression.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [baseline_cell, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_LIN_REG, has_retrain_cell=False,
                                              existing_cells_after5=original, no_hyperparams=True)
    process_notebook('ensaios_ml_linear_regression.ipynb', new_cells)

    # ── 6. Lasso (regression) ───────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_linear_regression_lasso.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_LASSO, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_linear_regression_lasso.ipynb', new_cells)

    # ── 7. Ridge (regression) ───────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_linear_regression_ridge.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_RIDGE, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_linear_regression_ridge.ipynb', new_cells)

    # ── 8. ElasticNet (regression) ──────────────────────────────────────────
    with open(BASE + 'ensaios_ml_linear_regression_elastic_net.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_ELASTICNET, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_linear_regression_elastic_net.ipynb', new_cells)

    # ── 9. DecisionTreeRegressor ─────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_decision_tree_regression.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_DT_REG, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_decision_tree_regression.ipynb', new_cells)

    # ── 10. RandomForestRegressor ────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_random_forest_regression.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_RF_REG, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_random_forest_regression.ipynb', new_cells)

    # ── 11. Polynomial (LinearRegression) ───────────────────────────────────
    with open(BASE + 'ensaios_ml_polynomial_regressor.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_POLY, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_polynomial_regressor.ipynb', new_cells)

    # ── 12. Polynomial + Lasso ───────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_polynomial_regressor_lasso.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_POLY_LASSO, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_polynomial_regressor_lasso.ipynb', new_cells)

    # ── 13. Polynomial + Ridge ───────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_polynomial_regressor_ridge.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_POLY_RIDGE, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_polynomial_regressor_ridge.ipynb', new_cells)

    # ── 14. Polynomial + ElasticNet ──────────────────────────────────────────
    with open(BASE + 'ensaios_ml_polynomial_regressor_elastic_net.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_POLY_EN, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_polynomial_regressor_elastic_net.ipynb', new_cells)

    # ── 15. XGBoost ──────────────────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_xgboost.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_XGBOOST, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_xgboost.ipynb', new_cells)

    # ── 16. LightGBM ─────────────────────────────────────────────────────────
    with open(BASE + 'ensaios_ml_lgboost.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [grid, retrain, final_test]
    new_cells = transform_supervised_regress(nb, PASSO2_LGBM, has_retrain_cell=True,
                                              existing_cells_after5=original, no_hyperparams=False)
    process_notebook('ensaios_ml_lgboost.ipynb', new_cells)

    # ── 17. KMeans (clustering) ───────────────────────────────────────────────
    with open(BASE + 'ensaios_clustering_kmeans.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [search_cell, retrain_cell, empty_cell]
    # Use only first cell as Passo 5 (search), ignore empty cell
    kmeans_original = [original[0]]
    new_cells = transform_clustering(nb,
        passo2_code=PASSO2_KMEANS,
        passo3_code=PASSO3_KMEANS,
        passo4_code=PASSO4_KMEANS,
        passo6_code=PASSO6_KMEANS,
        passo7_code=PASSO7_KMEANS,
        passo8_code=PASSO8_KMEANS,
        passo10_code=PASSO10_KMEANS,
        existing_cells_after5=kmeans_original
    )
    process_notebook('ensaios_clustering_kmeans.ipynb', new_cells)

    # ── 18. AffinityPropagation (clustering) ─────────────────────────────────
    with open(BASE + 'ensaios_clustering_ap.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    original = nb['cells'][6:]  # [search_cell]
    new_cells = transform_clustering(nb,
        passo2_code=PASSO2_AP,
        passo3_code=PASSO3_AP,
        passo4_code=PASSO4_AP,
        passo6_code=PASSO6_AP,
        passo7_code=PASSO7_AP,
        passo8_code=PASSO8_AP,
        passo10_code=PASSO10_AP,
        existing_cells_after5=original
    )
    process_notebook('ensaios_clustering_ap.ipynb', new_cells)

    print("\n=== All 18 notebooks processed successfully ===")


if __name__ == '__main__':
    main()
