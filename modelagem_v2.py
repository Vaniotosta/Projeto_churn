"""
Etapa 4: variáveis novas + ajuste de parâmetros + modelo final.
Uso: python modelagem_v2.py   (precisa de base_treino.csv e base_teste.csv na mesma pasta)

Regra importante: o modelo final é escolhido pela validação cruzada do TREINO.
O teste só é usado para mostrar o resultado, nunca para escolher.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_utils import criar_features

SEED = 42

# ------------------------------------------------------------------ dados
treino = pd.read_csv("base_treino.csv")
teste = pd.read_csv("base_teste.csv")
colunas_fora = ["id_cliente", "data_referencia", "churn"]
X_treino, y_treino = treino.drop(columns=colunas_fora), treino["churn"]
X_teste, y_teste = teste.drop(columns=colunas_fora), teste["churn"]


def montar(modelo, escalar=False, com_features=True):
    """Pipeline: (variáveis novas) -> (escala e one-hot) -> modelo."""
    pre = ColumnTransformer([
        ("num", StandardScaler() if escalar else "passthrough",
         make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_exclude="number")),
    ])
    passos = [("features", FunctionTransformer(criar_features))] if com_features else []
    return Pipeline(passos + [("pre", pre), ("modelo", modelo)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ------------------------------------------------------------------ candidatos
candidatos = {}

# referência: regressão logística da etapa anterior (sem variáveis novas, sem ajuste)
ref = montar(LogisticRegression(max_iter=2000), escalar=True, com_features=False)
auc_ref = cross_val_score(ref, X_treino, y_treino, cv=cv, scoring="roc_auc", n_jobs=1)
candidatos["Regressão logística (etapa 3)"] = (ref.fit(X_treino, y_treino), auc_ref.mean())

buscas = {
    "Regressão logística ajustada": (
        montar(LogisticRegression(max_iter=2000), escalar=True),
        {"modelo__C": loguniform(1e-3, 1e2)}, 15),
    "Random Forest ajustado": (
        montar(RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)),
        {"modelo__min_samples_leaf": [10, 20, 40, 80],
         "modelo__max_features": ["sqrt", 0.3]}, 4),
    "Gradient Boosting ajustado": (
        montar(HistGradientBoostingClassifier(random_state=SEED)),
        {"modelo__learning_rate": loguniform(0.01, 0.2),
         "modelo__max_leaf_nodes": [6, 10, 15, 31],
         "modelo__min_samples_leaf": [20, 50, 100],
         "modelo__l2_regularization": loguniform(1e-3, 10),
         "modelo__max_iter": [150, 300]}, 12),
}

for nome, (pipe, parametros, n_iter) in buscas.items():
    busca = RandomizedSearchCV(pipe, parametros, n_iter=n_iter, scoring="roc_auc",
                               cv=cv, random_state=SEED, n_jobs=1, refit=True)
    busca.fit(X_treino, y_treino)
    candidatos[nome] = (busca.best_estimator_, busca.best_score_)
    print(f"{nome:32s} AUC (validação cruzada) {busca.best_score_:.3f} | {busca.best_params_}")

# ------------------------------------------------------------------ escolhe pelo treino
melhor = max(candidatos, key=lambda n: candidatos[n][1])
modelo_final = candidatos[melhor][0]
print(f"\nModelo escolhido (maior AUC na validação cruzada): {melhor}")

# ------------------------------------------------------------------ avalia no teste
linhas, probs = [], {}
for nome, (pipe, auc_cv) in candidatos.items():
    p = pipe.predict_proba(X_teste)[:, 1]
    probs[nome] = p
    alvo = p >= np.quantile(p, 0.80)               # 20% de maior risco
    acertos = (alvo & (y_teste == 1)).sum()
    linhas.append({
        "modelo": nome,
        "auc_validacao_cruzada": auc_cv,
        "auc_teste": roc_auc_score(y_teste, p),
        "auc_precisao_recall_teste": average_precision_score(y_teste, p),
        "precisao_top20": acertos / alvo.sum(),
        "recall_top20": acertos / (y_teste == 1).sum(),
        "lift_top20": (acertos / alvo.sum()) / y_teste.mean(),
    })
res = pd.DataFrame(linhas).set_index("modelo").round(3)
res.to_csv("resultados_v2.csv", encoding="utf-8")
print("\n== Resultado no TESTE ==")
print(res.to_string())

# ------------------------------------------------------------------ gráfico de comparação
fig, ax = plt.subplots(figsize=(8, 4.5))
posicao = np.arange(len(res))
ax.barh(posicao - 0.2, res["auc_validacao_cruzada"], height=0.4, label="Validação cruzada (treino)", color="#9bb7d4")
ax.barh(posicao + 0.2, res["auc_teste"], height=0.4, label="Teste", color="#3b6ea8")
ax.set_yticks(posicao)
ax.set_yticklabels(res.index)
ax.set_xlim(0.70, 0.80)
ax.set_xlabel("AUC")
ax.set_title("Comparação dos modelos")
ax.legend(loc="lower right", fontsize=8)
for i, v in enumerate(res["auc_teste"]):
    ax.text(v + 0.001, i + 0.2, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("comparacao_modelos.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ o que pesa na decisão
imp = permutation_importance(modelo_final, X_teste, y_teste, scoring="roc_auc",
                             n_repeats=3, random_state=SEED, n_jobs=1)
imp = pd.Series(imp.importances_mean, index=X_teste.columns).sort_values().tail(10)
plt.figure(figsize=(7, 4.5))
imp.plot(kind="barh", color="#3b6ea8")
plt.xlabel("Queda do AUC se a variável for embaralhada")
plt.title(f"Variáveis mais importantes ({melhor})")
plt.tight_layout()
plt.savefig("importancia_final.png", dpi=130)
plt.close()
print("\nTop 5 variáveis:", imp.sort_values(ascending=False).head(5).round(3).to_dict())

# ------------------------------------------------------------------ lucro da campanha
# ATENÇÃO: hipóteses. Troque pelos números reais da empresa.
VALOR_CLIENTE_ANO = 1200   # R$ de receita anual de um cliente retido
CUSTO_OFERTA = 60          # R$ gasto em cada cliente contactado
TAXA_SUCESSO = 0.30        # % dos churns contactados que aceitam ficar

p = probs[melhor]
y_ord = y_teste.values[np.argsort(-p)]
pontos = []
for pct in range(5, 101, 5):
    n = int(len(p) * pct / 100)
    pontos.append((pct, y_ord[:n].sum() * TAXA_SUCESSO * VALOR_CLIENTE_ANO - n * CUSTO_OFERTA))
lucro = pd.DataFrame(pontos, columns=["pct", "lucro"])
melhor_corte = lucro.loc[lucro["lucro"].idxmax()]
print(f"\nMelhor corte: contactar {melhor_corte.pct:.0f}% dos clientes -> "
      f"lucro de R$ {melhor_corte.lucro:,.0f} (teste, {len(p)} clientes)")
print(f"Contactar todos: R$ {lucro.iloc[-1].lucro:,.0f}")

plt.figure(figsize=(7, 4.5))
plt.plot(lucro["pct"], lucro["lucro"] / 1000, marker="o", color="#3b6ea8")
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(melhor_corte.pct, color="#c0392b", linestyle="--")
plt.xlabel("% dos clientes contactados (do maior para o menor risco)")
plt.ylabel("Lucro da campanha (R$ mil)")
plt.title("Quantos clientes vale a pena contactar?")
plt.tight_layout()
plt.savefig("lucro_final.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ salva tudo que o app precisa
numericas = X_treino.select_dtypes(include="number").columns
categoricas = X_treino.columns.difference(numericas)
padrao = {**X_treino[numericas].median().to_dict(), **{c: X_treino[c].mode()[0] for c in categoricas}}
info = {
    "modelo": melhor,
    "colunas": X_treino.columns.tolist(),
    "padrao": padrao,
    "cortes": {"alto": float(np.quantile(p, 0.80)), "medio": float(np.quantile(p, 0.50))},
    "campanha": {"pct_contactar": int(melhor_corte.pct), "valor_cliente_ano": VALOR_CLIENTE_ANO,
                 "custo_oferta": CUSTO_OFERTA, "taxa_sucesso": TAXA_SUCESSO},
}
with open("info_modelo.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

joblib.dump(modelo_final, "modelo_churn.joblib")
pd.DataFrame({"id_cliente": teste["id_cliente"], "prob_churn": p.round(4), "churn_real": y_teste}) \
    .sort_values("prob_churn", ascending=False).to_csv("previsoes_teste.csv", index=False, encoding="utf-8")
print("\nSalvos: resultados_v2.csv, comparacao_modelos.png, importancia_final.png, lucro_final.png, "
      "info_modelo.json, modelo_churn.joblib, previsoes_teste.csv")
