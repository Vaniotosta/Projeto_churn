"""
Etapa 3: treino e avaliação dos modelos de churn.
Uso: python modelagem_churn.py   (precisa de base_treino.csv e base_teste.csv na mesma pasta)
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
    boosting = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              random_state=42, verbose=-1)
    nome_boosting = "LightGBM"
except ImportError:  # sem LightGBM instalado, usa o equivalente do scikit-learn
    boosting = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    nome_boosting = "Gradient Boosting (sklearn)"

# ------------------------------------------------------------------ dados
treino = pd.read_csv("base_treino.csv")
teste = pd.read_csv("base_teste.csv")

colunas_fora = ["id_cliente", "data_referencia", "churn"]
X_treino, y_treino = treino.drop(columns=colunas_fora), treino["churn"]
X_teste, y_teste = teste.drop(columns=colunas_fora), teste["churn"]

categoricas = X_treino.select_dtypes(exclude="number").columns.tolist()
numericas = X_treino.select_dtypes("number").columns.tolist()


def montar(modelo, escalar=False):
    pre = ColumnTransformer([
        ("num", StandardScaler() if escalar else "passthrough", numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])
    return Pipeline([("pre", pre), ("modelo", modelo)])


modelos = {
    "Regressão logística": montar(LogisticRegression(max_iter=1000), escalar=True),
    "Random Forest": montar(RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                   random_state=42, n_jobs=-1)),
    nome_boosting: montar(boosting),
}

# ------------------------------------------------------------------ 1) compara com validação cruzada (só treino)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("== AUC na validação cruzada (treino) ==")
for nome, pipe in modelos.items():
    auc_cv = cross_val_score(pipe, X_treino, y_treino, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{nome:28s} {auc_cv.mean():.3f} (+/- {auc_cv.std():.3f})")

# ------------------------------------------------------------------ 2) avalia UMA vez no teste
resultados, probs = [], {}
for nome, pipe in modelos.items():
    pipe.fit(X_treino, y_treino)
    p = pipe.predict_proba(X_teste)[:, 1]
    probs[nome] = p

    # campanha: ligar para os 20% de clientes com maior risco
    corte = np.quantile(p, 0.80)
    alvo = p >= corte
    verdadeiros = (alvo & (y_teste == 1)).sum()
    resultados.append({
        "modelo": nome,
        "auc": roc_auc_score(y_teste, p),
        "auc_precisao_recall": average_precision_score(y_teste, p),
        "precisao_top20": verdadeiros / alvo.sum(),
        "recall_top20": verdadeiros / (y_teste == 1).sum(),
        "lift_top20": (verdadeiros / alvo.sum()) / y_teste.mean(),
    })

res = pd.DataFrame(resultados).set_index("modelo").round(3)
res.to_csv("resultados_modelos.csv", encoding="utf-8")
print("\n== Resultado no TESTE ==")
print(res.to_string())

melhor = res["auc"].idxmax()
print(f"\nMelhor modelo: {melhor}")

# ------------------------------------------------------------------ 3) gráfico ROC
plt.figure(figsize=(6, 5))
for nome, p in probs.items():
    fpr, tpr, _ = roc_curve(y_teste, p)
    plt.plot(fpr, tpr, label=f"{nome} (AUC {roc_auc_score(y_teste, p):.2f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chute aleatório")
plt.xlabel("Falsos alarmes (clientes que ficariam, marcados como churn)")
plt.ylabel("Churns encontrados")
plt.title("Curva ROC no teste")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("curva_roc.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ 4) o que mais pesa na decisão
imp = permutation_importance(modelos[melhor], X_teste, y_teste, scoring="roc_auc",
                             n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=X_teste.columns).sort_values().tail(12)
plt.figure(figsize=(7, 5))
imp.plot(kind="barh", color="#3b6ea8")
plt.xlabel("Queda do AUC se a variável for embaralhada")
plt.title(f"Variáveis mais importantes ({melhor})")
plt.tight_layout()
plt.savefig("importancia_variaveis.png", dpi=130)
plt.close()
print("\nTop 5 variáveis:", imp.sort_values(ascending=False).head(5).round(3).to_dict())

# ------------------------------------------------------------------ 5) tradução para dinheiro
# ATENÇÃO: valores abaixo são HIPÓTESES. Troque pelos números reais da empresa.
VALOR_CLIENTE_ANO = 1200   # R$ de receita anual que um cliente retido gera
CUSTO_OFERTA = 60          # R$ gasto em cada cliente contactado (desconto, ligação)
TAXA_SUCESSO = 0.30        # % dos churns contactados que aceitam ficar

p = probs[melhor]
ordem = np.argsort(-p)
y_ord = y_teste.values[ordem]
pontos = []
for pct in range(5, 101, 5):
    n = int(len(p) * pct / 100)
    churns_pegos = y_ord[:n].sum()
    lucro = churns_pegos * TAXA_SUCESSO * VALOR_CLIENTE_ANO - n * CUSTO_OFERTA
    pontos.append((pct, lucro))
lucro_df = pd.DataFrame(pontos, columns=["pct_contactado", "lucro"])
melhor_corte = lucro_df.loc[lucro_df["lucro"].idxmax()]
print(f"\nMelhor corte: contactar {melhor_corte.pct_contactado:.0f}% dos clientes "
      f"-> lucro de R$ {melhor_corte.lucro:,.0f} (base de teste, {len(p)} clientes)")
print(f"Contactar todos: R$ {lucro_df.iloc[-1].lucro:,.0f}")

plt.figure(figsize=(7, 4.5))
plt.plot(lucro_df["pct_contactado"], lucro_df["lucro"] / 1000, marker="o", color="#3b6ea8")
plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(melhor_corte.pct_contactado, color="#c0392b", linestyle="--")
plt.xlabel("% dos clientes contactados (do maior para o menor risco)")
plt.ylabel("Lucro da campanha (R$ mil)")
plt.title("Quantos clientes vale a pena contactar?")
plt.tight_layout()
plt.savefig("lucro_por_corte.png", dpi=130)
plt.close()

# ------------------------------------------------------------------ 6) salva modelo e previsões
joblib.dump(modelos[melhor], "modelo_churn.joblib")
pd.DataFrame({"id_cliente": teste["id_cliente"], "prob_churn": p.round(4), "churn_real": y_teste}) \
    .sort_values("prob_churn", ascending=False) \
    .to_csv("previsoes_teste.csv", index=False, encoding="utf-8")
print("\nSalvos: resultados_modelos.csv, curva_roc.png, importancia_variaveis.png, "
      "lucro_por_corte.png, previsoes_teste.csv, modelo_churn.joblib")
