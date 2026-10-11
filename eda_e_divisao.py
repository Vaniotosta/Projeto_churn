"""
Etapa 2: divisão treino/teste + análise exploratória (EDA) do churn.
Uso: python eda_e_divisao.py   (precisa de base_churn_cartao.csv e tratar_faltantes.py na mesma pasta)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tratar_faltantes import tratar_faltantes

# ------------------------------------------------------------------ 1) divide ANTES de tratar
df = pd.read_csv("base_churn_cartao.csv")
treino_bruto, teste_bruto = train_test_split(
    df, test_size=0.2, stratify=df["churn"], random_state=42
)

# as medianas vêm só do treino (o teste não pode "espiar")
treino = tratar_faltantes(treino_bruto, ref=treino_bruto).reset_index(drop=True)
teste = tratar_faltantes(teste_bruto, ref=treino_bruto).reset_index(drop=True)

treino.to_csv("base_treino.csv", index=False, encoding="utf-8")
teste.to_csv("base_teste.csv", index=False, encoding="utf-8")

print(f"Treino: {treino.shape} | churn {treino['churn'].mean():.1%}")
print(f"Teste:  {teste.shape} | churn {teste['churn'].mean():.1%}\n")

# ------------------------------------------------------------------ 2) EDA (só com o treino)
taxa_geral = treino["churn"].mean()

print("== Churn por grupo (variáveis de texto e 0/1) ==")
for col in ["categoria_cartao", "genero", "estado_civil", "escolaridade", "uf",
            "usa_app", "debito_automatico", "programa_pontos"]:
    tabela = treino.groupby(col)["churn"].mean().mul(100).round(1)
    print(f"{col:18s} min {tabela.min():5.1f}% | max {tabela.max():5.1f}%")

print("\n== Correlação das variáveis numéricas com o churn ==")
numericas = treino.select_dtypes("number").drop(columns=["churn"])
corr = numericas.corrwith(treino["churn"]).sort_values()
print(corr.round(3).to_string())

# ------------------------------------------------------------------ 3) gráficos
def barras(ax, serie_grupo, titulo, xlabel):
    taxa = treino.groupby(serie_grupo, observed=True)["churn"].mean() * 100
    taxa.plot(kind="bar", ax=ax, color="#3b6ea8", width=0.7)
    ax.axhline(taxa_geral * 100, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% de churn")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(taxa.values):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Churn por perfil de cliente (linha vermelha = média geral de {taxa_geral:.0%})", fontsize=13)

barras(axes[0, 0], treino["meses_inativos_12m"].clip(upper=5), "Meses sem usar o cartão", "meses (5 = 5 ou mais)")
barras(axes[0, 1], treino["reclamacoes_12m"].clip(upper=3), "Reclamações em 12 meses", "reclamações (3 = 3 ou mais)")
barras(axes[0, 2], pd.cut(treino["tempo_relacionamento_meses"], [0, 12, 24, 48, 96, 300],
                          labels=["até 1 ano", "1-2 anos", "2-4 anos", "4-8 anos", "8+ anos"]),
       "Tempo como cliente", "")
barras(axes[1, 0], pd.cut(treino["variacao_gasto_3m_vs_6m"], [-2, -0.3, -0.1, 0.1, 2],
                          labels=["queda > 30%", "queda 10-30%", "estável", "alta > 10%"]),
       "Variação do gasto (3 meses vs 6)", "")
barras(axes[1, 1], treino["qtd_produtos"].clip(upper=5), "Produtos com o banco", "produtos (5 = 5 ou mais)")
barras(axes[1, 2], treino["usa_app"].map({0: "Não usa app", 1: "Usa app"}), "Uso do aplicativo", "")

plt.tight_layout()
plt.savefig("eda_churn.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(8, 9))
corr.plot(kind="barh", ax=ax, color=np.where(corr > 0, "#c0392b", "#3b6ea8"))
ax.set_title("Correlação com o churn (vermelho = aumenta o risco)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("correlacao_churn.png", dpi=130)
plt.close()
print("\nSalvos: base_treino.csv, base_teste.csv, eda_churn.png, correlacao_churn.png")
