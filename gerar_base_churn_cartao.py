"""
Gera uma base SINTÉTICA de clientes de cartão de crédito para modelagem de churn.
Todos os dados são fictícios (nenhum CPF, nome ou contato real).

Definição de churn: cliente que cancelou o cartão nos 90 dias após a data de referência.
Uso: python gerar_base_churn_cartao.py
"""
import numpy as np
import pandas as pd

SEED = 42
N = 10_000
TAXA_CHURN_ALVO = 0.17
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- perfil
idade = np.clip(rng.normal(41, 12, N), 18, 80).round().astype(int)
genero = rng.choice(["F", "M"], N, p=[0.52, 0.48])
estado_civil = rng.choice(["Solteiro", "Casado", "Divorciado", "Viúvo"], N, p=[0.38, 0.46, 0.12, 0.04])
escolaridade = rng.choice(["Médio", "Superior", "Pós-graduação"], N, p=[0.35, 0.45, 0.20])
uf = rng.choice(
    ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "DF", "SC"], N,
    p=[0.28, 0.14, 0.12, 0.09, 0.08, 0.08, 0.06, 0.05, 0.05, 0.05],
)

bonus_esc = pd.Series(escolaridade).map({"Médio": 1.0, "Superior": 1.45, "Pós-graduação": 2.0}).values
renda = rng.lognormal(np.log(3200), 0.55, N) * bonus_esc * (1 + (idade - 18) / 150)
renda = np.clip(renda, 1500, 60000).round(2)

# ---------------------------------------------------------------- produto
renda_ruido = renda * rng.lognormal(0, 0.25, N)
categoria = pd.cut(
    renda_ruido, bins=[0, 5000, 10000, 20000, np.inf],
    labels=["Básico", "Gold", "Platinum", "Black"],
).astype(str)

limite = (renda * rng.uniform(1.5, 4.0, N) / 100).round() * 100
limite = np.clip(limite, 800, 250000)

tempo_meses = np.clip(rng.gamma(2.2, 18, N) + 3, 3, 240).round().astype(int)
qtd_produtos = np.clip(1 + rng.poisson(0.6 + tempo_meses / 60), 1, 6)
debito_auto = rng.binomial(1, np.clip(0.25 + 0.15 * (qtd_produtos > 2), 0, 1))
usa_app = rng.binomial(1, sigmoid(2.0 - (idade - 35) / 12))
programa_pontos = rng.binomial(1, 0.55, N)

tarifa_base = pd.Series(categoria).map({"Básico": 19.9, "Gold": 34.9, "Platinum": 59.9, "Black": 119.9}).values
isento = rng.binomial(1, 0.35, N)
anuidade_mensal = np.where(isento == 1, 0.0, tarifa_base)

# ---------------------------------------------------------------- latentes (não vão para o CSV)
risco = rng.normal(0, 1, N)                       # insatisfação "escondida"
engaj = rng.normal(0, 1, N) + 0.3 * usa_app + 0.2 * debito_auto

# ---------------------------------------------------------------- comportamento de uso
utilizacao = np.clip(sigmoid(-0.6 + 0.7 * engaj + rng.normal(0, 0.4, N)), 0.01, 1.0)
gasto_medio = (limite * utilizacao * 0.85).round(2)
ticket_medio = (rng.lognormal(np.log(140), 0.35, N) * (renda / 5000) ** 0.3).round(2)
qtd_transacoes = np.clip((gasto_medio / ticket_medio).round(), 0, None).astype(int)
variacao_gasto = (rng.normal(0, 0.18, N) - 0.12 * risco).round(3)   # -0.30 = queda de 30%
meses_inativos = rng.binomial(12, sigmoid(-2.8 + 0.9 * risco - 0.6 * engaj))
compras_parceladas = rng.poisson(1.5 * utilizacao * 3)
mult_intl = pd.Series(categoria).map({"Básico": 0.5, "Gold": 0.8, "Platinum": 1.3, "Black": 2.0}).values
pct_internacional = (rng.beta(0.5, 6, N) * 100 * mult_intl).clip(0, 60).round(1)
resgates_pontos = rng.poisson(1.0, N) * programa_pontos

# ---------------------------------------------------------------- crédito e atendimento
atrasos = np.clip(rng.poisson(0.25 * np.exp(0.5 * risco) * (1 + utilizacao)), 0, 6)
p_rotativo = sigmoid(2 * (utilizacao - 0.65)) + 0.15 * (atrasos > 0)
saldo_rotativo = np.where(
    rng.random(N) < np.clip(p_rotativo, 0, 1), gasto_medio * rng.uniform(0.2, 1.2, N), 0.0
).round(2)
score = np.clip(rng.normal(650, 130, N) - 35 * atrasos + 40 * np.log10(renda / 3000), 300, 1000).round().astype(int)

reclamacoes = rng.poisson(0.35 * np.exp(0.5 * risco))
contatos_sac = reclamacoes + rng.poisson(1.0 * np.exp(0.2 * risco))
nps = np.clip(np.round(rng.normal(7.2 - 1.0 * risco - 0.8 * np.minimum(reclamacoes, 3), 1.8)), 0, 10)

# ---------------------------------------------------------------- churn
nps_logit = np.where(nps <= 6, 1, 0)
taxa_pesada = ((anuidade_mensal > 0) & (utilizacao < 0.2)).astype(int)

logit = (
    0.35 * risco
    + 0.30 * np.minimum(reclamacoes, 4)
    - 0.008 * tempo_meses
    - 0.25 * (qtd_produtos - 2)
    + 0.22 * meses_inativos
    - 1.8 * variacao_gasto
    + 0.9 * taxa_pesada
    - 0.4 * usa_app
    - 0.35 * debito_auto
    + 0.25 * atrasos
    + 0.5 * nps_logit
    - 0.012 * (idade - 40)
)

# ajusta o intercepto (busca binária) para chegar na taxa de churn desejada
lo, hi = -10.0, 10.0
for _ in range(60):
    meio = (lo + hi) / 2
    if sigmoid(logit + meio).mean() > TAXA_CHURN_ALVO:
        hi = meio
    else:
        lo = meio
prob_churn = sigmoid(logit + (lo + hi) / 2)
churn = rng.binomial(1, prob_churn)

# ---------------------------------------------------------------- monta o DataFrame
df = pd.DataFrame({
    "id_cliente": [f"CLI{100000 + i}" for i in range(N)],
    "data_referencia": "2026-09-30",
    "idade": idade,
    "genero": genero,
    "estado_civil": estado_civil,
    "escolaridade": escolaridade,
    "uf": uf,
    "renda_mensal": renda,
    "score_credito": score,
    "categoria_cartao": categoria,
    "limite_credito": limite,
    "anuidade_mensal": anuidade_mensal,
    "tempo_relacionamento_meses": tempo_meses,
    "qtd_produtos": qtd_produtos,
    "debito_automatico": debito_auto,
    "usa_app": usa_app,
    "programa_pontos": programa_pontos,
    "resgates_pontos_12m": resgates_pontos,
    "gasto_medio_mensal_6m": gasto_medio,
    "qtd_transacoes_mensal": qtd_transacoes,
    "ticket_medio": ticket_medio,
    "utilizacao_limite": (utilizacao * 100).round(1),
    "variacao_gasto_3m_vs_6m": variacao_gasto,
    "meses_inativos_12m": meses_inativos,
    "compras_parceladas_6m": compras_parceladas,
    "pct_gasto_internacional": pct_internacional,
    "saldo_rotativo": saldo_rotativo,
    "atrasos_12m": atrasos,
    "contatos_sac_12m": contatos_sac,
    "reclamacoes_12m": reclamacoes,
    "nps": nps,
    "churn": churn,
})

# sujeira realista: valores faltantes
df.loc[rng.random(N) < 0.22, "nps"] = np.nan             # nem todo cliente responde a pesquisa
df.loc[rng.random(N) < 0.02, "renda_mensal"] = np.nan
df.loc[rng.random(N) < 0.03, "escolaridade"] = np.nan

df.to_csv("base_churn_cartao.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- dicionário de dados
dicionario = pd.DataFrame([
    ("id_cliente", "Identificador fictício do cliente", "texto"),
    ("data_referencia", "Data em que as variáveis foram medidas", "data"),
    ("idade", "Idade em anos", "número"),
    ("genero", "F ou M", "categoria"),
    ("estado_civil", "Estado civil", "categoria"),
    ("escolaridade", "Nível de escolaridade (tem valores faltantes)", "categoria"),
    ("uf", "Estado do cliente", "categoria"),
    ("renda_mensal", "Renda mensal em R$ (tem valores faltantes)", "número"),
    ("score_credito", "Score de crédito de 300 a 1000", "número"),
    ("categoria_cartao", "Básico, Gold, Platinum ou Black", "categoria"),
    ("limite_credito", "Limite do cartão em R$", "número"),
    ("anuidade_mensal", "Anuidade mensal em R$ (0 = isento)", "número"),
    ("tempo_relacionamento_meses", "Meses desde que virou cliente", "número"),
    ("qtd_produtos", "Quantidade de produtos com o banco (1 a 6)", "número"),
    ("debito_automatico", "1 = tem contas no débito automático", "0/1"),
    ("usa_app", "1 = usa o aplicativo do banco", "0/1"),
    ("programa_pontos", "1 = participa do programa de pontos", "0/1"),
    ("resgates_pontos_12m", "Resgates de pontos nos últimos 12 meses", "número"),
    ("gasto_medio_mensal_6m", "Gasto médio mensal no cartão (6 meses) em R$", "número"),
    ("qtd_transacoes_mensal", "Média de compras por mês", "número"),
    ("ticket_medio", "Valor médio por compra em R$", "número"),
    ("utilizacao_limite", "% do limite usado", "número"),
    ("variacao_gasto_3m_vs_6m", "Variação do gasto dos últimos 3 meses contra os 6 meses (-0.30 = queda de 30%)", "número"),
    ("meses_inativos_12m", "Meses sem nenhuma compra nos últimos 12", "número"),
    ("compras_parceladas_6m", "Compras parceladas nos últimos 6 meses", "número"),
    ("pct_gasto_internacional", "% do gasto feito no exterior", "número"),
    ("saldo_rotativo", "Saldo em rotativo do cartão em R$", "número"),
    ("atrasos_12m", "Faturas pagas com atraso nos últimos 12 meses", "número"),
    ("contatos_sac_12m", "Contatos com o SAC nos últimos 12 meses", "número"),
    ("reclamacoes_12m", "Reclamações registradas nos últimos 12 meses", "número"),
    ("nps", "Nota de 0 a 10 na pesquisa de satisfação (tem valores faltantes)", "número"),
    ("churn", "VARIÁVEL ALVO: 1 = cancelou o cartão em até 90 dias após a data de referência", "0/1"),
], columns=["variavel", "descricao", "tipo"])
dicionario.to_csv("dicionario_variaveis.csv", index=False, encoding="utf-8")

print(df.shape, f"| taxa de churn: {df['churn'].mean():.1%}")
