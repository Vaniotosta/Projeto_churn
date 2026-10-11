"""
Funções compartilhadas pelo treino do modelo (modelagem_v2.py) e pelo app (app.py).
"""
import pandas as pd

# Regras dos "sinais de alerta" (usadas na variável nova e nos avisos do app)
LIMITE_MESES_INATIVO = 3
LIMITE_RECLAMACOES = 2
LIMITE_QUEDA_GASTO = -0.2
LIMITE_NPS_DETRATOR = 6


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis novas a partir das colunas originais da base."""
    df = df.copy()

    # peso da anuidade em relação ao que o cliente realmente gasta
    df["anuidade_sobre_gasto"] = df["anuidade_mensal"] / (df["gasto_medio_mensal_6m"] + 1)
    # quanto do limite está preso no rotativo
    df["rotativo_sobre_limite"] = df["saldo_rotativo"] / df["limite_credito"]
    # que parte dos contatos no SAC virou reclamação
    df["reclamacoes_por_contato"] = df["reclamacoes_12m"] / (df["contatos_sac_12m"] + 1)
    df["cliente_novo"] = (df["tempo_relacionamento_meses"] < 12).astype(int)
    df["queda_forte_gasto"] = (df["variacao_gasto_3m_vs_6m"] < -0.3).astype(int)

    # quantos sinais de alerta o cliente acumula (0 a 5)
    df["sinais_alerta"] = (
        (df["meses_inativos_12m"] >= LIMITE_MESES_INATIVO).astype(int)
        + (df["reclamacoes_12m"] >= LIMITE_RECLAMACOES).astype(int)
        + (df["variacao_gasto_3m_vs_6m"] < LIMITE_QUEDA_GASTO).astype(int)
        + (df["atrasos_12m"] >= 1).astype(int)
        + (df["nps"] <= LIMITE_NPS_DETRATOR).astype(int)
    )
    return df


def sinais_alerta(linha: dict) -> list:
    """Lista, em português, os sinais de alerta de um cliente."""
    avisos = []
    if linha["meses_inativos_12m"] >= LIMITE_MESES_INATIVO:
        avisos.append(f"{linha['meses_inativos_12m']} meses sem usar o cartão")
    if linha["reclamacoes_12m"] >= LIMITE_RECLAMACOES:
        avisos.append(f"{linha['reclamacoes_12m']} reclamações em 12 meses")
    if linha["variacao_gasto_3m_vs_6m"] < LIMITE_QUEDA_GASTO:
        avisos.append(f"gasto caiu {abs(linha['variacao_gasto_3m_vs_6m']):.0%} nos últimos 3 meses")
    if linha["atrasos_12m"] >= 1:
        avisos.append(f"{linha['atrasos_12m']} atraso(s) de fatura em 12 meses")
    if linha["nps"] <= LIMITE_NPS_DETRATOR:
        avisos.append(f"nota de satisfação baixa ({linha['nps']})")
    return avisos


def faixa_risco(prob: float, cortes: dict) -> str:
    """Transforma a probabilidade em Baixo, Médio ou Alto."""
    if prob >= cortes["alto"]:
        return "Alto"
    if prob >= cortes["medio"]:
        return "Médio"
    return "Baixo"


def montar_entrada(valores: dict, info: dict) -> pd.DataFrame:
    """Junta o que foi digitado com os valores padrão das demais colunas."""
    linha = dict(info["padrao"])
    linha.update(valores)
    return pd.DataFrame([linha])[info["colunas"]]
