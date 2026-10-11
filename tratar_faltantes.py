"""
Tratamento dos valores faltantes da base de churn de cartão de crédito.
Uso: python tratar_faltantes.py
"""
import pandas as pd


def resumo_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Mostra quantos valores faltam em cada coluna."""
    qtd = df.isna().sum()
    resumo = pd.DataFrame({"qtd_faltantes": qtd, "pct_faltantes": (qtd / len(df) * 100).round(1)})
    return resumo[resumo["qtd_faltantes"] > 0].sort_values("qtd_faltantes", ascending=False)


def tratar_faltantes(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Preenche os faltantes.
    `ref` é a base usada para calcular as medianas. Hoje usamos a base inteira;
    na hora de treinar o modelo, passe só a base de TREINO para não vazar informação do teste.
    """
    ref = df if ref is None else ref
    df = df.copy()

    # 1) Guarda a informação de que o dado faltava (às vezes o "não respondeu" também diz algo)
    df["renda_faltante"] = df["renda_mensal"].isna().astype(int)
    df["nps_faltante"] = df["nps"].isna().astype(int)

    # 2) Escolaridade (categoria): vira uma categoria própria
    df["escolaridade"] = df["escolaridade"].fillna("Não informado")

    # 3) Renda: mediana da categoria do cartão (Básico, Gold, Platinum, Black)
    mediana_renda = ref.groupby("categoria_cartao")["renda_mensal"].median()
    df["renda_mensal"] = df["renda_mensal"].fillna(df["categoria_cartao"].map(mediana_renda))

    # 4) NPS: mediana geral
    df["nps"] = df["nps"].fillna(ref["nps"].median())

    return df


if __name__ == "__main__":
    df = pd.read_csv("base_churn_cartao.csv")

    print("== ANTES ==")
    print(resumo_faltantes(df), "\n")

    # O faltante tem relação com o churn? (se tiver, a flag ajuda o modelo)
    for col in ["renda_mensal", "nps", "escolaridade"]:
        taxa = df.groupby(df[col].isna())["churn"].mean().round(3)
        print(f"Churn quando {col} falta: {taxa.get(True):.1%} | quando não falta: {taxa.get(False):.1%}")

    df_tratado = tratar_faltantes(df)

    print("\n== DEPOIS ==")
    print("Faltantes restantes:", int(df_tratado.isna().sum().sum()))
    comparacao = pd.DataFrame({
        "renda_antes": df["renda_mensal"].describe(),
        "renda_depois": df_tratado["renda_mensal"].describe(),
        "nps_antes": df["nps"].describe(),
        "nps_depois": df_tratado["nps"].describe(),
    }).loc[["mean", "50%", "std"]].round(2)
    print(comparacao)

    df_tratado.to_csv("base_churn_cartao_tratada.csv", index=False, encoding="utf-8")
    print("\nSalvo: base_churn_cartao_tratada.csv", df_tratado.shape)
