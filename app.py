"""
App de risco de churn (Streamlit).
Uso: streamlit run app.py   (rode antes modelagem_v2.py para gerar o modelo)
"""
import json

import joblib
import pandas as pd
import streamlit as st

from churn_utils import faixa_risco, montar_entrada, sinais_alerta

st.set_page_config(page_title="Risco de churn - cartão", page_icon="💳", layout="wide")


@st.cache_resource
def carregar():
    modelo = joblib.load("modelo_churn.joblib")
    with open("info_modelo.json", encoding="utf-8") as f:
        info = json.load(f)
    return modelo, info


modelo, info = carregar()
cortes = info["cortes"]
campanha = info["campanha"]

st.title("💳 Risco de cancelamento do cartão")
st.caption("Projeto com dados sintéticos. A probabilidade mostra o risco de o cliente cancelar "
           "em até 90 dias.")

aba_um, aba_lote = st.tabs(["Um cliente", "Lista de clientes (CSV)"])

# ------------------------------------------------------------------ um cliente
with aba_um:
    st.subheader("Dados do cliente")
    c1, c2, c3 = st.columns(3)

    with c1:
        categoria = st.selectbox("Categoria do cartão", ["Básico", "Gold", "Platinum", "Black"], index=1)
        idade = st.slider("Idade", 18, 80, 40)
        renda = st.number_input("Renda mensal (R$)", 1500.0, 60000.0, 5000.0, step=500.0)
        limite = st.number_input("Limite do cartão (R$)", 800.0, 250000.0, 12000.0, step=1000.0)
        anuidade = st.number_input("Anuidade mensal (R$, 0 = isento)", 0.0, 200.0, 34.9, step=5.0)
        tempo = st.slider("Tempo como cliente (meses)", 3, 240, 36)

    with c2:
        produtos = st.slider("Produtos com o banco", 1, 6, 2)
        usa_app = st.checkbox("Usa o aplicativo", value=True)
        debito = st.checkbox("Tem débito automático", value=False)
        gasto = st.number_input("Gasto médio mensal no cartão (R$)", 0.0, 100000.0, 4000.0, step=250.0)
        utilizacao = st.slider("% do limite usado", 0, 100, 40)
        variacao = st.slider("Variação do gasto (últimos 3 meses vs 6), em %", -80, 60, 0)

    with c3:
        inativos = st.slider("Meses sem usar o cartão (de 12)", 0, 12, 0)
        reclamacoes = st.slider("Reclamações em 12 meses", 0, 8, 0)
        contatos = st.slider("Contatos no SAC em 12 meses", 0, 15, 1)
        atrasos = st.slider("Faturas atrasadas em 12 meses", 0, 6, 0)
        nps = st.slider("Nota de satisfação (0 a 10)", 0, 10, 8)

    valores = {
        "categoria_cartao": categoria, "idade": idade, "renda_mensal": renda,
        "limite_credito": limite, "anuidade_mensal": anuidade,
        "tempo_relacionamento_meses": tempo, "qtd_produtos": produtos,
        "usa_app": int(usa_app), "debito_automatico": int(debito),
        "gasto_medio_mensal_6m": gasto, "utilizacao_limite": float(utilizacao),
        "variacao_gasto_3m_vs_6m": variacao / 100, "meses_inativos_12m": inativos,
        "reclamacoes_12m": reclamacoes, "contatos_sac_12m": max(contatos, reclamacoes),
        "atrasos_12m": atrasos, "nps": nps,
    }

    prob = float(modelo.predict_proba(montar_entrada(valores, info))[0, 1])
    faixa = faixa_risco(prob, cortes)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Probabilidade de cancelar", f"{prob:.0%}")
    m2.metric("Faixa de risco", faixa)

    avisos = sinais_alerta(valores)
    if avisos:
        st.markdown("**Sinais de alerta deste cliente:**")
        for aviso in avisos:
            st.markdown(f"- {aviso}")
    else:
        st.markdown("Nenhum sinal de alerta forte neste cliente.")

    if faixa == "Alto":
        st.error("Risco alto: vale incluir este cliente na campanha de retenção.")
    elif faixa == "Médio":
        st.warning("Risco médio: acompanhar e reforçar o relacionamento.")
    else:
        st.success("Risco baixo: manter o relacionamento normal.")

    st.caption("As demais informações do cliente usam valores típicos da base de treino.")

# ------------------------------------------------------------------ lista de clientes
with aba_lote:
    st.subheader("Calcular o risco de vários clientes")
    st.write("Envie um CSV com as colunas da base. Colunas que faltarem serão preenchidas com "
             "valores típicos. Para testar, use o arquivo `base_teste.csv`.")
    arquivo = st.file_uploader("Arquivo CSV", type="csv")

    if arquivo is not None:
        base = pd.read_csv(arquivo)
        faltando = [c for c in info["colunas"] if c not in base.columns]
        if faltando:
            st.info(f"{len(faltando)} coluna(s) não vieram no arquivo e foram preenchidas "
                    f"com valores típicos.")
        entrada = base.copy()
        for coluna in faltando:
            entrada[coluna] = info["padrao"][coluna]
        entrada = entrada[info["colunas"]]

        resultado = base.copy()
        resultado["prob_churn"] = modelo.predict_proba(entrada)[:, 1].round(4)
        resultado["faixa_risco"] = resultado["prob_churn"].apply(lambda p: faixa_risco(p, cortes))
        resultado = resultado.sort_values("prob_churn", ascending=False)

        n_campanha = int(len(resultado) * campanha["pct_contactar"] / 100)
        k1, k2, k3 = st.columns(3)
        k1.metric("Clientes", f"{len(resultado):,}".replace(",", "."))
        k2.metric("Risco alto", int((resultado["faixa_risco"] == "Alto").sum()))
        k3.metric(f"Sugestão: contactar ({campanha['pct_contactar']}%)", n_campanha)

        colunas_tabela = [c for c in ["id_cliente", "prob_churn", "faixa_risco"] if c in resultado.columns]
        st.dataframe(resultado[colunas_tabela].head(200))
        st.download_button("Baixar resultado completo (CSV)",
                           resultado.to_csv(index=False).encode("utf-8"),
                           file_name="clientes_com_risco.csv", mime="text/csv")
