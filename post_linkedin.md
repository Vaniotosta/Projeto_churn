Montei um projeto de ponta a ponta para prever churn em cartão de crédito e quero compartilhar o que aprendi. 💳

O problema: o banco não consegue ligar para todos os clientes. Para quem vale a pena ligar?

O que fiz:
✅ Criei uma base sintética com 10 mil clientes (nenhum dado real)
✅ Tratei os valores faltantes sem deixar o teste "espiar" o treino
✅ Fiz a análise exploratória e criei variáveis novas
✅ Comparei regressão logística, Random Forest e Gradient Boosting
✅ Montei um app em Streamlit para consultar o risco de cada cliente

O que mais me chamou atenção:
👉 Ligando só para os 20% de maior risco, o modelo encontra cerca de metade dos cancelamentos.
👉 A queda no gasto e os meses sem usar o cartão pesaram mais do que perfil e renda.
👉 O modelo mais simples empatou com os mais complexos. Ajustar e criar variáveis deu só um ganho pequeno, e isso também é resultado.

Ponto de honestidade: como os dados são sintéticos, o projeto mostra o método, e não o desempenho em um banco real.

O código e o app estão no meu GitHub. Se você trabalha com retenção de clientes, queria ouvir como vocês decidem quem contactar.

#CiênciaDeDados #MachineLearning #Churn #Python #Streamlit #DataScience
