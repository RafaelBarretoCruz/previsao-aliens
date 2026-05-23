# Regras para o Claude Code — Estrutura do Notebook N2
> Baseado no documento **N2_Series_Temp.pdf** — Atividade N2, Séries Temporais 2026
> Este arquivo define **exatamente** como o Claude Code deve criar e estruturar o arquivo `.ipynb` da N2.

---

## Instruções Gerais para o Claude Code

Você deve criar um Jupyter Notebook (`.ipynb`) que implemente um **pipeline completo de séries temporais com SARIMA**. O notebook deve seguir **rigorosamente** as seções abaixo, na **ordem exata** apresentada. Cada seção deve conter:
1. Uma **célula Markdown** com o título da seção e comentários técnicos explicando o que está sendo feito e por quê.
2. Uma ou mais **células de código** com a implementação.
3. Outputs comentados com Markdown logo após a execução, quando necessário.

**Regra de ouro:** nunca pule uma seção. Se um item não for aplicável à base escolhida, documente o motivo em Markdown.

---

## Estrutura Obrigatória do Notebook (14 seções)

---

### Célula 0 — Imports e Configuração Global

**Tipo:** Markdown + Código

**Markdown deve dizer:**
> "# Pipeline SARIMA — N2 Séries Temporais 2026
> **Base de dados:** [nome da base]
> **Variável analisada:** [nome da variável]
> **Grupo:** [nomes dos integrantes]"

**Código deve conter:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error
import itertools

plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 11
```

---

### Seção 4.1 — Apresentação da Base

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.1 Apresentação da Base`
- Texto explicando: origem da base de dados, descrição do contexto, variável temporal utilizada, variável analisada, frequência da série, período coberto, quantidade de observações e possíveis problemas encontrados.

**Código deve:**
1. Carregar a base de dados:
```python
df = pd.read_excel('nome_da_base.xlsx')  # ou pd.read_csv(...)
print(df.shape)
print(df.dtypes)
df.head(10)
```
2. Exibir informações descritivas:
```python
print(f"Período: {df['coluna_data'].min()} a {df['coluna_data'].max()}")
print(f"Observações: {len(df)}")
print(df.describe())
```
3. Plotar visualização inicial da série temporal:
```python
plt.figure(figsize=(14, 4))
plt.plot(serie)
plt.title('Visualização Inicial da Série Temporal')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()
```

**Regra:** O gráfico de visualização inicial é **obrigatório** nesta seção.

---

### Seção 4.2 — Tratamento e Preparação dos Dados

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.2 Tratamento e Preparação dos Dados`
- Texto descrevendo cada tratamento aplicado e por quê.

**Código deve cobrir (quando aplicável):**
```python
# Conversão da coluna de data
df['data'] = pd.to_datetime(df['data'])

# Ordenação temporal
df = df.sort_values('data')

# Definição do índice temporal
df = df.set_index('data')

# Verificação de valores ausentes
print("Valores ausentes:", df['variavel'].isna().sum())

# Tratamento de valores faltantes (escolher a estratégia adequada)
# Opção 1: forward fill
df['variavel'] = df['variavel'].fillna(method='ffill')
# Opção 2: interpolação linear
# df['variavel'] = df['variavel'].interpolate(method='linear')

# Agregação na frequência correta
serie = df['variavel'].resample('MS').mean()  # ajustar alias conforme a série

print("Série pronta:")
print(f"  Início: {serie.index[0]}")
print(f"  Fim:    {serie.index[-1]}")
print(f"  Frequência: {serie.index.freq}")
print(f"  Observações: {len(serie)}")
print(f"  Valores nulos: {serie.isna().sum()}")
```

**Regra:** Ao final desta seção, a variável `serie` deve estar definida, com índice temporal e pronta para análise.

---

### Seção 4.3 — Análise Visual da Série

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.3 Análise Visual da Série`
- Após o gráfico: interpretação comentada observando comportamento geral, tendência, sazonalidade, períodos de crescimento/queda e possíveis mudanças estruturais.

**Código deve gerar pelo menos 2 gráficos:**

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Série completa
axes[0].plot(serie)
axes[0].set_title('Comportamento Geral da Série')
axes[0].set_ylabel('Valor')

# Médias móveis (para visualizar tendência)
axes[1].plot(serie, label='Original', alpha=0.5)
axes[1].plot(serie.rolling(window=12).mean(), label='Média Móvel (12)', color='red')
axes[1].plot(serie.rolling(window=6).mean(), label='Média Móvel (6)', color='orange')
axes[1].set_title('Série com Médias Móveis — Visualização de Tendência')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Markdown de interpretação (após os gráficos) deve responder:**
- A série possui tendência? Qual direção?
- É possível observar sazonalidade?
- Há períodos de crescimento ou queda relevantes?
- Existem possíveis quebras estruturais?

---

### Seção 4.4 — Decomposição STL

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.4 Decomposição STL`
- Explicação: "STL (Seasonal-Trend decomposition using Loess) decompõe a série em: y_t = Tendência_t + Sazonalidade_t + Ruído_t"

**Código:**
```python
from statsmodels.tsa.seasonal import STL

m = 12  # período sazonal — ajustar conforme a série (definido na seção 4.5)
stl = STL(serie.dropna(), period=m)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(result.observed);   axes[0].set_ylabel('Observado');   axes[0].set_title('Série Observada')
axes[1].plot(result.trend);      axes[1].set_ylabel('Tendência')
axes[2].plot(result.seasonal);   axes[2].set_ylabel('Sazonalidade')
axes[3].plot(result.resid);      axes[3].set_ylabel('Resíduo')
plt.suptitle(f'Decomposição STL — Período m={m}', fontsize=13)
plt.tight_layout()
plt.show()
```

**Markdown após o gráfico deve interpretar:**
- Comportamento da tendência
- Regularidade e amplitude da sazonalidade
- Comportamento do resíduo (se há padrão residual visível)

---

### Seção 4.5 — Força da Sazonalidade

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.5 Força da Sazonalidade`
- Explicação de como a força da sazonalidade influencia a escolha do `m` no SARIMA.

**Código:**
```python
def calcular_forca_sazonalidade(serie, periodo):
    try:
        stl = STL(serie.dropna(), period=periodo)
        result = stl.fit()
        var_sazonal = np.var(result.seasonal)
        var_residuo = np.var(result.resid)
        forca = max(0, 1 - var_residuo / (var_sazonal + var_residuo))
        return forca
    except:
        return np.nan

# Testar candidatos de m
m_candidatos = [3, 4, 6, 7, 12, 24, 52]
resultados_m = {}

for m_c in m_candidatos:
    f = calcular_forca_sazonalidade(serie, m_c)
    if not np.isnan(f):
        resultados_m[m_c] = f
        print(f"m = {m_c:2d}: força da sazonalidade = {f:.4f}")

m_ideal = max(resultados_m, key=resultados_m.get)
print(f"\n✅ Período sazonal escolhido: m = {m_ideal} (força = {resultados_m[m_ideal]:.4f})")
m = m_ideal  # define m globalmente
```

**Markdown após a análise deve explicar:**
- Qual m foi escolhido e por quê
- Como a força da sazonalidade justifica o uso do SARIMA em vez do ARIMA simples

---

### Seção 4.6 — Testes de Estacionariedade

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.6 Testes de Estacionariedade`
- Explicação de ADF e KPSS e suas hipóteses.

**Código:**
```python
def testar_estacionariedade(serie, nome='Série'):
    print(f"\n{'='*55}")
    print(f"Testes de Estacionariedade — {nome}")
    print('='*55)
    
    # ADF
    adf = adfuller(serie.dropna())
    print(f"\n[ADF] H₀: não estacionária")
    print(f"  Estatística: {adf[0]:.4f} | p-valor: {adf[1]:.4f}")
    print(f"  → {'✅ Estacionária (rejeita H₀)' if adf[1] <= 0.05 else '❌ Não estacionária (não rejeita H₀)'}")
    
    # KPSS
    kpss_res = kpss(serie.dropna(), regression='c', nlags='auto')
    print(f"\n[KPSS] H₀: estacionária")
    print(f"  Estatística: {kpss_res[0]:.4f} | p-valor: {kpss_res[1]:.4f}")
    print(f"  → {'❌ Não estacionária (rejeita H₀)' if kpss_res[1] <= 0.05 else '✅ Estacionária (não rejeita H₀)'}")
    
    return adf[1], kpss_res[1]

# Testar série original
p_adf_orig, p_kpss_orig = testar_estacionariedade(serie, 'Série Original')
```

**Se necessário, aplicar diferenciações:**
```python
# Diferenciação simples (d=1)
serie_diff1 = serie.diff().dropna()
p_adf_d1, p_kpss_d1 = testar_estacionariedade(serie_diff1, 'Série Diferenciada (d=1)')

# Diferenciação sazonal (D=1)
serie_diff_saz = serie.diff(m).dropna()
p_adf_ds, p_kpss_ds = testar_estacionariedade(serie_diff_saz, f'Série Diferenciada Sazonal (D=1, m={m})')

# Combinação
serie_diff_comb = serie.diff(m).diff().dropna()
p_adf_comb, p_kpss_comb = testar_estacionariedade(serie_diff_comb, 'Série Diferenciada Combinada')
```

**Markdown após os testes deve responder:**
- A série original é estacionária?
- Há evidências de tendência?
- Qual valor de `d` e `D` foi definido para o SARIMA?
- Os testes ADF e KPSS concordam ou há conflito?

---

### Seção 4.7 — Análise de ACF e PACF

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.7 Análise de ACF e PACF`
- Explicação de como usar ACF/PACF para identificar parâmetros p, q, P, Q.

**Código:**
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(serie.dropna(),       lags=48, ax=axes[0,0], title='ACF — Série Original')
plot_pacf(serie.dropna(),      lags=48, ax=axes[0,1], title='PACF — Série Original')
plot_acf(serie_diff1.dropna(), lags=48, ax=axes[1,0], title='ACF — Série Diferenciada (d=1)')
plot_pacf(serie_diff1.dropna(),lags=48, ax=axes[1,1], title='PACF — Série Diferenciada (d=1)')

plt.tight_layout()
plt.show()
```

**Markdown após os gráficos deve justificar a escolha inicial dos parâmetros:**
- `p` = ? (baseado no PACF)
- `d` = ? (baseado nos testes de estacionariedade)
- `q` = ? (baseado no ACF)
- `D` = ? (baseado na diferenciação sazonal)
- `m` = ? (baseado na força da sazonalidade — Seção 4.5)

---

### Seção 4.8 — Definição dos Parâmetros do SARIMA

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.8 Definição dos Parâmetros do SARIMA`
- Apresentação da notação: `SARIMA(p, d, q)(P, D, Q, m)`
- Justificativa dos ranges de busca escolhidos.

**Código:**
```python
# Definir espaço de busca com base na análise anterior
p_range = range(0, 3)
d_range = range(0, 2)   # máximo d=2 por boa prática
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)
# m já definido na Seção 4.5

def fit_sarima(serie, order, seasonal_order):
    try:
        modelo = SARIMAX(
            serie,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        return modelo
    except:
        return None

# Grid Search
resultados_sarima = []
total = len(list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range)))
print(f"Testando {total} combinações de parâmetros...")

for p, d, q, P, D, Q in itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range):
    if p == 0 and d == 0 and q == 0 and P == 0 and D == 0 and Q == 0:
        continue
    order = (p, d, q)
    seasonal_order = (P, D, Q, m)
    modelo = fit_sarima(train, order, seasonal_order)
    if modelo is not None:
        resultados_sarima.append({
            'order': order,
            'seasonal_order': seasonal_order,
            'bic': modelo.bic,
            'aic': modelo.aic,
            'modelo': modelo
        })

resultados_sarima = sorted(resultados_sarima, key=lambda x: x['bic'])

print(f"\n📊 Top 5 modelos SARIMA por BIC:")
for i, r in enumerate(resultados_sarima[:5]):
    print(f"  [{i+1}] SARIMA{r['order']}x{r['seasonal_order']} — BIC: {r['bic']:.2f} | AIC: {r['aic']:.2f}")

melhor_sarima = resultados_sarima[0]['modelo']
print(f"\n✅ Melhor modelo selecionado: SARIMA{resultados_sarima[0]['order']}x{resultados_sarima[0]['seasonal_order']}")
```

**Regra:** O ranking dos top 5 modelos por BIC é **obrigatório** nesta seção.

---

### Seção 4.9 — Divisão Treino e Teste

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.9 Divisão Treino e Teste`
- Justificativa do tamanho do conjunto de teste escolhido.

**Código:**
```python
# A divisão deve respeitar a ordem temporal
h = 12  # horizonte de previsão — justificar a escolha

train = serie[:-h]
test  = serie[-h:]

print(f"Tamanho do treino: {len(train)} observações ({train.index[0]} a {train.index[-1]})")
print(f"Tamanho do teste:  {len(test)} observações ({test.index[0]} a {test.index[-1]})")
print(f"Proporção treino/teste: {len(train)/len(serie)*100:.1f}% / {len(test)/len(serie)*100:.1f}%")

# Visualização da divisão
plt.figure(figsize=(14, 4))
plt.plot(train, label='Treino', color='steelblue')
plt.plot(test, label='Teste', color='orange')
plt.axvline(x=test.index[0], color='red', linestyle='--', label='Corte')
plt.legend()
plt.title('Divisão Treino / Teste')
plt.tight_layout()
plt.show()
```

**Regra:** A divisão deve ser feita **antes** dos base models e do SARIMA. A variável `h`, `train` e `test` devem estar definidas aqui e usadas em todas as seções seguintes.

---

### Seção 4.10 — Base Models

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.10 Construção dos Base Models`
- Breve descrição de cada base model implementado.

**Código deve implementar TODOS os 7 base models:**

```python
from sklearn.metrics import mean_absolute_error

resultados_base = {}

# 1. Média Histórica
pred_mh = pd.Series([train.mean()] * len(test), index=test.index)
resultados_base['Média Histórica'] = {
    'treino': mean_absolute_error(train[1:], [train.mean()] * len(train[1:])),
    'teste':  mean_absolute_error(test, pred_mh)
}

# 2. Média Acumulada
pred_ma_treino = train.expanding().mean().shift(1).dropna()
pred_ma_teste  = pd.Series([train.mean()] * len(test), index=test.index)
resultados_base['Média Acumulada'] = {
    'treino': mean_absolute_error(train[1:], pred_ma_treino),
    'teste':  mean_absolute_error(test, pred_ma_teste)
}

# 3. SMA
k_sma = 3
pred_sma_treino = train.rolling(window=k_sma).mean().shift(1).dropna()
pred_sma_teste  = pd.Series([train[-k_sma:].mean()] * len(test), index=test.index)
resultados_base[f'SMA (k={k_sma})'] = {
    'treino': mean_absolute_error(train[k_sma:], pred_sma_treino),
    'teste':  mean_absolute_error(test, pred_sma_teste)
}

# 4. EMA
alpha_ema = 0.3
pred_ema_treino = train.ewm(alpha=alpha_ema, adjust=False).mean().shift(1).dropna()
last_ema = train.ewm(alpha=alpha_ema, adjust=False).mean().iloc[-1]
pred_ema_teste  = pd.Series([last_ema] * len(test), index=test.index)
resultados_base[f'EMA (α={alpha_ema})'] = {
    'treino': mean_absolute_error(train[1:], pred_ema_treino),
    'teste':  mean_absolute_error(test, pred_ema_teste)
}

# 5. Taxa de Variação
k_taxa = 2
taxa = (train.iloc[-1] - train.iloc[-1-k_taxa]) / train.iloc[-1-k_taxa]
pred_taxa_teste = pd.Series([train.iloc[-1] * (1 + taxa)] * len(test), index=test.index)
resultados_base[f'Taxa de Variação (k={k_taxa})'] = {
    'treino': np.nan,  # calcular manualmente se necessário
    'teste':  mean_absolute_error(test, pred_taxa_teste)
}

# 6. Seasonal Naive
pred_snaive_teste = pd.Series(
    [train.iloc[-m + (i % m)] for i in range(len(test))],
    index=test.index
)
resultados_base['Seasonal Naive'] = {
    'treino': np.nan,
    'teste':  mean_absolute_error(test, pred_snaive_teste)
}

# 7. Delta (Drift)
k_delta = 2
delta = (train.iloc[-1] - train.iloc[-1-k_delta]) / k_delta
pred_delta_teste = pd.Series(
    [train.iloc[-1] + delta * (i + 1) for i in range(len(test))],
    index=test.index
)
resultados_base[f'Delta (k={k_delta})'] = {
    'treino': np.nan,
    'teste':  mean_absolute_error(test, pred_delta_teste)
}

# Tabela de resultados
df_base = pd.DataFrame(resultados_base).T
df_base.columns = ['MAE_Treino', 'MAE_Teste']
print("\n📊 Base Models — MAE:")
print(df_base.sort_values('MAE_Teste').to_string())
```

---

### Seção 4.11 — Treinamento do Modelo SARIMA

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.11 Treinamento do Modelo SARIMA`
- Apresentação dos parâmetros do melhor modelo encontrado.

**Código:**
```python
# O grid search já foi feito na Seção 4.8
# Aqui apresentamos o resumo e a previsão

print("Parâmetros do melhor modelo:")
print(f"  order:          {resultados_sarima[0]['order']}")
print(f"  seasonal_order: {resultados_sarima[0]['seasonal_order']}")
print(f"  BIC:            {resultados_sarima[0]['bic']:.2f}")
print()
print(melhor_sarima.summary())

# Previsão sobre o conjunto de teste
forecast = melhor_sarima.get_forecast(steps=h)
pred_sarima = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)

# Gráfico: valores reais vs previstos
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(serie, label='Série Real', color='black', linewidth=1.5)
ax.plot(melhor_sarima.fittedvalues, label='Ajuste In-Sample', color='steelblue', linestyle='--', alpha=0.7)
ax.plot(pred_sarima, label=f"Previsão SARIMA{resultados_sarima[0]['order']}x{resultados_sarima[0]['seasonal_order']}", color='red')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], alpha=0.2, color='red', label='IC 95%')
ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=2, label='Corte treino/teste')
ax.legend(loc='upper left')
ax.set_title('SARIMA — Valores Reais vs Previstos')
plt.tight_layout()
plt.show()
```

---

### Seção 4.12 — Diagnóstico dos Resíduos

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.12 Diagnóstico dos Resíduos`
- Explicação do que se espera de resíduos de um bom modelo (ruído branco).

**Código:**
```python
residuos = melhor_sarima.resid.dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Resíduos ao longo do tempo
axes[0, 0].plot(residuos)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Resíduos ao Longo do Tempo')

# Histograma
axes[0, 1].hist(residuos, bins=30, edgecolor='black', color='steelblue')
axes[0, 1].set_title('Distribuição dos Resíduos')

# ACF dos resíduos
plot_acf(residuos, lags=40, ax=axes[1, 0], title='ACF dos Resíduos')

# PACF dos resíduos
plot_pacf(residuos, lags=40, ax=axes[1, 1], title='PACF dos Resíduos')

plt.tight_layout()
plt.show()

# Teste de Ljung-Box
lb = acorr_ljungbox(residuos, lags=[10, 20, 30], return_df=True)
print("\nTeste de Ljung-Box:")
print(lb)
print()
for lag, row in lb.iterrows():
    status = "✅ Ruído branco" if row['lb_pvalue'] > 0.05 else "⚠️ Autocorrelação restante"
    print(f"  Lag {lag}: p-valor = {row['lb_pvalue']:.4f} — {status}")
```

**Markdown após os testes deve responder:**
- Os resíduos parecem ruído branco?
- Ainda existe autocorrelação significativa?
- O modelo capturou bem a estrutura temporal da série?
- Há sinais de que o modelo pode ser melhorado?

---

### Seção 4.13 — Avaliação de Desempenho

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.13 Avaliação de Desempenho`
- Contexto sobre o que se está comparando.

**Código:**
```python
mae_sarima_treino = mean_absolute_error(train, melhor_sarima.fittedvalues.dropna()[:len(train)])
mae_sarima_teste  = mean_absolute_error(test, pred_sarima)

# Tabela completa
tabela_final = df_base.copy()
tabela_final.loc['SARIMA (estático)'] = [mae_sarima_treino, mae_sarima_teste]

tabela_final = tabela_final.sort_values('MAE_Teste')
tabela_final['MAE_Teste'] = tabela_final['MAE_Teste'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else '-')
tabela_final['MAE_Treino'] = tabela_final['MAE_Treino'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else '-')

print("\n📊 Tabela Final de Desempenho (ordenada por MAE_Teste):")
print(tabela_final.to_string())
```

**Markdown após a tabela deve discutir:**
- Qual modelo teve menor erro no treino?
- Qual modelo teve menor erro no teste?
- Houve possível overfitting em algum modelo?
- O SARIMA superou os base models?
- O ganho do SARIMA foi relevante?

---

### Seção 4.14 — Rolling Forecast

**Tipo:** Markdown + Código

**Markdown deve conter:**
- Título `## 4.14 Rolling Forecast`
- Explicação: "O rolling forecast simula o uso real do modelo: a cada período, novas observações são incorporadas. É uma estratégia mais realista que o corte fixo treino-teste."

**Código:**
```python
def rolling_forecast_sarima(serie, train_size, order, seasonal_order, h=1):
    previsoes = []
    indices   = []
    for i in range(len(serie) - train_size):
        window = serie.iloc[:train_size + i]
        try:
            modelo = SARIMAX(
                window,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            pred = modelo.forecast(steps=h)
            previsoes.append(pred.iloc[0])
        except:
            previsoes.append(np.nan)
        indices.append(serie.index[train_size + i])
    return pd.Series(previsoes, index=indices)

train_size = len(train)

# Testar os top 5 modelos no rolling forecast
print("🔄 Rolling Forecast — Top 5 modelos:")
resultados_rolling = {}

for i, r in enumerate(resultados_sarima[:5]):
    preds = rolling_forecast_sarima(serie, train_size, r['order'], r['seasonal_order'])
    mae_roll = mean_absolute_error(test.dropna(), preds.dropna())
    resultados_rolling[f"SARIMA{r['order']}x{r['seasonal_order']}"] = {'mae': mae_roll, 'preds': preds}
    print(f"  [{i+1}] SARIMA{r['order']}x{r['seasonal_order']}: MAE Rolling = {mae_roll:.4f}")

melhor_rolling_nome = min(resultados_rolling, key=lambda x: resultados_rolling[x]['mae'])
melhor_rolling_preds = resultados_rolling[melhor_rolling_nome]['preds']
print(f"\n✅ Melhor modelo no Rolling: {melhor_rolling_nome}")

# Gráfico comparativo: estático vs rolling
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(serie, label='Real', color='black')
ax.plot(pred_sarima, label='SARIMA Estático', color='red', linestyle='--')
ax.plot(melhor_rolling_preds, label=f'Rolling ({melhor_rolling_nome})', color='green', linestyle='-.')
ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=2)
ax.legend()
ax.set_title('Comparação: Previsão Estática vs Rolling Forecast')
plt.tight_layout()
plt.show()

# MAE comparativo
print(f"\nMAE SARIMA estático: {mae_sarima_teste:.4f}")
print(f"MAE Rolling melhor:  {resultados_rolling[melhor_rolling_nome]['mae']:.4f}")
```

**Markdown após o código deve responder:**
- O rolling forecast melhorou o desempenho?
- O modelo se beneficiou da atualização com novos dados?
- Esse comportamento faz sentido para a série analisada?

---

## Regras Adicionais para o Claude Code

1. **Cada seção = 1 célula Markdown + 1 ou mais células de código.** Não misturar código e texto na mesma célula.
2. **A variável `m` deve ser definida na Seção 4.5** e reutilizada em todas as seções subsequentes.
3. **A variável `train` e `test` devem ser definidas na Seção 4.9** e reutilizadas em 4.10, 4.11, 4.12, 4.13 e 4.14.
4. **O grid search do SARIMA deve ser feito na Seção 4.8** com o conjunto de treino. O melhor modelo (`melhor_sarima`) deve ser reutilizado nas seções 4.11, 4.12 e 4.13.
5. **Não usar dados de teste para selecionar o modelo.** O BIC é calculado somente sobre o conjunto de treino.
6. **Todas as células de código devem ser executáveis** sem erros. Se um bloco depende de uma variável definida anteriormente, garantir que a célula anterior foi executada.
7. **Salvar o notebook como:** `N2_SARIMA_[nome_da_base].ipynb`
8. **O nome da base de dados e a variável analisada devem ser preenchidos na Célula 0.** Substitua os placeholders `[nome da base]` e `[nome da variável]`.
9. **O notebook deve ser auto-suficiente:** incluir todos os imports na Célula 0, não distribuir imports ao longo do notebook.
10. **Os gráficos devem ter:** título, labels nos eixos, legenda quando houver mais de uma linha, e `plt.tight_layout()` antes de `plt.show()`.

---

## Resumo da Estrutura (ordem das células)

| # | Seção | Tipo | Obrigatório |
|---|-------|------|-------------|
| 0 | Imports e Configuração | MD + Código | ✅ |
| 1 | 4.1 Apresentação da Base | MD + Código | ✅ |
| 2 | 4.2 Tratamento e Preparação | MD + Código | ✅ |
| 3 | 4.3 Análise Visual | MD + Código | ✅ |
| 4 | 4.4 Decomposição STL | MD + Código | ✅ |
| 5 | 4.5 Força da Sazonalidade | MD + Código | ✅ |
| 6 | 4.6 Testes de Estacionariedade | MD + Código | ✅ |
| 7 | 4.7 Análise ACF e PACF | MD + Código | ✅ |
| 8 | 4.8 Parâmetros SARIMA + Grid Search | MD + Código | ✅ |
| 9 | 4.9 Divisão Treino / Teste | MD + Código | ✅ |
| 10 | 4.10 Base Models (todos os 7) | MD + Código | ✅ |
| 11 | 4.11 Treinamento SARIMA | MD + Código | ✅ |
| 12 | 4.12 Diagnóstico dos Resíduos | MD + Código | ✅ |
| 13 | 4.13 Avaliação de Desempenho (tabela) | MD + Código | ✅ |
| 14 | 4.14 Rolling Forecast | MD + Código | ✅ |

> **Fonte:** N2_Series_Temp.pdf — Atividade N2, Disciplina Séries Temporais 2026, Instituto Germinare.
