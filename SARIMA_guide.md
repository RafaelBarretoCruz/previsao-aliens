# Guia Completo de Séries Temporais com SARIMA
> Extraído das Aulas 1–7 — Disciplina Séries Temporais (Daniel Freitas, 2026)
> Este arquivo serve como referência para o Claude executar um pipeline completo de SARIMA sobre uma base fornecida.

---

## 1. Conceitos Fundamentais de Séries Temporais (Aula 1)

Uma **série temporal** é um conjunto de observações ordenadas no tempo (segundos, minutos, dias, semanas, meses).

### Características
- Dependência ao longo do tempo
- Padrões sazonais e cíclicos
- Previsão (forecasting): prever valores futuros (ex: demanda de produto, cotação de ações)

### Padrões Principais
- **Tendência (Trend):** comportamento de longo prazo — crescente, decrescente, estacionária ou não-linear
- **Sazonalidade:** padrões repetitivos que ocorrem em intervalos regulares (mensal, anual) devido a fatores sazonais ou periódicos
- **Ciclicidade:** variações de mais longo prazo associadas a ciclos econômicos, sem periodicidade fixa
- **Ruído:** flutuações aleatórias que não seguem padrões previsíveis

### Tipos de Intervalo
- **Regulares:** observações em datas fixas e uniformes
- **Irregulares:** observações sem periodicidade definida

### Setup no Pandas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Leitura de arquivo externo com índice temporal
df = pd.read_excel('sua_serie.xlsx', parse_dates=['data'], index_col='data')

# Definir frequência da série
df = df.resample('MS').mean()   # 'MS' = início do mês; 'D' = diário; 'W' = semanal; 'QS' = trimestral

# Visualização inicial
plt.figure(figsize=(12, 4))
plt.plot(df)
plt.title('Série Temporal Original')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()
```

#### Aliases de frequência comuns (pandas)
| Alias | Descrição |
|-------|-----------|
| `D`   | Diário    |
| `W`   | Semanal   |
| `MS`  | Início de mês |
| `QS`  | Início de trimestre |
| `YS`  | Início de ano |

---

## 2. Estatísticas Dinâmicas e Base Models (Aula 2)

### Estatísticas Dinâmicas
As propriedades dos dados podem variar com o tempo. Em vez de calcular uma única média ou desvio-padrão para todo o conjunto, utilizamos **médias móveis** e **desvios-padrão móveis** para acompanhar a evolução dos parâmetros estatísticos em janelas temporais específicas.

### Métricas de Erro

| Métrica | O que mede | Características |
|---------|-----------|-----------------|
| **MAE** | Média da diferença absoluta entre valor real e previsto | Todos os erros com o mesmo peso |
| **MSE** | Média dos erros elevados ao quadrado | Penaliza mais erros grandes, sensível a outliers |
| **RMSE** | Raiz quadrada do MSE, erro na mesma escala dos dados | Mais severa com erros grandes |
| **MAPE** | Erro médio expresso em porcentagem | Bom para comparar entre escalas, mas problemático com valores pequenos |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_real, y_pred)
mse = mean_squared_error(y_real, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
```

### Base Models (modelos de referência)

Os base models são modelos simples usados como ponto de partida para comparação com modelos mais complexos. Devem ser implementados **antes** do SARIMA para avaliar ganho preditivo.

#### 1. Média Histórica
```python
media_historica = train.mean()
pred_media = pd.Series([media_historica] * len(test), index=test.index)
mae_media = mean_absolute_error(test, pred_media)
```

#### 2. Média Acumulada (Expanding Mean)
Prevê cada novo valor com a média de todos os valores anteriores, adaptando-se ao longo do tempo.
```python
pred_acumulada = train.expanding().mean().shift(1)
# No teste, usa a média acumulada até o fim do treino
pred_acumulada_test = pd.Series([train.mean()] * len(test), index=test.index)
```

#### 3. Média Móvel Simples (SMA)
Prevê o próximo valor usando a média dos últimos k pontos observados.
```python
k = 3  # janela
pred_sma = train.rolling(window=k).mean().shift(1)
# Para o teste (last k values do treino)
last_k = train[-k:].mean()
pred_sma_test = pd.Series([last_k] * len(test), index=test.index)
```

#### 4. Média Móvel Exponencial (EMA)
Atribui pesos maiores aos valores mais recentes, decaindo exponencialmente para os mais antigos.

**Fórmula:** `EMA_t = α * y_(t-1) + (1 - α) * EMA_(t-1)`

```python
alpha = 0.3
pred_ema = train.ewm(alpha=alpha, adjust=False).mean().shift(1)
last_ema = train.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
pred_ema_test = pd.Series([last_ema] * len(test), index=test.index)
```

#### 5. Taxa de Variação
Prevê com base na variação percentual entre o valor atual e o valor de k períodos atrás.

**Fórmula:**
```
taxa = (x_t - x_(t-k)) / x_(t-k)
previsão = x_t * (1 + taxa)
```

```python
k = 2
taxa = (train.iloc[-1] - train.iloc[-1-k]) / train.iloc[-1-k]
pred_taxa = train.iloc[-1] * (1 + taxa)
pred_taxa_test = pd.Series([pred_taxa] * len(test), index=test.index)
```

#### 6. Seasonal Naive
Prevê o próximo valor com base no valor observado s períodos atrás. Assume repetição sazonal.

```python
s = 12  # período sazonal (ex: 12 para dados mensais)
pred_snaive = train.shift(s)
# Para o teste
pred_snaive_test = pd.Series(
    [train.iloc[-s + (i % s)] for i in range(len(test))],
    index=test.index
)
```

#### 7. Delta (Drift)
Projeta o próximo valor com base na tendência média dos últimos k períodos.

**Fórmula:**
```
delta = (x_t - x_(t-k)) / k
previsão = x_t + delta
```

```python
k = 2
delta = (train.iloc[-1] - train.iloc[-1-k]) / k
pred_delta_test = pd.Series(
    [train.iloc[-1] + delta * (i + 1) for i in range(len(test))],
    index=test.index
)
```

### Comparação dos Base Models

```python
base_models = {
    'Média Histórica': pred_media_test,
    'Média Acumulada': pred_acumulada_test,
    'SMA': pred_sma_test,
    'EMA': pred_ema_test,
    'Taxa de Variação': pred_taxa_test,
    'Seasonal Naive': pred_snaive_test,
    'Delta': pred_delta_test,
}

resultados = {}
for nome, pred in base_models.items():
    mae = mean_absolute_error(test, pred)
    resultados[nome] = mae

resultados_df = pd.DataFrame.from_dict(resultados, orient='index', columns=['MAE'])
resultados_df = resultados_df.sort_values('MAE')
print(resultados_df)
```

---

## 3. Estacionariedade (Aula 3)

### Definição
Uma série temporal é **estacionária** quando suas propriedades estatísticas não mudam ao longo do tempo. É importante verificar estacionariedade pois muitos modelos preditivos partem desse princípio.

### Tipos de Estacionariedade

**Estrita (Forte):** a distribuição de qualquer subconjunto não muda ao longo do tempo. Difícil de verificar na prática.

**Fraca:** a série satisfaz 3 condições:
1. Média constante no tempo
2. Variância constante no tempo
3. Autocovariância depende apenas do lag (não do tempo)

### Autocovariância e Autocorrelação

**Autocovariância:** mede a covariância de uma série consigo mesma em diferentes defasagens (lags).

**Autocorrelação (ACF):** versão padronizada pela variância. Responde: "O quanto o presente se parece com o passado, em média?"

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(serie, lags=40, ax=axes[0], title='ACF')
plot_pacf(serie, lags=40, ax=axes[1], title='PACF')
plt.tight_layout()
plt.show()
```

**Como interpretar o ACF:**
- Barras caem rápido → série estacionária
- Barras ficam altas por muito tempo → série não estacionária (memória longa)

### Testes de Estacionariedade

| Ferramenta | O que faz | Para que serve |
|-----------|-----------|----------------|
| **ACF** | Visualiza visualmente a memória da série | Exploração inicial |
| **ADF** | Testa se a série NÃO é estacionária | Diagnóstico estatístico |
| **KPSS** | Testa se a série É estacionária | Diagnóstico complementar |

#### Teste ADF (Augmented Dickey-Fuller)
- H₀: não estacionária
- H₁: estacionária
- p ≤ 0.05 → rejeita H₀ (série estacionária)
- p > 0.05 → não rejeita H₀ (série não estacionária)
- **Queremos rejeitar H₀**

```python
from statsmodels.tsa.stattools import adfuller, kpss

def testar_estacionariedade(serie, nome='Série'):
    print(f"\n{'='*50}")
    print(f"Testes de Estacionariedade — {nome}")
    print('='*50)
    
    # Teste ADF
    adf_result = adfuller(serie.dropna())
    print(f"\nTeste ADF:")
    print(f"  Estatística: {adf_result[0]:.4f}")
    print(f"  p-valor:     {adf_result[1]:.4f}")
    print(f"  Conclusão:   {'Estacionária (rejeita H0)' if adf_result[1] <= 0.05 else 'Não estacionária (não rejeita H0)'}")
    
    # Teste KPSS
    kpss_result = kpss(serie.dropna(), regression='c', nlags='auto')
    print(f"\nTeste KPSS:")
    print(f"  Estatística: {kpss_result[0]:.4f}")
    print(f"  p-valor:     {kpss_result[1]:.4f}")
    print(f"  Conclusão:   {'Não estacionária (rejeita H0)' if kpss_result[1] <= 0.05 else 'Estacionária (não rejeita H0)'}")
    
    return adf_result[1], kpss_result[1]
```

#### Interpretação conjunta ADF + KPSS

| Resultado ADF | Resultado KPSS | Conclusão |
|--------------|----------------|-----------|
| Rejeita H₀   | Não rejeita H₀ | ✅ Estacionária (os dois concordam) |
| Não rejeita H₀ | Rejeita H₀   | ✅ Não estacionária (os dois concordam) |
| Rejeita H₀   | Rejeita H₀    | ⚠️ Inconclusivo — os dois discordam |
| Não rejeita H₀ | Não rejeita H₀ | ⚠️ Inconclusivo — os dois são conservadores |

---

## 4. Modelo AR (AutoRegressivo) e PACF (Aula 4)

### Definição
O modelo AR é um modelo de séries temporais em que o valor atual da série depende linearmente de valores passados da própria série.

**Quando utilizar:**
- Quando a série é estacionária (ou foi transformada para ser)
- Quando os valores passados têm relação com os dados futuros
- Quando se deseja algo mais sofisticado do que os Base Models

### PACF — Partial Autocorrelation Function
- Mede a correlação entre a série e seus valores defasados **removendo** o efeito intermediário dos lags menores
- Foca no impacto "puro" de cada lag, sem interferência dos anteriores
- Serve principalmente para identificar até qual lag um modelo AR deve considerar
- O **último lag significativo** antes de cair para a zona de confiança indica o limite máximo para **p**

### BIC — Bayesian Information Criterion
Critério de seleção de modelos que equilibra ajuste e simplicidade. Menor BIC = melhor modelo.

```python
from statsmodels.tsa.ar_model import AutoReg
import itertools

# Seleção do p por BIC
melhor_bic = np.inf
melhor_p = 0

for p in range(1, 15):
    try:
        modelo = AutoReg(train, lags=p).fit()
        if modelo.bic < melhor_bic:
            melhor_bic = modelo.bic
            melhor_p = p
    except:
        continue

print(f"Melhor p pelo BIC: {melhor_p} (BIC = {melhor_bic:.2f})")
```

### Teste de Ljung-Box (resíduos)
Verifica conjuntamente se todas as autocorrelações dos resíduos até o lag h são zero (ruído branco).
- H₀: não há autocorrelação nos resíduos até h
- H₁: existe autocorrelação nos lags
- p > 0.05 → não rejeita H₀ → resíduos são ruído branco ✅
- **Queremos NÃO rejeitar H₀**

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_result = acorr_ljungbox(residuos, lags=[10, 20], return_df=True)
print(lb_result)
```

---

## 5. Diferenciação (Aula 5)

Muitas séries temporais não são estacionárias: apresentam tendência ou sazonalidade. Os modelos AR, MA e ARMA exigem estacionariedade. A **diferenciação** é a técnica matemática usada para remover tendência ou sazonalidade.

### Tipos de Diferenciação

| Tipo | Uso principal |
|------|---------------|
| Primeira diferença | Remove tendência linear |
| Segunda diferença | Remove tendência quadrática |
| Diferença de ordem d | Remove tendências mais complexas |
| Diferença sazonal | Remove sazonalidade |
| Combinação | Remove tendência e sazonalidade ao mesmo tempo |

### Diferenciação Simples (1ª ordem)
```python
serie_diff1 = serie.diff().dropna()
# Testar estacionariedade
testar_estacionariedade(serie_diff1, 'Série Diferenciada (d=1)')
```

### Diferenciação de Ordem Superior
```python
serie_diff2 = serie.diff().diff().dropna()
testar_estacionariedade(serie_diff2, 'Série Diferenciada (d=2)')
```

### Diferenciação Sazonal
Se a série tem padrão repetitivo (mensal, semanal etc.):
```python
s = 12  # período sazonal
serie_diff_sazonal = serie.diff(s).dropna()
testar_estacionariedade(serie_diff_sazonal, 'Série Diferenciada Sazonal')
```

### Combinação (tendência + sazonalidade)
```python
s = 12
serie_diff_combinada = serie.diff(s).diff().dropna()
testar_estacionariedade(serie_diff_combinada, 'Série Diferenciada Combinada')
```

### Retornando à escala original
```python
# Se diff() foi aplicada, integrar para voltar à escala original
# Para d=1:
previsao_original = previsao_diff.cumsum() + serie.iloc[-1]
```

---

## 6. ARIMA — AR + I + MA (Aula 6)

### Moving Average (MA(q))
Modela o valor atual como uma combinação linear dos **erros passados** (não dos valores passados como no AR).
- `MA(q) = ARIMA(0, 0, q)` no Python

### ARMA(p,q)
Combina AR e MA em uma mesma equação. Adequado para séries estacionárias.

### ARIMA(p, d, q)
Generalização do ARMA: acrescenta o **I** (Integrated), que são diferenciações aplicadas à série para remover a não-estacionariedade.

```python
from statsmodels.tsa.arima.model import ARIMA

# Grid search por BIC
melhor_bic = np.inf
melhor_ordem = None
melhor_modelo = None

for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            if p == 0 and d == 0 and q == 0:
                continue  # ignora modelo trivial
            try:
                modelo = ARIMA(train, order=(p, d, q)).fit()
                if modelo.bic < melhor_bic:
                    melhor_bic = modelo.bic
                    melhor_ordem = (p, d, q)
                    melhor_modelo = modelo
            except:
                continue

print(f"Melhor ordem ARIMA: {melhor_ordem} (BIC = {melhor_bic:.2f})")
print(melhor_modelo.summary())
```

### Forecast com intervalo de confiança
```python
h = len(test)  # horizonte = tamanho do conjunto de teste

forecast = melhor_modelo.get_forecast(steps=h)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)  # intervalo de 95%

# Plotagem
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(serie, label='Série Real', color='black')
ax.plot(melhor_modelo.fittedvalues, label='Ajuste In-Sample', color='blue', linestyle='--')
ax.plot(pred_mean, label='Previsão', color='red')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                alpha=0.2, color='red', label='IC 95%')
ax.axvline(x=test.index[0], color='gray', linestyle=':', label='Corte treino/teste')
ax.legend()
ax.set_title(f'ARIMA{melhor_ordem} — Previsão')
plt.tight_layout()
plt.show()
```

---

## 7. SARIMA — Seasonal ARIMA (Aula 7)

### Definição

**SARIMA(p, d, q)(P, D, Q, m)** modela a série em duas camadas:

| Parâmetro | Parte | Descrição |
|-----------|-------|-----------|
| `p` | Não sazonal | Ordem AR — quantos lags passados são usados |
| `d` | Não sazonal | Grau de diferenciação para estacionariedade |
| `q` | Não sazonal | Ordem MA — quantos erros passados são usados |
| `P` | Sazonal | Ordem AR sazonal |
| `D` | Sazonal | Grau de diferenciação sazonal |
| `Q` | Sazonal | Ordem MA sazonal |
| `m` | Sazonal | Período/ciclo sazonal (ex: 12 = mensal com sazonalidade anual; 7 = diário com sazonalidade semanal) |

### Decomposição STL
Antes de modelar, decompor a série para entender seus componentes:

**STL (Seasonal-Trend decomposition using Loess):**

`y_t = Tendência_t + Sazonalidade_t + Ruído_t`

```python
from statsmodels.tsa.seasonal import STL

# Testar diferentes períodos para encontrar o m com maior força sazonal
for m_candidato in [3, 4, 7, 12, 52]:
    try:
        stl = STL(serie, period=m_candidato)
        result = stl.fit()
        
        # Força da sazonalidade
        var_sazonal = np.var(result.seasonal)
        var_residuo = np.var(result.resid)
        forca = max(0, 1 - var_residuo / (var_sazonal + var_residuo))
        print(f"m = {m_candidato}: força da sazonalidade = {forca:.4f}")
    except:
        continue
```

```python
# Decomposição com o m escolhido
m = 12  # ajustar conforme a série
stl = STL(serie, period=m)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(result.observed);  axes[0].set_ylabel('Observado')
axes[1].plot(result.trend);     axes[1].set_ylabel('Tendência')
axes[2].plot(result.seasonal);  axes[2].set_ylabel('Sazonalidade')
axes[3].plot(result.resid);     axes[3].set_ylabel('Resíduo')
plt.suptitle(f'Decomposição STL (m={m})', fontsize=14)
plt.tight_layout()
plt.show()
```

### Força da Sazonalidade
```python
def forca_sazonalidade(serie, periodo):
    stl = STL(serie.dropna(), period=periodo)
    result = stl.fit()
    var_sazonal = np.var(result.seasonal)
    var_residuo = np.var(result.resid)
    forca = max(0, 1 - var_residuo / (var_sazonal + var_residuo))
    return forca

# Buscar o m ideal
m_candidatos = [3, 4, 6, 7, 12, 24, 52]
resultados_m = {}
for m_c in m_candidatos:
    try:
        f = forca_sazonalidade(serie, m_c)
        resultados_m[m_c] = f
        print(f"m = {m_c}: força = {f:.4f}")
    except:
        pass

m_ideal = max(resultados_m, key=resultados_m.get)
print(f"\nMelhor m: {m_ideal} (força = {resultados_m[m_ideal]:.4f})")
```

### Pipeline SARIMA Completo

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# ── Parâmetros de busca ──────────────────────────────────────────
m = 12          # período sazonal — ajustar conforme a série
h = 12          # horizonte de previsão (tamanho do conjunto de teste)

p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)

# ── Função de ajuste ─────────────────────────────────────────────
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

# ── Grid Search ──────────────────────────────────────────────────
resultados_sarima = []

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

# Ordenar por BIC
resultados_sarima = sorted(resultados_sarima, key=lambda x: x['bic'])

print("Top 5 modelos SARIMA por BIC:")
for i, r in enumerate(resultados_sarima[:5]):
    print(f"  [{i+1}] SARIMA{r['order']}x{r['seasonal_order']} — BIC: {r['bic']:.2f}")

melhor_sarima = resultados_sarima[0]['modelo']
print(f"\nMelhor modelo: SARIMA{resultados_sarima[0]['order']}x{resultados_sarima[0]['seasonal_order']}")
print(melhor_sarima.summary())
```

### Diagnóstico dos Resíduos

```python
# Gráfico dos resíduos
residuos = melhor_sarima.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Resíduos ao longo do tempo
axes[0, 0].plot(residuos)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Resíduos ao longo do tempo')

# Histograma dos resíduos
axes[0, 1].hist(residuos, bins=30, edgecolor='black')
axes[0, 1].set_title('Distribuição dos Resíduos')

# ACF dos resíduos
plot_acf(residuos, lags=40, ax=axes[1, 0], title='ACF dos Resíduos')

# PACF dos resíduos
plot_pacf(residuos, lags=40, ax=axes[1, 1], title='PACF dos Resíduos')

plt.tight_layout()
plt.show()

# Teste Ljung-Box
lb_result = acorr_ljungbox(residuos.dropna(), lags=[10, 20, 30], return_df=True)
print("\nTeste de Ljung-Box nos resíduos:")
print(lb_result)
print("\nInterpretação:")
for lag, row in lb_result.iterrows():
    status = "✅ OK (ruído branco)" if row['lb_pvalue'] > 0.05 else "⚠️ Autocorrelação restante"
    print(f"  Lag {lag}: p-valor = {row['lb_pvalue']:.4f} — {status}")
```

### Previsão com SARIMA

```python
# Previsão estática (sobre o conjunto de teste)
forecast = melhor_sarima.get_forecast(steps=h)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)

# Plotagem
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(serie, label='Série Real', color='black')
ax.plot(melhor_sarima.fittedvalues, label='Ajuste In-Sample', color='blue', linestyle='--', alpha=0.7)
ax.plot(pred_mean, label='Previsão SARIMA', color='red')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                alpha=0.2, color='red', label='IC 95%')
ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=2, label='Corte treino/teste')
ax.legend()
ax.set_title(f'SARIMA{resultados_sarima[0]["order"]}x{resultados_sarima[0]["seasonal_order"]} — Previsão')
plt.tight_layout()
plt.show()

# MAE do SARIMA
mae_sarima_treino = mean_absolute_error(train, melhor_sarima.fittedvalues.dropna())
mae_sarima_teste = mean_absolute_error(test, pred_mean)
print(f"MAE SARIMA — Treino: {mae_sarima_treino:.4f} | Teste: {mae_sarima_teste:.4f}")
```

### Rolling Forecast

O Rolling Forecast simula o uso real do modelo: a cada período, novas observações são incorporadas para gerar previsões futuras. Evita o viés de um único corte treino-teste fixo.

```python
def rolling_forecast_sarima(serie, train_size, order, seasonal_order, h=1):
    """
    Rolling forecast 1-step-ahead com SARIMA.
    A cada passo, o modelo é re-treinado com todos os dados disponíveis até aquele ponto.
    """
    previsoes = []
    indices = []
    
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
            indices.append(serie.index[train_size + i])
        except:
            previsoes.append(np.nan)
            indices.append(serie.index[train_size + i])
    
    return pd.Series(previsoes, index=indices)

# Executar rolling forecast para os top 5 modelos
train_size = len(train)
resultados_rolling = {}

print("Executando Rolling Forecast para os top 5 modelos...")
for i, r in enumerate(resultados_sarima[:5]):
    print(f"  Testando modelo [{i+1}]: SARIMA{r['order']}x{r['seasonal_order']}")
    preds_rolling = rolling_forecast_sarima(
        serie, train_size,
        order=r['order'],
        seasonal_order=r['seasonal_order']
    )
    mae_rolling = mean_absolute_error(test.dropna(), preds_rolling.dropna())
    resultados_rolling[f"SARIMA{r['order']}x{r['seasonal_order']}"] = {
        'mae': mae_rolling,
        'previsoes': preds_rolling
    }
    print(f"    MAE Rolling: {mae_rolling:.4f}")

# Melhor modelo no rolling
melhor_rolling = min(resultados_rolling, key=lambda x: resultados_rolling[x]['mae'])
print(f"\nMelhor modelo no Rolling Forecast: {melhor_rolling}")
```

---

## 8. Pipeline Completo — Checklist de Execução

Use este checklist ao rodar o SARIMA sobre uma base fornecida:

```python
# ════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error
import itertools

# ════════════════════════════════════════════════════════════════
# PASSO 1: CARGA E PREPARAÇÃO DOS DADOS
# ════════════════════════════════════════════════════════════════
# df = pd.read_excel('sua_base.xlsx', parse_dates=['data'], index_col='data')
# serie = df['variavel'].resample('MS').mean().dropna()

# ════════════════════════════════════════════════════════════════
# PASSO 2: VISUALIZAÇÃO INICIAL
# ════════════════════════════════════════════════════════════════
# plt.figure(figsize=(12,4)); plt.plot(serie); plt.title('Série Original'); plt.show()

# ════════════════════════════════════════════════════════════════
# PASSO 3: DECOMPOSIÇÃO STL + FORÇA DA SAZONALIDADE
# ════════════════════════════════════════════════════════════════
# → Identificar m (período sazonal)

# ════════════════════════════════════════════════════════════════
# PASSO 4: TESTES DE ESTACIONARIEDADE (ADF + KPSS)
# ════════════════════════════════════════════════════════════════
# → Decidir d (diferenciação) e D (diferenciação sazonal)

# ════════════════════════════════════════════════════════════════
# PASSO 5: ACF e PACF
# ════════════════════════════════════════════════════════════════
# → Identificar candidatos para p, q, P, Q

# ════════════════════════════════════════════════════════════════
# PASSO 6: DIVISÃO TREINO / TESTE
# ════════════════════════════════════════════════════════════════
# h = 12  # ou outro horizonte adequado
# train = serie[:-h]
# test  = serie[-h:]

# ════════════════════════════════════════════════════════════════
# PASSO 7: BASE MODELS (referência)
# ════════════════════════════════════════════════════════════════
# → Calcular MAE de cada base model no treino e no teste

# ════════════════════════════════════════════════════════════════
# PASSO 8: GRID SEARCH SARIMA + RANKING POR BIC
# ════════════════════════════════════════════════════════════════
# → Encontrar os top 5 modelos por BIC

# ════════════════════════════════════════════════════════════════
# PASSO 9: DIAGNÓSTICO DOS RESÍDUOS
# ════════════════════════════════════════════════════════════════
# → Ljung-Box, ACF dos resíduos, histograma

# ════════════════════════════════════════════════════════════════
# PASSO 10: PREVISÃO ESTÁTICA + MÉTRICAS
# ════════════════════════════════════════════════════════════════
# → MAE treino e teste — tabela comparativa

# ════════════════════════════════════════════════════════════════
# PASSO 11: ROLLING FORECAST
# ════════════════════════════════════════════════════════════════
# → Top 5 modelos; comparar MAE rolling vs estático
```

---

## 9. Tabela de Comparação Final

```python
# Organizar todos os resultados em uma tabela
tabela_resultados = pd.DataFrame([
    {'Modelo': 'Média Histórica',  'MAE_Treino': mae_media_treino,   'MAE_Teste': mae_media_teste},
    {'Modelo': 'Média Acumulada',  'MAE_Treino': mae_acum_treino,    'MAE_Teste': mae_acum_teste},
    {'Modelo': 'SMA',              'MAE_Treino': mae_sma_treino,     'MAE_Teste': mae_sma_teste},
    {'Modelo': 'EMA',              'MAE_Treino': mae_ema_treino,     'MAE_Teste': mae_ema_teste},
    {'Modelo': 'Taxa de Variação', 'MAE_Treino': mae_taxa_treino,    'MAE_Teste': mae_taxa_teste},
    {'Modelo': 'Seasonal Naive',   'MAE_Treino': mae_snaive_treino,  'MAE_Teste': mae_snaive_teste},
    {'Modelo': 'Delta',            'MAE_Treino': mae_delta_treino,   'MAE_Teste': mae_delta_teste},
    {'Modelo': 'SARIMA (estático)','MAE_Treino': mae_sarima_treino,  'MAE_Teste': mae_sarima_teste},
    {'Modelo': 'SARIMA (rolling)', 'MAE_Treino': '-',                'MAE_Teste': mae_rolling_teste},
]).set_index('Modelo').sort_values('MAE_Teste')

print("\n📊 Tabela de Comparação — MAE por Modelo")
print(tabela_resultados.to_string())
```

---

## 10. Referências e Bibliotecas

```python
# Instalação (se necessário)
# pip install statsmodels scikit-learn pandas numpy matplotlib openpyxl

# Imports principais
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
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
```

> **Fonte:** Aulas 1–7, Disciplina Séries Temporais — Daniel Freitas, Instituto Germinare, 2026.
