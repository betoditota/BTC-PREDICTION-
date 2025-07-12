import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(page_title="CryptoVision BTC Forecast", layout="wide")

# Título
st.title("📈 CryptoVision Bitcoin Forecast")
st.write("Previsão do preço do Bitcoin usando o modelo ARIMA.")

# Função para buscar dados do BTC
@st.cache_data
def load_data():
    btc = yf.download("BTC-USD", period="60d", interval="1d")
    return btc

data = load_data()

# Mostrar gráfico histórico
st.subheader("Histórico do Bitcoin (últimos 60 dias)")
st.line_chart(data['Close'])

# ARIMA forecast
st.subheader("Previsão com ARIMA (3 dias)")

try:
    model = ARIMA(data['Close'], order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=3)
    forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 4)]

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast
    })

    # Mostrar resultados
    st.write(forecast_df)

    # Gráfico forecast
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Histórico')
    ax.plot(forecast_dates, forecast, label='Previsão', linestyle='--')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço USD')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro na previsão: {e}")

# Footer
st.markdown("---")
st.caption("🔗 Powered by CryptoVision ARIMA Engine | Streamlit Deployment")
