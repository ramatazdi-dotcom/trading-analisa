import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# =========================
# FUNGSI MAPPING TICKER → YAHOO FINANCE
# =========================
def map_ticker_to_yahoo(ticker: str, market_type: str) -> str:
    """
    Mengubah kode yang kamu ketik jadi kode yang dimengerti Yahoo Finance.
    - Forex : EURUSD -> EURUSD=X, XAUUSD -> GC=F, dll.
    - Saham Indonesia : BBCA -> BBCA.JK
    """
    if not ticker:
        return ticker

    t = ticker.strip().upper()

    if market_type == "Forex":
        special_map = {
            "XAUUSD": "GC=F",  # emas
            "XAGUSD": "SI=F",  # silver
        }
        if t in special_map:
            return special_map[t]

        if not t.endswith("=X"):
            return t + "=X"
        return t

    elif market_type == "Saham Indonesia":
        if not t.endswith(".JK"):
            return t + ".JK"
        return t

    else:
        # Crypto & Saham US biasanya langsung pakai saja
        return t


# =========================
# FUNGSI INDIKATOR & RISK
# =========================
def compute_rsi(series, period: int = 14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_rol = pd.Series(gain, index=series.index).rolling(window=period).mean()
    loss_rol = pd.Series(loss, index=series.index).rolling(window=period).mean()

    rs = gain_rol / loss_rol
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan EMA 10/20/50/100/200 ke dataframe."""
    close = df["Close"]
    df["EMA10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA100"] = close.ewm(span=100, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    return df


def generate_smart_signal(df: pd.DataFrame):
    """
    Sinyal lebih ketat berbasis EMA + RSI:
    - BUY: trend bullish (EMA50 > EMA200), momentum bullish (EMA10 crossover up EMA20),
      harga di atas EMA20, RSI di 45–70
    - SELL: trend bearish (EMA50 < EMA200), momentum bearish (EMA10 crossover down EMA20),
      harga di bawah EMA20, RSI di 30–55
    """
    if len(df) < 60:
        return "DATA KURANG", "Data belum cukup panjang untuk analisis EMA & RSI."

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Trend besar
    trend_bull = last["EMA50"] > last["EMA200"]
    trend_bear = last["EMA50"] < last["EMA200"]

    # Momentum (crossover EMA10 - EMA20)
    mom_bull = last["EMA10"] > last["EMA20"] and prev["EMA10"] <= prev["EMA20"]
    mom_bear = last["EMA10"] < last["EMA20"] and prev["EMA10"] >= prev["EMA20"]

    price_above_ema20 = last["Close"] > last["EMA20"]
    price_below_ema20 = last["Close"] < last["EMA20"]

    rsi = last["RSI"]

        # BUY setup ketat (RSI 45–80, overbought di atas 80)
    if (
        trend_bull
        and mom_bull
        and price_above_ema20
        and 45 <= rsi <= 80
    ):
        return "BUY", (
            "Trend besar bullish (EMA50 > EMA200), momentum baru menguat "
            "(EMA10 cross up EMA20), harga di atas EMA20, dan RSI mendukung (45–80)."
        )

    # SELL setup ketat (RSI 20–55, oversold di bawah 20)
    if (
        trend_bear
        and mom_bear
        and price_below_ema20
        and 20 <= rsi <= 55
    ):
        return "SELL", (
            "Trend besar bearish (EMA50 < EMA200), momentum baru melemah "
            "(EMA10 cross down EMA20), harga di bawah EMA20, dan RSI mendukung (20–55)."
        )


    # SELL setup ketat
    if (
        trend_bear
        and mom_bear
        and price_below_ema20
        and 30 <= rsi <= 55
    ):
        return "SELL", (
            "Trend besar bearish (EMA50 < EMA200), momentum baru melemah "
            "(EMA10 cross down EMA20), harga di bawah EMA20, dan RSI mendukung (30–55)."
        )

    # WAIT (penjelasan berdasarkan kondisi)
    if trend_bull and not mom_bull:
        return "WAIT", (
            "Trend besar masih bullish (EMA50 > EMA200), namun momentum belum memberi sinyal kuat "
            "(EMA10 belum cross up EMA20) atau harga belum cukup kuat di atas EMA20."
        )
    if trend_bear and not mom_bear:
        return "WAIT", (
            "Trend besar masih bearish (EMA50 < EMA200), namun momentum belum memberi sinyal kuat "
            "(EMA10 belum cross down EMA20) atau harga belum cukup lemah di bawah EMA20."
        )

    return "WAIT", (
        "Kondisi EMA dan RSI belum searah untuk setup BUY/SELL yang rapi. "
        "Lebih aman menunggu struktur yang lebih jelas."
    )


def compute_position_sizing(balance, risk_pct, stop_pct, price):
    if balance <= 0 or risk_pct <= 0 or stop_pct <= 0 or price <= 0:
        return None, None, None, None

    risk_amount = balance * (risk_pct / 100.0)
    risk_per_unit = price * (stop_pct / 100.0)

    if risk_per_unit == 0:
        return None, None, None, None

    qty = risk_amount / risk_per_unit
    sl_price = price * (1 - stop_pct / 100.0)
    tp_price = price + (price - sl_price) * 2
    return risk_amount, qty, sl_price, tp_price


def position_from_entry_sl(balance, risk_pct, entry_price, sl_price, tp_price=None):
    """Dipakai di mode screenshot: hitung size dari entry & SL."""
    if balance <= 0 or risk_pct <= 0:
        return None, None, None

    if entry_price <= 0 or sl_price <= 0 or entry_price == sl_price:
        return None, None, None

    risk_amount = balance * (risk_pct / 100.0)
    risk_per_unit = abs(entry_price - sl_price)
    qty = risk_amount / risk_per_unit

    rr = None
    if tp_price and tp_price != entry_price and tp_price > 0:
        rr = abs(tp_price - entry_price) / abs(entry_price - sl_price)

    return risk_amount, qty, rr


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Trader Analyzer (EMA + RSI)",
    layout="wide"
)

from PIL import Image

logo = Image.open("app_logo.png")
st.sidebar.image(logo, width=160)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Pengaturan")

    # PRESET GAYA TRADING
    st.markdown("### Preset Gaya Trading")
    trading_style = st.selectbox(
        "Pilih gaya trading",
        ["Custom", "Scalping", "Intraday", "Swing", "Long-term"],
        index=2  # default Intraday
    )

    # Default nilai berdasarkan preset
    if trading_style == "Scalping":
        default_risk_pct = 1.0
        default_stop_pct = 1.0
        tf_hint = "Rekomendasi: 1m – 15m, fokus EMA10 & EMA20."
    elif trading_style == "Intraday":
        default_risk_pct = 1.5
        default_stop_pct = 2.0
        tf_hint = "Rekomendasi: 15m – 1H, kombinasi EMA20 & EMA50."
    elif trading_style == "Swing":
        default_risk_pct = 2.0
        default_stop_pct = 4.0
        tf_hint = "Rekomendasi: 4H – 1D, fokus EMA50/100/200."
    elif trading_style == "Long-term":
        default_risk_pct = 1.0
        default_stop_pct = 7.0
        tf_hint = "Rekomendasi: 1W – 1M, lihat EMA100 & EMA200."
    else:
        default_risk_pct = 2.0
        default_stop_pct = 3.0
        tf_hint = "Pilih sendiri timeframe & EMA sesuai gaya trading."

    st.caption(tf_hint)

    st.markdown("### Jenis Pasar")
    market_type = st.selectbox(
        "Pilih jenis pasar",
        ["Crypto", "Saham US", "Saham Indonesia", "Forex"],
        index=0
    )

    st.markdown("Contoh ticker:")
    if market_type == "Crypto":
        st.caption("- BTC-USD, ETH-USD, SOL-USD")
    elif market_type == "Saham US":
        st.caption("- AAPL, TSLA, MSFT, NVDA")
    elif market_type == "Saham Indonesia":
        st.caption("- BBCA, BBRI, TLKM (otomatis jadi .JK)")
    else:
        st.caption("- EURUSD, USDJPY, GBPUSD, XAUUSD")

    st.markdown("---")
    st.markdown("### Money & Risk Management")

    balance = st.number_input(
        "Total modal (dalam USD / IDR ekuivalen)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    # gunakan default dari preset sebagai value awal
    risk_pct = st.slider(
        "Risiko per trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=float(default_risk_pct),
        step=0.5
    )

    stop_pct = st.slider(
        "Jarak Stop Loss (%) dari harga entry (mode data harga)",
        min_value=1.0,
        max_value=15.0,
        value=float(default_stop_pct),
        step=0.5
    )

    st.caption(
        "Preset hanya mengisi nilai awal. "
        "Kamu masih bisa mengubah risk & stop loss sesuai kebutuhan."
    )

    st.caption(
        "Catatan: ini hanya kalkulasi umum untuk bantu manajemen risiko, "
        "bukan saran finansial."
    )

# =========================
# HEADER & MODE
# =========================
st.markdown(
    """
    <h1 style="margin-bottom:0px;">Trader Analyzer <span style="font-size:60%;">(EMA + RSI)</span></h1>
    <p style="color:#aaaaaa;">Alat bantu tambahan analisis teknikal sederhana untuk crypto, saham, dan forex.
    <b>Bukan</b> rekomendasi resmi suatu instrumen tertentu.</p>
    """,
    unsafe_allow_html=True
)

mode = st.radio(
    "Pilih cara analisis:",
    ["📡 Data harga (Yahoo Finance)", "🖼️ Screenshot dari TradingView"],
    horizontal=True
)

st.markdown("---")

# =====================================================
# MODE 1 – DATA HARGA (YAHOO FINANCE)
# =====================================================
if mode.startswith("📡"):
    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])

    with top_col1:
        ticker = st.text_input(
            "Ticker / Kode (lihat contoh di sidebar)",
            value="BTC-USD"
        )
    with top_col2:
        period = st.selectbox(
            "Periode data",
            ["1mo", "3mo", "6mo", "1y"],
            index=1
        )
    with top_col3:
        interval = st.selectbox(
            "Interval candle",
            ["1d", "4h", "1h", "30m", "15m"],
            index=0
        )

    run_btn = st.button("🚀 Ambil & Analisis Data", type="primary")

    if run_btn:
        if not ticker:
            st.error("Masukkan ticker terlebih dahulu.")
        else:
            yahoo_ticker = map_ticker_to_yahoo(ticker, market_type)

            with st.spinner(f"Mengambil data untuk {yahoo_ticker} ..."):
                data = yf.download(yahoo_ticker, period=period, interval=interval)

            if data.empty:
                st.error("Data tidak ditemukan. Coba ticker atau periode lain.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                st.success(f"Data berhasil diambil untuk: {ticker}")
                st.caption(f"Menggunakan kode Yahoo Finance: **{yahoo_ticker}**")

                # Indikator: EMA + RSI
                data = compute_emas(data)
                data["RSI"] = compute_rsi(data["Close"], period=14)

                data_ind = data.dropna(subset=["EMA10", "EMA20", "EMA50", "EMA200", "RSI"])

                if data_ind.empty:
                    st.warning("Data masih terlalu sedikit untuk menghitung indikator.")
                else:
                    signal, reason = generate_smart_signal(data_ind)
                    last_row = data_ind.iloc[-1]

                    chart_col, signal_col = st.columns([2.2, 1.1])

                    with chart_col:
                        st.subheader("📈 Grafik Harga (Candlestick) & EMA")

                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=data_ind.index,
                            open=data_ind["Open"],
                            high=data_ind["High"],
                            low=data_ind["Low"],
                            close=data_ind["Close"],
                            name="Harga"
                        ))
                        # Tampilkan beberapa EMA utama
                        fig.add_trace(go.Scatter(
                            x=data_ind.index,
                            y=data_ind["EMA20"],
                            mode="lines",
                            name="EMA20"
                        ))
                        fig.add_trace(go.Scatter(
                            x=data_ind.index,
                            y=data_ind["EMA50"],
                            mode="lines",
                            name="EMA50"
                        ))
                        fig.add_trace(go.Scatter(
                            x=data_ind.index,
                            y=data_ind["EMA200"],
                            mode="lines",
                            name="EMA200"
                        ))
                        fig.update_layout(
                            xaxis_title="Tanggal",
                            yaxis_title="Harga",
                            xaxis_rangeslider_visible=False,
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("📉 RSI (14)")
                        st.line_chart(data_ind[["RSI"]])

                    with signal_col:
                        st.subheader("🧠 Sinyal Keputusan (EMA + RSI)")
                        if signal == "BUY":
                            st.markdown("## ✅ Sinyal: **BUY**")
                        elif signal == "SELL":
                            st.markdown("## ❌ Sinyal: **SELL**")
                        elif signal == "WAIT":
                            st.markdown("## ⏳ Sinyal: **WAIT**")
                        else:
                            st.markdown("## ⚠️ Sinyal: **DATA KURANG**")
                        st.write(reason)

                        st.markdown("---")
                        st.markdown("**Ringkasan Harga Terakhir:**")
                        st.write(f"Close: {last_row['Close']:.4f}")
                        st.write(f"High: {last_row['High']:.4f}")
                        st.write(f"Low: {last_row['Low']:.4f}")
                        st.write(f"Volume: {last_row['Volume']}")

                        st.markdown("**EMA Terakhir:**")
                        st.write(
                            f"EMA10: {last_row['EMA10']:.4f} | "
                            f"EMA20: {last_row['EMA20']:.4f} | "
                            f"EMA50: {last_row['EMA50']:.4f}"
                        )
                        st.write(
                            f"EMA100: {last_row['EMA100']:.4f} | "
                            f"EMA200: {last_row['EMA200']:.4f}"
                        )

                        # RISK MANAGEMENT – MODE DATA HARGA
                        st.markdown("---")
                        st.subheader("🛡️ Money & Risk Management (otomatis)")

                        entry_price = st.number_input(
                            "Harga entry (default = Close terakhir)",
                            min_value=0.0,
                            value=float(last_row["Close"]),
                            step=float(max(last_row["Close"] * 0.001, 0.0001))
                        )

                        risk_amount, qty, sl_price, tp_price = compute_position_sizing(
                            balance, risk_pct, stop_pct, entry_price
                        )

                        if None in (risk_amount, qty, sl_price, tp_price):
                            st.info("Lengkapi nilai modal / risiko / stop loss untuk melihat kalkulasi.")
                        else:
                            st.write(f"Modal: **{balance:,.2f}**")
                            st.write(f"Risiko per trade: **{risk_pct:.1f}%** → ~**{risk_amount:,.2f}**")
                            st.write(f"Perkiraan jumlah unit yang boleh dibeli: **{qty:,.4f}**")

                            st.markdown("**Level Harga Penting (berdasarkan % SL):**")
                            st.write(f"- Entry: **{entry_price:.4f}**")
                            st.write(f"- Stop Loss (SL): **{sl_price:.4f}**")
                            st.write(f"- Take Profit (TP, ± 1:2 RR): **{tp_price:.4f}**")

                            st.caption(
                                "Ini hanya simulasi kalkulasi sederhana. "
                                "Sesuaikan lagi dengan kondisi broker/exchange dan rencana pribadi."
                            )

    st.caption(
        "Disclaimer: Mode ini menggunakan data dari Yahoo Finance. "
        "Harga bisa sedikit berbeda dengan broker / exchange kamu."
        "Keputusan sepenuhnya menjadi tanggung jawab kamu ya..."
    )

# =====================================================
# MODE 2 – SCREENSHOT TRADINGVIEW
# =====================================================
else:
    st.subheader("🖼️ Analisis dari Screenshot TradingView")

    st.markdown(
        "- Upload screenshot chart dari TradingView\n"
        "- Lalu isi form analisa (trend, timeframe, entry/SL/TP)\n"
        "- Aplikasi akan bantu merapikan analisa & menghitung ukuran posisi."
    )

    uploaded = st.file_uploader(
        "Upload screenshot (PNG/JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Screenshot TradingView", use_container_width=True)

        st.markdown("### Informasi Chart")

        col_a, col_b = st.columns(2)
        with col_a:
            pair_name = st.text_input("Nama pair / symbol", value="XAUUSD")
            timeframe = st.selectbox(
                "Timeframe pada chart",
                ["1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W", "1M"],
            )
        with col_b:
            trend_view = st.selectbox(
                "Trend yang kamu lihat",
                ["Naik (uptrend)", "Turun (downtrend)", "Sideways"],
                index=1
            )
            notes = st.text_area(
                "Catatan pola / zona (support/resistance, pola candle, dll.)",
                placeholder="Contoh: harga baru reject dari resistance, ada bearish engulfing dekat supply zone..."
            )

        st.markdown("### Level Harga (dari chart TradingView)")
        col_e, col_s, col_t = st.columns(3)
        with col_e:
            entry_price = st.number_input("Entry", min_value=0.0, value=0.0, format="%.5f")
        with col_s:
            sl_manual = st.number_input("Stop Loss (SL)", min_value=0.0, value=0.0, format="%.5f")
        with col_t:
            tp_manual = st.number_input("Take Profit (TP)", min_value=0.0, value=0.0, format="%.5f")

        st.markdown("### Ringkasan & Risk Management")

        if entry_price > 0 and sl_manual > 0 and entry_price != sl_manual:
            risk_amount, qty, rr = position_from_entry_sl(
                balance, risk_pct, entry_price, sl_manual, tp_manual if tp_manual > 0 else None
            )

            st.write(f"Pair: **{pair_name}** | Timeframe: **{timeframe}**")
            st.write(f"Trend yang kamu lihat: **{trend_view}**")

            if notes:
                st.markdown("**Catatan analisa:**")
                st.write(notes)

            st.markdown("---")
            st.write(f"Modal: **{balance:,.2f}**")
            st.write(f"Risiko per trade: **{risk_pct:.1f}%** → ~**{risk_amount:,.2f}**")
            st.write(f"Perkiraan jumlah unit (lot/koin) yang boleh dibuka: **{qty:,.4f}**")

            st.markdown("**Level harga yang kamu rencanakan:**")
            st.write(f"- Entry: **{entry_price:.5f}**")
            st.write(f"- Stop Loss: **{sl_manual:.5f}**")
            if tp_manual > 0:
                st.write(f"- Take Profit: **{tp_manual:.5f}**")
            if rr is not None:
                st.write(f"- Perkiraan Risk:Reward (RR): **{rr:.2f} : 1**")

            st.caption(
                "Perhitungan berdasarkan level yang kamu baca sendiri dari TradingView. "
                "Aplikasi ini hanya membantu mengubahnya jadi rencana & ukuran posisi."
            )
        else:
            st.info("Isi dulu entry & stop loss (minimal). TP opsional tapi disarankan.")
    else:
        st.info("Silakan upload screenshot chart TradingView terlebih dahulu.")

    st.caption(
        "Disclaimer: Mode ini tidak membaca gambar secara otomatis. "
        "Kamu tetap melakukan analisa visual di TradingView, aplikasi hanya membantu merapikan & menghitung risiko."
    )
