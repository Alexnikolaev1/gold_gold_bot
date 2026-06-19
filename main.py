import datetime

import streamlit as st

from aurum import __version__
from aurum.backtest import run_backtest, run_backtest_sweep
from aurum.chart import generate_chart, generate_equity_chart
from aurum.config import (
    AUTO_REFRESH_SEC,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_LOOKBACK,
    DEFAULT_DEPOSIT,
    DEFAULT_RISK_PCT,
    MIN_BARS_FOR_FEATURES,
    MIN_DEPOSIT,
    MIN_RULES_CONSENSUS,
    TICKER_GOLD,
)
from aurum.data import fetch_historical_data, fetch_realtime_data, fetch_training_data, is_comex_session_active
from aurum.features import calculate_indicators
from aurum.ml import load_meta, models_exist, train_models
from aurum.notifications import format_signal_alert, send_telegram_alert
from aurum.risk import calculate_trade_levels, position_size_oz
from aurum.ui.dashboard import render_performance_sidebar, render_preflight_banner
from aurum.ui.fxpro_panel import render_fxpro_tab
from aurum.signals import consensus_score, process_signals

st.set_page_config(
    page_title="AURUM QUANT TERMINAL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #E5E9F0; }
    h1, h2, h3 { color: #FFD700 !important; font-family: 'Courier New', monospace; }
    .metric-box {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border: 1px solid #FFD700; padding: 15px; border-radius: 8px; text-align: center;
    }
    .signal-buy { color: #00FF66; font-size: 42px; font-weight: bold; font-family: 'Courier New', monospace; }
    .signal-sell { color: #FF3333; font-size: 42px; font-weight: bold; font-family: 'Courier New', monospace; }
    .signal-hold { color: #888888; font-size: 42px; font-weight: bold; font-family: 'Courier New', monospace; }
    .disclaimer { color: #888; font-size: 12px; border-left: 3px solid #FFD700; padding-left: 10px; margin: 10px 0; }
    div[data-testid="stSidebar"] { background-color: #111827; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_training():
    return fetch_training_data()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_realtime():
    return fetch_realtime_data()


def _ensure_models():
    if models_exist():
        return True
    st.warning("Первичный запуск: обучение ML-моделей (2–3 мин)...")
    hist = _cached_training()
    if hist.empty:
        st.error("Не удалось загрузить данные Yahoo Finance. Проверьте интернет.")
        return False
    with st.spinner("Калибровка XGBoost + isotonic scaling..."):
        meta = train_models(hist)
    st.success(
        f"Модели обучены на {meta.samples} барах | "
        f"Buy prec: {meta.buy_precision:.0%} | Sell prec: {meta.sell_precision:.0%}"
    )
    return True


def _render_header():
    session_ok, session_msg = is_comex_session_active()
    status_color = "#00FF66" if session_ok else "#FF3333"
    st.title(f"AURUM QUANTITATIVE TERMINAL [{TICKER_GOLD}]")
    st.caption(f"v{__version__} • Institutional Gold Pipeline • {session_msg}")
    st.markdown(
        f'<p class="disclaimer">⚠️ Confidence — percentile-ранг модели среди последних {CONFIDENCE_LOOKBACK} баров, не гарантия прибыли. '
        f"Порог сигнала: {CONFIDENCE_THRESHOLD:.0%} percentile + консенсус {MIN_RULES_CONSENSUS}/6 правил. "
        f"Торгуйте только на демо до подтверждения edge на бэктесте.</p>",
        unsafe_allow_html=True,
    )
    return session_ok


def main():
    _render_header()
    render_preflight_banner()

    st.sidebar.header("CONTROL CORE")
    deposit = st.sidebar.number_input(
        "Депозит ($)",
        min_value=int(MIN_DEPOSIT),
        max_value=1_000_000,
        value=int(DEFAULT_DEPOSIT),
        step=50,
        help=f"Минимум ${MIN_DEPOSIT:.0f}. Размер позиции считается от этого депозита и риска %.",
    )
    risk_pct = st.sidebar.slider("Риск на сделку (%)", 0.1, 5.0, DEFAULT_RISK_PCT, 0.1)
    auto_refresh = st.sidebar.toggle("Авто-обновление", value=True)
    notify_tg = st.sidebar.toggle("Telegram-алерты", value=False)

    if st.sidebar.button("ПЕРЕОБУЧИТЬ ML"):
        with st.spinner("Загрузка истории и переобучение..."):
            hist = fetch_historical_data()
            if not hist.empty:
                _cached_training.clear()
                meta = train_models(hist)
                st.sidebar.success(f"Готово | Buy prec: {meta.buy_precision:.0%}")
            else:
                st.sidebar.error("Ошибка загрузки данных.")

    if not _ensure_models():
        return

    meta = load_meta()
    if meta:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**ML Метрики (holdout)**")
        st.sidebar.metric("Buy precision", f"{meta.buy_precision:.1%}")
        st.sidebar.metric("Sell precision", f"{meta.sell_precision:.1%}")
        st.sidebar.metric("Обучающих баров", meta.samples)

    render_performance_sidebar()

    with st.spinner("Синхронизация котировок COMEX..."):
        df = _cached_realtime()

    if df.empty or len(df) < MIN_BARS_FOR_FEATURES:
        st.error("Недостаточно данных. Дождитесь открытия сессии или проверьте тикер.")
        return

    df_analyzed = calculate_indicators(df)
    current = df_analyzed.iloc[-1]
    result = process_signals(df_analyzed)
    levels = calculate_trade_levels(df_analyzed, result.signal)

    signal = result.signal
    confidence = result.confidence
    price = float(current["Close"])

    oz = position_size_oz(deposit, risk_pct, levels.risk_per_oz) if signal in ("BUY", "SELL") else 0.0

    if notify_tg and signal in ("BUY", "SELL") and confidence >= CONFIDENCE_THRESHOLD:
        last_key = f"last_alert_{signal}"
        last_price = st.session_state.get(last_key, 0)
        if abs(price - last_price) > 1.0:
            msg = format_signal_alert(signal, price, confidence, levels.tp1, levels.tp2, levels.sl, oz)
            if send_telegram_alert(msg):
                st.session_state[last_key] = price
                st.sidebar.success("Telegram: сигнал отправлен")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-box">
            <span style='color:#888;font-size:14px'>ЦЕНА GOLD</span><br>
            <span style='font-size:32px;font-weight:bold;color:#FFF'>${price:.2f}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        sig_class = {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-hold")
        st.markdown(
            f"""<div class="metric-box">
            <span style='color:#888;font-size:14px'>СИГНАЛ</span><br>
            <span class="{sig_class}">{signal}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        conf_color = "#FFD700" if confidence >= CONFIDENCE_THRESHOLD else "#E5E9F0"
        st.markdown(
            f"""<div class="metric-box">
            <span style='color:#888;font-size:14px'>CONFIDENCE</span><br>
            <span style='font-size:32px;font-weight:bold;color:{conf_color}'>{confidence*100:.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        pos_text = f"{oz:.1f} oz" if oz > 0 else "Ожидание"
        st.markdown(
            f"""<div class="metric-box">
            <span style='color:#888;font-size:14px'>ПОЗИЦИЯ</span><br>
            <span style='font-size:32px;font-weight:bold;color:#00DDFF'>{pos_text}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("P(Buy) raw", f"{result.p_buy*100:.1f}%")
    p2.metric("P(Sell) raw", f"{result.p_sell*100:.1f}%")
    p3.metric("Conf Buy %ile", f"{result.conf_buy*100:.1f}%")
    p4.metric("Conf Sell %ile", f"{result.conf_sell*100:.1f}%")
    p5.metric("Консенсус", f"{result.rules_buy}/{result.rules_sell}")

    if signal in ("BUY", "SELL"):
        st.markdown("### Спецификация ордера")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Stop Loss", f"${levels.sl}", delta=f"{levels.sl - price:.2f}", delta_color="inverse")
        o2.metric("Take Profit 1", f"${levels.tp1}", delta=f"{levels.tp1 - price:.2f}")
        o3.metric("Take Profit 2", f"${levels.tp2}", delta=f"{levels.tp2 - price:.2f}")
        o4.metric("R:R → TP1", f"{levels.risk_reward_tp1:.2f}")

    tab_chart, tab_rules, tab_backtest, tab_fxpro, tab_log = st.tabs(
        ["График", "Правила", "Бэктест", "FxPro", "Лог"]
    )

    with tab_chart:
        st.plotly_chart(generate_chart(df_analyzed, levels.tp1, levels.tp2, levels.sl, signal), use_container_width=True)

    with tab_rules:
        st.markdown("#### Активные факторы консенсуса")
        if result.rule_details:
            for detail in result.rule_details:
                st.success(detail)
        else:
            st.info("Нет срабатываний правил на текущем баре.")
        st.progress(consensus_score(result), text=f"Сила консенсуса: {consensus_score(result)*100:.0f}%")

    with tab_backtest:
        bt_threshold = st.slider(
            "Порог confidence для бэктеста",
            0.55,
            0.95,
            CONFIDENCE_THRESHOLD,
            0.05,
            help="Live-сигналы используют порог из настроек (90%). Здесь можно проверить другие уровни.",
        )
        if st.button("Запустить бэктест на истории"):
            with st.spinner("Симуляция сделок..."):
                hist = _cached_training()
                bt = run_backtest(hist, deposit, risk_pct, confidence_threshold=bt_threshold)
                sweep = run_backtest_sweep(hist, deposit, risk_pct)
            if bt and bt.total_trades > 0:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Сделок", bt.total_trades)
                m2.metric("Win Rate", f"{bt.win_rate:.1%}")
                m3.metric("Profit Factor", f"{bt.profit_factor:.2f}")
                m4.metric("Total PnL", f"${bt.total_pnl:,.0f}")
                m5, m6, m7 = st.columns(3)
                m5.metric("Max Drawdown", f"{bt.max_drawdown:.1%}")
                m6.metric("Sharpe (ann.)", f"{bt.sharpe:.2f}")
                m7.metric("Avg R:R", f"{bt.avg_rr:.2f}")
                st.plotly_chart(generate_equity_chart(bt.equity_curve), use_container_width=True)
                st.dataframe(bt.trades.tail(20), use_container_width=True)
            else:
                st.warning(
                    f"При пороге {bt_threshold:.0%} сделок не найдено. "
                    "Снизьте порог слайдером или переобучите модель."
                )
            if not sweep.empty:
                st.markdown("#### Сравнение порогов confidence")
                st.dataframe(sweep, use_container_width=True, hide_index=True)

    with tab_fxpro:
        render_fxpro_tab(deposit, risk_pct)

    with tab_log:
        log_time = datetime.datetime.now().strftime("%H:%M:%S")
        logs = [
            f"[{log_time}] Баров обработано: {len(df_analyzed)}",
            f"[{log_time}] Z-Corr DXY: {float(current['Corr_DXY_Z']):.4f}",
            f"[{log_time}] POC: {'ВЫШЕ' if price > current['POC'] else 'НИЖЕ'} (${float(current['POC']):.2f})",
            f"[{log_time}] RSI: {float(current['RSI']):.1f} | BB Squeeze: {float(current['BB_Squeeze']):.4f}",
            f"[{log_time}] Сигнал: {signal} | conf={confidence:.3f}",
        ]
        st.text_area("System Log", "\n".join(logs), height=150, label_visibility="collapsed")

    st.caption(f"Обновлено: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh:
        st.markdown(
            f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SEC}">',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
