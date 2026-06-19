import streamlit as st

from aurum.broker.ctrader.auth import CTraderAuth
from aurum.broker.ctrader.client import CTraderBroker
from aurum.config import (
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_REDIRECT_URI,
    CTRADER_SYMBOL,
    TRADING_MODE,
)
from aurum.execution.engine import ExecutionEngine
from aurum.execution.state import StateStore
from aurum.preflight import run_preflight


def render_fxpro_tab(deposit: float, risk_pct: float) -> None:
    st.markdown("### FxPro / cTrader — Авто-исполнение")
    st.caption(
        "FxPro работает на **cTrader**. Бот исполняет сигналы AURUM на **XAUUSD** "
        "с SL, частичным TP1 (50%) и полным TP2."
    )

    report = run_preflight()
    cols = st.columns(len(report.checks))
    for col, check in zip(cols, report.checks):
        col.metric(check.name.replace("_", " ").title(), "OK" if check.ok else "FAIL")

    mode = st.selectbox(
        "Режим торговли",
        ["paper", "demo", "live"],
        index=["paper", "demo", "live"].index(TRADING_MODE if TRADING_MODE in ("paper", "demo", "live") else "paper"),
    )
    if mode == "live":
        st.error("LIVE — реальные деньги. Только после 2+ недель успешного demo.")

    tab_setup, tab_trade, tab_history = st.tabs(["Подключение", "Торговля", "История"])

    with tab_setup:
        _render_oauth_setup()
        _render_connection_test(mode)

    with tab_trade:
        _render_trading_controls(mode, deposit, risk_pct)

    with tab_history:
        _render_trade_history()


def _render_oauth_setup() -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Open API")
        st.markdown(
            "1. [openapi.ctrader.com](https://openapi.ctrader.com) → Create App\n"
            "2. Scope: **trading** | Redirect: `http://localhost:8501`\n"
            "3. Client ID + Secret → `.env`"
        )
        client_id = st.text_input("Client ID", value=CTRADER_CLIENT_ID, type="password")
        client_secret = st.text_input("Client Secret", value=CTRADER_CLIENT_SECRET, type="password")
        redirect_uri = st.text_input("Redirect URI", value=CTRADER_REDIRECT_URI)
    with col2:
        st.markdown("#### OAuth")
        auth = CTraderAuth(client_id, client_secret, redirect_uri)
        if client_id and client_secret:
            st.link_button("Авторизовать cTrader", auth.get_auth_url())
            code = st.text_input("Auth code (?code=...)")
            if st.button("Сохранить токены") and code:
                try:
                    auth.exchange_code(code.strip())
                    st.success("Токены сохранены в data/ctrader_tokens.json")
                except Exception as exc:
                    st.error(str(exc))
        if st.button("Обновить access token"):
            try:
                auth.refresh()
                st.success("Token refreshed")
            except Exception as exc:
                st.error(str(exc))


def _render_connection_test(mode: str) -> None:
    st.markdown("#### Проверка")
    if mode == "paper":
        st.info("Paper mode — брокер не требуется")
        return
    if st.button("Проверить подключение к FxPro"):
        broker = CTraderBroker(host=mode)
        with st.spinner("Подключение к cTrader..."):
            if broker.connect():
                st.success(f"Подключено | Баланс: ${broker.get_balance():,.2f}")
                sym = broker.get_symbol()
                if sym:
                    st.info(f"{sym.name} (id={sym.symbol_id}) | min vol={sym.min_volume}")
                accounts = broker.get_accounts()
                if accounts:
                    st.dataframe(accounts, use_container_width=True)
                broker.disconnect()
            else:
                err = broker.get_last_error()
                st.error(f"Ошибка подключения{': ' + err if err else ''}")


def _render_trading_controls(mode: str, deposit: float, risk_pct: float) -> None:
    store = StateStore()
    state = store.load()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Режим", state.mode or mode)
    m2.metric("Баланс", f"${state.balance:,.0f}")
    m3.metric("Daily PnL", f"${state.daily_pnl:,.0f}")
    m4.metric("Символ", CTRADER_SYMBOL)

    if state.open_trade:
        t = state.open_trade
        st.success(
            f"**{t.side}** @ ${t.entry_price:.2f} | SL=${t.sl} TP1=${t.tp1} TP2=${t.tp2} "
            f"| TP1={'✓' if t.tp1_hit else '—'} | conf={t.confidence:.0%}"
        )
    else:
        st.info("Нет открытых позиций")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Запустить цикл", type="primary", use_container_width=True):
            with st.spinner("Анализ + исполнение..."):
                engine = ExecutionEngine(mode=mode, deposit=deposit, risk_pct=risk_pct)
                engine.connect()
                summary = engine.run_cycle()
                st.success(str(summary))
                if mode != "paper":
                    engine.disconnect()
    with c2:
        if st.button("🔄 Синхронизация", use_container_width=True):
            engine = ExecutionEngine(mode=mode, deposit=deposit, risk_pct=risk_pct)
            if engine.connect():
                engine.sync_with_broker()
                st.success("Синхронизировано")
                engine.disconnect()
    with c3:
        if st.button("🛑 Emergency Close", use_container_width=True):
            engine = ExecutionEngine(mode=mode, deposit=deposit, risk_pct=risk_pct)
            engine.connect()
            result = engine.emergency_close_all()
            st.warning(f"Emergency: {result}")
            engine.disconnect()

    st.markdown("**24/7:** `python run_trader.py` | Railway: `Procfile.worker`")


def _render_trade_history() -> None:
    state = StateStore().load()
    if state.trade_history:
        st.dataframe(state.trade_history[::-1], use_container_width=True)
        total = sum(t.get("pnl", 0) for t in state.trade_history)
        wins = sum(1 for t in state.trade_history if t.get("pnl", 0) > 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total PnL", f"${total:,.0f}")
        c2.metric("Win Rate", f"{wins / len(state.trade_history):.0%}")
        c3.metric("Сделок", len(state.trade_history))
    else:
        st.info("Сделок пока нет")
    if state.logs:
        st.markdown("#### Журнал")
        st.code("\n".join(state.logs[-40:]))
