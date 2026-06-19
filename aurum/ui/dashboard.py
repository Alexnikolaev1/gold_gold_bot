import streamlit as st

from aurum.execution.state import StateStore
from aurum.preflight import run_preflight


def render_performance_sidebar() -> None:
    state = StateStore().load()
    if not state.trade_history:
        return
    wins = [t for t in state.trade_history if t.get("pnl", 0) > 0]
    total_pnl = sum(t.get("pnl", 0) for t in state.trade_history)
    win_rate = len(wins) / len(state.trade_history) if state.trade_history else 0
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Trading Performance**")
    st.sidebar.metric("Total PnL", f"${total_pnl:,.0f}")
    st.sidebar.metric("Win Rate", f"{win_rate:.0%}")
    st.sidebar.metric("Сделок", len(state.trade_history))
    st.sidebar.metric("Daily PnL", f"${state.daily_pnl:,.0f}")


def render_preflight_banner() -> None:
    report = run_preflight()
    failed = [c for c in report.checks if not c.ok]
    if failed:
        with st.expander(f"⚠️ Preflight: {len(failed)} issue(s)", expanded=False):
            for check in report.checks:
                icon = "✅" if check.ok else "❌"
                st.write(f"{icon} **{check.name}**: {check.detail}")
