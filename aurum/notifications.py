import logging

from aurum.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_telegram_alert(message: str) -> bool:
    from aurum.telegram_bot import load_chat_id, send_message

    chat_id = load_chat_id() or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False
    return send_message(chat_id, message)


def format_signal_alert(
    signal: str,
    price: float,
    confidence: float,
    tp1: float,
    tp2: float,
    sl: float,
    oz: float,
) -> str:
    return format_entry_alert(
        signal=signal,
        price=price,
        confidence=confidence,
        tp1=tp1,
        tp2=tp2,
        sl=sl,
        oz=oz,
    )


def format_entry_alert(
    signal: str,
    price: float,
    confidence: float,
    tp1: float,
    tp2: float,
    sl: float,
    oz: float,
    rr_tp1: float = 0.0,
    rules_active: int = 0,
    total_rules: int = 6,
    bar_time: str = "",
) -> str:
    emoji = "🟢" if signal == "BUY" else "🔴"
    lines = [
        f"{emoji} <b>AURUM ENTRY — {signal}</b>",
        f"Цена входа: <b>${price:.2f}</b>",
        f"Confidence: <b>{confidence * 100:.1f}%</b>",
        f"SL: <b>${sl:.1f}</b> | TP1: <b>${tp1:.1f}</b> | TP2: <b>${tp2:.1f}</b>",
    ]
    if rr_tp1 > 0:
        lines.append(f"R:R → TP1: <b>{rr_tp1:.2f}</b>")
    if rules_active > 0:
        lines.append(f"Консенсус: <b>{rules_active}/{total_rules}</b> правил")
    lines.append(f"Размер: <b>{oz:.2f} oz</b> (риск от депозита)")
    if bar_time:
        lines.append(f"Бар: {bar_time}")
    lines.append("\n⚡ Сигнал на вход — проверьте терминал или /balance")
    return "\n".join(lines)
