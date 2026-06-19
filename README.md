# AURUM v2.2 — Gold Quant Terminal + FxPro Auto-Trader

Профессиональный терминал для золота **GC=F / XAUUSD** с ML-сигналами, бэктестом и **авто-исполнением на FxPro (cTrader Open API)**.

## Быстрый старт

```bash
pip install -r requirements.txt
python preflight.py          # проверка системы
streamlit run main.py          # терминал
python run_trader.py           # paper auto-trader 24/7
```

## Режимы торговли

| Режим | Описание |
|-------|----------|
| `paper` | Симуляция, без риска (по умолчанию) |
| `demo` | FxPro demo через cTrader API |
| `live` | Реальные деньги — только после demo |

## FxPro подключение

1. Счёт **cTrader** у [FxPro](https://www.fxpro.com)
2. App на [openapi.ctrader.com](https://openapi.ctrader.com) (scope: trading)
3. `.env` → Client ID, Secret
4. UI → вкладка **FxPro** → OAuth → проверка подключения
5. `TRADING_MODE=demo python run_trader.py`

## Архитектура

```
gold_bot/
├── main.py              # Streamlit терминал
├── run_trader.py        # 24/7 worker
├── preflight.py         # health checks
├── aurum/
│   ├── broker/ctrader/  # FxPro API
│   ├── execution/       # Signal → order engine
│   ├── preflight.py     # System checks
│   └── ui/              # Dashboard + FxPro panel
└── data/                # tokens, state, logs
```

## Риск-менеджмент

- Percentile confidence ≥ 90% + 3/6 правил консенсуса
- Min R:R ≥ 1.2 перед входом
- SL за swing / ATR | TP1 partial 50% → breakeven | TP2 full
- Daily loss limit 3% | Cooldown 6 баров | Max 1 позиция

## Railway

- **Web:** `Procfile` (Streamlit)
- **Worker:** `Procfile.worker` (auto-trader)
- Env vars из `.env.example`

MIT. Торгуйте на свой страх и риск.
