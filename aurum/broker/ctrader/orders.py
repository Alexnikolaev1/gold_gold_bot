def price_distance_to_relative(distance: float) -> int:
    """Convert price distance to cTrader relative SL/TP units (1/100000 of price)."""
    return int(round(abs(distance) * 100_000))


def oz_to_protocol_volume(oz: float, lot_size_cents: int, step_volume: int, min_volume: int) -> int:
    """
    Convert ounces to cTrader volume in cents.
    lot_size_cents: symbol lot size from ProtoOASymbol (typically 10000 = 100 oz).
    """
    lot_oz = lot_size_cents / 100.0
    if lot_oz <= 0:
        lot_oz = 100.0
    lots = oz / lot_oz
    volume = int(round(lots * 100))
    volume = max(min_volume, volume)
    if step_volume > 0:
        volume = (volume // step_volume) * step_volume
    return max(min_volume, volume)


def protocol_volume_to_oz(volume: int, lot_size_cents: int) -> float:
    lot_oz = lot_size_cents / 100.0
    if lot_oz <= 0:
        lot_oz = 100.0
    return (volume / 100.0) * lot_oz
