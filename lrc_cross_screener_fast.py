# lrc_cross_screener_fast.py
# Binance SPOT + FUTURES(PERP): USDT paritelerinde LRC orta çizgi kesişmeleri (High LRC vs Low LRC)
# CIKTI: SADECE Telegram (Excel yok)
# - Mesajlar timeframe'e gore AYRI AYRI gonderilir
# - Coin format: Spot -> XXXUSDT, Perp -> XXXUSDT.P
# - Sadece USDT pariteleri
# - Alfabetik siralama

import os
import time
import math
from datetime import datetime, timezone

import numpy as np
import requests


# ===================== AYARLAR =====================
LRC_LEN = int(os.getenv("LRC_LEN", "298"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))

TIMEFRAMES = {
    "1d": {"bin": "1d", "bars_per_day": 1},
    "4h": {"bin": "4h", "bars_per_day": 6},
    "3h": {"bin": "3h", "bars_per_day": 8},
    "2h": {"bin": "2h", "bars_per_day": 12},
    "1h": {"bin": "1h", "bars_per_day": 24},
}

SLEEP_BETWEEN_REQUESTS = float(os.getenv("SLEEP_BETWEEN_REQUESTS", "0.12"))

# Stabil ve turev token filtreleri
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI",
    "UST", "USTC", "PAX", "PAXG",
}
BANNED_SUBSTRINGS = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ===================== HTTP =====================
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def get_json(url, timeout=30, max_retries=6):
    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (418, 429):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 20.0)
    return None


# ===================== SYMBOL LISTS =====================
def get_futures_usdt_perp_symbols_filtered():
    data = get_json("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20)
    if not data:
        return []

    out = []
    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue

        sym = s.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if any(b in sym for b in BANNED_SUBSTRINGS):
            continue

        base = sym[:-4]
        if base in STABLE_BASES:
            continue

        out.append(sym)

    return sorted(out)


def get_spot_usdt_symbols_filtered():
    data = get_json("https://api.binance.com/api/v3/exchangeInfo", timeout=20)
    if not data:
        return []

    out = []
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue

        sym = s.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if any(b in sym for b in BANNED_SUBSTRINGS):
            continue

        base = sym[:-4]
        if base in STABLE_BASES:
            continue

        out.append(sym)

    return sorted(set(out))


# ===================== KLINES =====================
def fetch_klines(symbol, interval, limit, futures=False):
    if futures:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    else:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    data = get_json(url, timeout=30)
    return data if isinstance(data, list) else []


# ===================== LRC (LINREG) =====================
def rolling_linreg_end_values(arr, length):
    n = len(arr)
    out = np.full(n, np.nan)
    if n < length:
        return out

    x = np.arange(length)
    x_mean = x.mean()
    denom = float(((x - x_mean) ** 2).sum())

    for i in range(length - 1, n):
        y = np.array(arr[i - length + 1: i + 1], dtype=float)
        y_mean = float(y.mean())
        cov = float(((x - x_mean) * (y - y_mean)).sum())
        slope = cov / denom
        intercept = y_mean - slope * x_mean
        out[i] = slope * (length - 1) + intercept

    return out


def detect_last_cross_from_klines(klines, lrc_len, lookback_bars):
    """Son lookback_bars icinde (varsa) EN SON kesismeyi dondurur."""
    if not klines:
        return None

    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    times = [int(k[0]) for k in klines]  # open time ms

    high_lrc = rolling_linreg_end_values(highs, lrc_len)
    low_lrc = rolling_linreg_end_values(lows, lrc_len)

    n = len(highs)
    start = max(lrc_len, n - lookback_bars)

    last = None
    for i in range(start, n):
        if i - 1 < 0:
            continue

        if any(math.isnan(x) for x in (high_lrc[i], low_lrc[i], high_lrc[i - 1], low_lrc[i - 1])):
            continue

        prev_high, prev_low = float(high_lrc[i - 1]), float(low_lrc[i - 1])
        curr_high, curr_low = float(high_lrc[i]), float(low_lrc[i])

        ctype = None
        if prev_high < prev_low and curr_high > curr_low:
            ctype = "Crossover"
        elif prev_high > prev_low and curr_high < curr_low:
            ctype = "Crossunder"

        if ctype:
            last = {"idx": i, "ts": times[i], "type": ctype}

    return last


def iso_from_ms(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ===================== TELEGRAM =====================
def tg_send_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram ayarlari yok (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Mesaj gonderilmeyecek.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        r = session.post(url, json=payload, timeout=30)
        ok = r.status_code == 200
        if not ok:
            print("Telegram sendMessage hata:", r.status_code, r.text[:300])
        return ok
    except Exception as e:
        print("Telegram sendMessage exception:", str(e))
        return False


def chunk_lines(lines, max_chars=3500):
    """Telegram 4096 limitine takilmamak icin satirlari parcala."""
    chunks = []
    buf = ""
    for ln in lines:
        # +1 newline
        if len(buf) + len(ln) + 1 > max_chars:
            if buf:
                chunks.append(buf)
            buf = ln
        else:
            buf = ln if not buf else (buf + "\n" + ln)
    if buf:
        chunks.append(buf)
    return chunks


def build_tf_message(tf_key, items):
    """items: list of dict {SymbolLabel, CrossType, CrossTimeISO} sorted."""
    header = f"LRC Kesisme | {tf_key} | son {LOOKBACK_DAYS} gun"

    if not items:
        return [f"{header}\nKesisme yok."]

    # tablo
    # sembol en uzun: xxxxUSDT.P (10-15 arasi)
    lines = [header, "```", f"{'SYMBOL':<16} {'TYPE':<11} {'TIME(UTC)':<20}", "-" * 47]

    for it in items:
        sym = it["SymbolLabel"]
        ctype = it["CrossType"]
        ts = it["CrossTimeISO"]
        # kisa tip + emoji
        if ctype == "Crossover":
            cshort = "UP"
            emj = "🟢"
        else:
            cshort = "DOWN"
            emj = "🟠"

        # ISO'yu kisa yap
        tshort = ts.replace("+00:00", "Z")
        if len(tshort) > 19:
            tshort = tshort[:19] + "Z"

        lines.append(f"{sym:<16} {emj}{cshort:<8} {tshort:<20}")

    lines.append("```")

    # chunk: kod bloklari bozulmasin diye, her chunk'i ayri kod blogu olarak at
    # burada basit: eger cok kalabaliksa satirlari parcala
    if len("\n".join(lines)) <= 3500:
        return ["\n".join(lines)]

    # parcalama: header + kod blogu sabit kalsin
    body_lines = lines[3:-1]  # tablo baslik + ayirac + satirlar
    chunks = chunk_lines(body_lines, max_chars=2800)
    out_msgs = []
    for i, ch in enumerate(chunks, 1):
        part_header = header + (f" (part {i}/{len(chunks)})" if len(chunks) > 1 else "")
        out_msgs.append("\n".join([part_header, "```", f"{'SYMBOL':<16} {'TYPE':<11} {'TIME(UTC)':<20}", "-" * 47, ch, "```"]))
    return out_msgs


# ===================== SCAN =====================
def scan_market(symbols, is_futures, market_suffix):
    results = []
    total = len(symbols)

    for si, sym in enumerate(symbols, 1):
        print(f"[{si}/{total}] {sym} ({'PERP' if is_futures else 'SPOT'}) taraniyor...")

        for tf_key, meta in TIMEFRAMES.items():
            interval = meta["bin"]
            lookback_bars = LOOKBACK_DAYS * meta["bars_per_day"]
            limit_needed = LRC_LEN + lookback_bars + 10

            kl = fetch_klines(sym, interval, limit_needed, futures=is_futures)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if not kl:
                continue

            last = detect_last_cross_from_klines(kl, LRC_LEN, lookback_bars)
            if not last:
                continue

            results.append({
                "SymbolLabel": sym + market_suffix,
                "Timeframe": tf_key,
                "CrossType": last["type"],
                "CrossTimeISO": iso_from_ms(last["ts"]),
            })

    return results


def run():
    spot_syms = get_spot_usdt_symbols_filtered()
    fut_syms = get_futures_usdt_perp_symbols_filtered()

    print(f"Spot USDT sembol sayisi: {len(spot_syms)}")
    print(f"Futures PERP USDT sembol sayisi: {len(fut_syms)}")

    all_rows = []
    # Spot
    all_rows.extend(scan_market(spot_syms, is_futures=False, market_suffix=""))
    # Futures
    all_rows.extend(scan_market(fut_syms, is_futures=True, market_suffix=".P"))

    # Timeframe'e gore grupla ve alfabetik sirala
    by_tf = {k: [] for k in TIMEFRAMES.keys()}
    for r in all_rows:
        by_tf.setdefault(r["Timeframe"], []).append(r)

    for tf in by_tf:
        by_tf[tf] = sorted(by_tf[tf], key=lambda x: (x["SymbolLabel"].upper(), x["CrossType"]))

    # Telegram'a ayri ayri gonder
    for tf_key in TIMEFRAMES.keys():
        msgs = build_tf_message(tf_key, by_tf.get(tf_key, []))
        for m in msgs:
            tg_send_message(m)
            time.sleep(0.5)

    # Konsola kisa ozet
    total = len(all_rows)
    print("Toplam kesisme kaydi:", total)


def main():
    start = datetime.now(timezone.utc)
    print("LRC tarama basladi (UTC):", start.isoformat())

    run()

    end = datetime.now(timezone.utc)
    print("Bitti (UTC):", end.isoformat(), "Sure(s):", (end - start).total_seconds())


if __name__ == "__main__":
    main()
