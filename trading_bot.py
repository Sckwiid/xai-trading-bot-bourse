#!/usr/bin/env python3
"""
Bot de trading algorithmique xAI -> Hyperliquid Testnet.

Lancement rapide:
1) Créer un environnement virtuel et installer les dépendances:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2) Exporter les variables d'environnement:
   export XAI_API_KEY="..."
   export HL_WALLET_ADDRESS="0x..."
   export HL_PRIVATE_KEY="0x..."

3) Préparer les prompts:
   - System_prompt.txt
   - User_prompt.txt
   (par défaut dans le dossier courant, sinon configurer SYSTEM_PROMPT_PATH et USER_PROMPT_PATH)

4) Démarrer le bot:
   python trading_bot.py

Option utile de debug:
   python trading_bot.py --run-once
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests
import feedparser
from apscheduler.schedulers.blocking import BlockingScheduler
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = "grok-4.20-0309-reasoning"
SCHEDULE_TZ = ZoneInfo("America/New_York")
HL_DEX = "xyz"
XAI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
TRADING_LOG_PATH = Path("trading_bot.log")
NAV_HISTORY_PATH = Path("nav_history.csv")
LATEST_PROMPT_PATH = Path("latest_prompt.txt")
LATEST_RESPONSE_PATH = Path("latest_response.txt")
DECISIONS_HISTORY_PATH = Path("decisions_history.jsonl")
REQUIRE_FULL_SIGNAL_COVERAGE = True
MARKET_NEWS_TICKERS = [
    "NDX",
    "TSLA",
    "NVDA",
    "MSFT",
    "GOOGL",
    "PLTR",
    "AAPL",
    "META",
    "COIN",
    "HOOD",
    "INTC",
    "ORCL",
    "GOLD",
]
PROMPT_TO_EXCHANGE_COIN = {
    "NDX": "xyz:XYZ100",
    "GOOG": "xyz:GOOGL",
}
YAHOO_NEWS_SYMBOL_CANDIDATES = {
    "NDX": ["^NDX", "QQQ", "NDX"],
    "GOOGL": ["GOOGL", "GOOG"],
    "GOLD": ["GC=F", "GLD", "GOLD"],
}

# Les libellés supportés pour parser la réponse brute du LLM.
FIELD_LABELS = [
    "INVALIDATION CONDITION",
    "RISK USD",
    "CONFIDENCE",
    "IS ADD",
    "JUSTIFICATION",
    "PROFIT TARGET",
    "COIN",
    "LEVERAGE",
    "SIGNAL",
    "HOLD_REASON",
    "QUANTITY",
    "STOP LOSS",
]
FIELD_LABELS_PATTERN = "|".join(re.escape(label) for label in FIELD_LABELS)

# Détecte une ligne d'en-tête de type "XYZ:MSFT", "ABC:TSLA", etc.
BLOCK_HEADER_RE = re.compile(
    r"(?m)^\s*[A-Z][A-Z0-9_-]*\s*:\s*\[?[A-Za-z0-9._/\-]+\]?\s*$"
)


@dataclass
class TradeSignal:
    coin: str
    signal: str
    quantity: Optional[float]
    leverage: Optional[int]
    stop_loss: Optional[float]
    profit_target: Optional[float]
    confidence: Optional[float]
    is_add: Optional[bool]
    hold_reason: Optional[str]
    justification: Optional[str]


def setup_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(TRADING_LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"Variable d'environnement manquante: {var_name}")
    return value


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    if not path.is_file():
        raise ValueError(f"Le chemin n'est pas un fichier: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Le fichier est vide: {path}")
    return content


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def get_expected_exchange_coins() -> list[str]:
    expected: list[str] = []
    for ticker in MARKET_NEWS_TICKERS:
        mapped = PROMPT_TO_EXCHANGE_COIN.get(ticker, f"xyz:{ticker}")
        expected.append(mapped)
    return expected


def inject_runtime_markers(base_user_prompt: str) -> str:
    now_et = datetime.now(SCHEDULE_TZ).strftime("%A, %B %d, %Y %I:%M %p %Z")
    watchlist_text = ", ".join(MARKET_NEWS_TICKERS)
    rendered = base_user_prompt
    rendered = rendered.replace("{{NOW_ET}}", now_et)
    rendered = rendered.replace("{{WATCHLIST}}", watchlist_text)
    return rendered


def get_current_nav(exchange: Exchange, wallet_address: str) -> Optional[float]:
    try:
        user_state = exchange.info.user_state(wallet_address, dex=HL_DEX)
    except Exception:
        logging.exception("Impossible de récupérer le NAV (user_state).")
        return None

    if not isinstance(user_state, dict):
        return None
    margin_summary = user_state.get("marginSummary", {})
    cross_summary = user_state.get("crossMarginSummary", {})
    nav_value = parse_float(str(margin_summary.get("accountValue", "")))
    if nav_value is None:
        nav_value = parse_float(str(cross_summary.get("accountValue", "")))
    return nav_value


def append_nav_history(nav_value: Optional[float], timestamp: datetime) -> None:
    if nav_value is None:
        logging.warning("NAV indisponible, nav_history.csv non mis à jour.")
        return

    file_exists = NAV_HISTORY_PATH.exists()
    with NAV_HISTORY_PATH.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp_et", "nav"])
        writer.writerow([timestamp.isoformat(), nav_value])


def append_decisions_history(
    signals: list[dict[str, Any]],
    timestamp: datetime,
    wallet_address: str,
) -> None:
    summary: dict[str, int] = {}
    for signal in signals:
        action = str(signal.get("signal", "unknown")).strip().lower() or "unknown"
        summary[action] = summary.get(action, 0) + 1

    record = {
        "timestamp_et": timestamp.isoformat(),
        "wallet": wallet_address,
        "dex": HL_DEX,
        "signal_count": len(signals),
        "summary": summary,
        "signals": signals,
    }
    with DECISIONS_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def normalize_signal_coin_for_coverage(raw_coin: str) -> str:
    mapped = PROMPT_TO_EXCHANGE_COIN.get(raw_coin.upper(), raw_coin)
    return normalize_coin(mapped)


def _compact_news_title(title: str) -> str:
    compact = re.sub(r"\s+", " ", title.strip())
    # Évite de casser le format demandé avec des doubles quotes.
    compact = compact.replace('"', "'")
    return compact


def _fetch_rss_titles_for_symbol(symbol: str, max_titles: int = 3) -> list[str]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    parsed = feedparser.parse(url)
    entries = getattr(parsed, "entries", None)
    if not isinstance(entries, list) or not entries:
        return []

    titles: list[str] = []
    seen: set[str] = set()
    for item in entries:
        title = item.get("title") if isinstance(item, dict) else getattr(item, "title", None)
        if not isinstance(title, str):
            continue
        clean_title = _compact_news_title(title)
        if not clean_title or clean_title in seen:
            continue
        seen.add(clean_title)
        titles.append(clean_title)
        if len(titles) >= max_titles:
            break
    return titles


def fetch_market_news() -> tuple[str, list[str]]:
    lines: list[str] = []
    missing_tickers: list[str] = []

    for ticker in MARKET_NEWS_TICKERS:
        candidates = YAHOO_NEWS_SYMBOL_CANDIDATES.get(ticker, [ticker])
        titles: list[str] = []
        for symbol in candidates:
            try:
                titles = _fetch_rss_titles_for_symbol(symbol, max_titles=3)
            except Exception:
                logging.exception("Erreur RSS Yahoo Finance pour %s via %s", ticker, symbol)
                continue
            if titles:
                break

        if not titles:
            missing_tickers.append(ticker)
            continue

        joined_titles = " | ".join(f'"{title}"' for title in titles)
        lines.append(f"[{ticker}] : {joined_titles}")

    return "\n".join(lines).strip(), missing_tickers


def inject_market_news_into_user_prompt(base_user_prompt: str, market_news_block: str) -> str:
    marker = "{{MARKET_NEWS}}"
    replacement = market_news_block.strip()
    if marker in base_user_prompt:
        return base_user_prompt.replace(marker, replacement)

    if replacement:
        logging.warning("Balise {{MARKET_NEWS}} absente dans User_prompt.txt, insertion en tête.")
        return f"{replacement}\n\n{base_user_prompt.strip()}"
    return base_user_prompt


def extract_watchlist_from_prompt(prompt_text: str) -> set[str]:
    # Extrait les symboles de type "xyz:TSLA" et conserve aussi le ticker normalisé.
    raw_symbols = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9._/\-]+\b", prompt_text))
    watchlist: set[str] = set()
    for symbol in raw_symbols:
        watchlist.add(symbol.strip())
        watchlist.add(normalize_coin(symbol))
    return watchlist


def build_live_account_context(exchange: Exchange, wallet_address: str, base_user_prompt: str) -> str:
    user_state = exchange.info.user_state(wallet_address, dex=HL_DEX)
    positions_raw = user_state.get("assetPositions", []) if isinstance(user_state, dict) else []

    positions: list[dict[str, Any]] = []
    watchlist = extract_watchlist_from_prompt(base_user_prompt)
    # Ajoute les symboles monitorés pour garantir un snapshot de prix cohérent.
    for ticker in MARKET_NEWS_TICKERS:
        watchlist.add(ticker)
        mapped_coin = PROMPT_TO_EXCHANGE_COIN.get(ticker, f"xyz:{ticker}")
        watchlist.add(mapped_coin)
        watchlist.add(normalize_coin(mapped_coin))

    for item in positions_raw:
        position = item.get("position", {}) if isinstance(item, dict) else {}
        coin = str(position.get("coin", "")).strip()
        if not coin:
            continue

        entry_px = parse_float(str(position.get("entryPx", "")))
        liq_px = parse_float(str(position.get("liquidationPx", "")))
        qty = parse_float(str(position.get("szi", "")))
        unrealized = parse_float(str(position.get("unrealizedPnl", "")))

        lev_data = position.get("leverage", {})
        lev_value: Optional[int] = None
        if isinstance(lev_data, dict):
            lev_value = parse_int(str(lev_data.get("value", "")))

        positions.append(
            {
                "symbol": coin,
                "quantity": qty,
                "entry_price": entry_px,
                "current_price": None,  # Rempli après lecture des mids.
                "liquidation_price": liq_px,
                "unrealized_pnl": unrealized,
                "leverage": lev_value,
            }
        )
        watchlist.add(coin)
        watchlist.add(normalize_coin(coin))

    mids_raw = exchange.info.all_mids(dex=HL_DEX)
    mids: dict[str, Any] = mids_raw if isinstance(mids_raw, dict) else {}

    norm_watchlist = {normalize_coin(sym) for sym in watchlist if sym}
    filtered_prices: dict[str, float] = {}
    for key, value in mids.items():
        k = str(key).strip()
        if not k:
            continue
        if k not in watchlist and normalize_coin(k) not in norm_watchlist:
            continue
        parsed_value = parse_float(str(value))
        if parsed_value is not None:
            filtered_prices[k] = parsed_value

    # Enrichit les positions avec current_price si trouvé.
    for position in positions:
        symbol = str(position.get("symbol", ""))
        price = filtered_prices.get(symbol)
        if price is None:
            symbol_norm = normalize_coin(symbol)
            for k, v in filtered_prices.items():
                if normalize_coin(k) == symbol_norm:
                    price = v
                    break
        position["current_price"] = price

    nav_value: Optional[float] = None
    if isinstance(user_state, dict):
        margin_summary = user_state.get("marginSummary", {})
        cross_summary = user_state.get("crossMarginSummary", {})
        nav_value = parse_float(str(margin_summary.get("accountValue", "")))
        if nav_value is None:
            nav_value = parse_float(str(cross_summary.get("accountValue", "")))

    generated_at = datetime.now(SCHEDULE_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    context = (
        "LIVE ACCOUNT SNAPSHOT (HYPERLIQUID TESTNET)\n"
        f"Generated at: {generated_at}\n"
        f"Wallet: {wallet_address}\n\n"
        "CURRENT NAV\n"
        f"{nav_value if nav_value is not None else 'N/A'}\n\n"
        "CURRENT PRICES OF TRADEABLE COINS\n"
        f"{json.dumps(filtered_prices, ensure_ascii=False)}\n\n"
        "CURRENT OPEN POSITIONS\n"
        f"{json.dumps(positions, ensure_ascii=False)}"
    )
    return context


def inject_live_context_into_user_prompt(base_user_prompt: str, live_context: str) -> str:
    marker = "{{LIVE_ACCOUNT_STATE}}"
    if marker in base_user_prompt:
        return base_user_prompt.replace(marker, live_context)

    return (
        f"{base_user_prompt.strip()}\n\n"
        "IMPORTANT DATA OVERRIDE\n"
        "Use ONLY the live snapshot below for NAV, prices, and open positions. "
        "Ignore any conflicting or stale account data that may appear above.\n\n"
        f"{live_context}\n"
    )


def _compute_retry_sleep(attempt: int, retry_backoff_sec: float) -> float:
    return retry_backoff_sec * (2 ** (attempt - 1))


def call_xai(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> str:
    payload = {
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    safe_max_retries = max(1, int(max_retries))
    safe_backoff = max(0.1, float(retry_backoff_sec))

    data: dict[str, Any] | list[Any] | None = None
    for attempt in range(1, safe_max_retries + 1):
        try:
            response = requests.post(XAI_API_URL, headers=headers, json=payload, timeout=timeout_sec)
        except requests.RequestException as exc:
            if attempt < safe_max_retries:
                sleep_sec = _compute_retry_sleep(attempt, safe_backoff)
                logging.warning(
                    "Erreur réseau xAI (tentative %d/%d): %s. Retry dans %.1fs.",
                    attempt,
                    safe_max_retries,
                    exc,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(
                f"Échec d'appel xAI après {safe_max_retries} tentatives (erreur réseau)."
            ) from exc

        if response.status_code >= 400:
            body_excerpt = response.text[:500].replace("\n", " ").strip()
            if response.status_code in XAI_RETRYABLE_STATUS_CODES and attempt < safe_max_retries:
                sleep_sec = _compute_retry_sleep(attempt, safe_backoff)
                logging.warning(
                    "xAI HTTP %s (tentative %d/%d). Retry dans %.1fs. Body=%s",
                    response.status_code,
                    attempt,
                    safe_max_retries,
                    sleep_sec,
                    body_excerpt,
                )
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(f"xAI HTTP {response.status_code}: {body_excerpt}")

        try:
            data = response.json()
        except ValueError as exc:
            if attempt < safe_max_retries:
                sleep_sec = _compute_retry_sleep(attempt, safe_backoff)
                logging.warning(
                    "Réponse JSON invalide xAI (tentative %d/%d). Retry dans %.1fs.",
                    attempt,
                    safe_max_retries,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
                continue
            raise RuntimeError("Réponse JSON invalide de xAI après plusieurs tentatives.") from exc

        break

    if data is None:
        raise RuntimeError("Aucune réponse exploitable reçue de xAI.")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Réponse xAI inattendue (impossible de lire choices[0].message.content): {json.dumps(data)[:1200]}"
        ) from exc

    if isinstance(content, str):
        return content.strip()

    # Certains fournisseurs OpenAI-compatibles peuvent renvoyer une liste de blocs.
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    raise RuntimeError(f"Type de contenu xAI non supporté: {type(content).__name__}")


def normalize_coin(raw_coin: str) -> str:
    coin = raw_coin.strip()
    coin = coin.strip("[]")
    if ":" in coin:
        coin = coin.split(":")[-1]
    coin = coin.strip("[]")
    coin = coin.strip().upper()
    # Garde un format prudent pour le ticker Hyperliquid.
    coin = re.sub(r"[^A-Z0-9._-]", "", coin)
    return coin


def resolve_exchange_coin_symbol(exchange: Exchange, requested_coin: str) -> Optional[str]:
    symbol = requested_coin.strip()
    if not symbol:
        return None
    mapped_symbol = PROMPT_TO_EXCHANGE_COIN.get(symbol.upper())
    if mapped_symbol:
        symbol = mapped_symbol

    info = getattr(exchange, "info", None)
    name_to_coin = getattr(info, "name_to_coin", {}) if info is not None else {}

    if isinstance(name_to_coin, dict) and symbol in name_to_coin:
        return symbol

    normalized = normalize_coin(symbol)
    matches: list[str] = []
    if isinstance(name_to_coin, dict):
        for candidate in name_to_coin.keys():
            if not isinstance(candidate, str):
                continue
            if normalize_coin(candidate) == normalized:
                matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        logging.warning("Coin ambigu pour '%s': correspondances=%s", symbol, matches)
        return matches[0]

    # Fallback: laisse le SDK tenter avec la valeur d'origine.
    return symbol


def parse_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def parse_int(value: Optional[str]) -> Optional[int]:
    num = parse_float(value)
    return int(round(num)) if num is not None else None


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def parse_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a", "na", "nan"}:
        return None
    return text


def normalize_signal(raw_signal: Optional[str], is_add: Optional[bool]) -> str:
    if raw_signal is None:
        return "add" if is_add else "hold"
    signal = raw_signal.strip().lower().replace(" ", "_")
    aliases = {
        "buy": "buy_to_enter",
        "buy_to_enter": "buy_to_enter",
        "add": "add",
        "sell": "sell",
        "close": "close",
        "hold": "hold",
        "do_nothing": "hold",
    }
    normalized = aliases.get(signal, signal)
    if is_add and normalized == "buy_to_enter":
        return "add"
    return normalized


def extract_field(block: str, label: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(label)}\s*$\s*(.*?)\s*(?=^\s*(?:{FIELD_LABELS_PATTERN})\s*$|\Z)"
    )
    match = pattern.search(block)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def extract_header_coin(block: str) -> Optional[str]:
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(
            r"^[A-Z][A-Z0-9_-]*\s*:\s*(\[?[A-Za-z0-9._/\-]+\]?)\s*$",
            stripped,
        )
        if match:
            return normalize_coin(match.group(1))
        # Dès qu'on entre dans une zone de labels, inutile de continuer.
        if stripped.upper() in FIELD_LABELS:
            break
    return None


def split_blocks(raw_text: str) -> list[str]:
    starts = [m.start() for m in BLOCK_HEADER_RE.finditer(raw_text)]
    if starts:
        blocks: list[str] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(raw_text)
            block = raw_text[start:end].strip()
            if block:
                blocks.append(block)
        return blocks

    # Fallback: découpe par lignes vides pour des formats sans en-têtes "XYZ:TICKER".
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", raw_text) if chunk.strip()]
    return chunks if chunks else [raw_text.strip()]


def parse_signals(raw_text: str) -> list[dict[str, Any]]:
    parsed_signals: list[dict[str, Any]] = []
    blocks = split_blocks(raw_text)
    seen_coins: set[str] = set()

    for block in blocks:
        coin_raw = extract_field(block, "COIN")
        header_coin = extract_header_coin(block)
        coin = normalize_coin(coin_raw) if coin_raw else header_coin
        if not coin:
            logging.warning("Bloc ignoré (coin introuvable): %s", block[:300])
            continue

        signal_raw = extract_field(block, "SIGNAL")
        quantity = parse_float(extract_field(block, "QUANTITY"))
        leverage = parse_int(extract_field(block, "LEVERAGE"))
        stop_loss = parse_float(extract_field(block, "STOP LOSS"))
        profit_target = parse_float(extract_field(block, "PROFIT TARGET"))
        confidence = parse_float(extract_field(block, "CONFIDENCE"))
        is_add = parse_bool(extract_field(block, "IS ADD"))
        hold_reason = parse_text(extract_field(block, "HOLD_REASON"))
        justification = parse_text(extract_field(block, "JUSTIFICATION"))
        signal = normalize_signal(signal_raw, is_add)
        if signal == "hold" and not hold_reason:
            hold_reason = justification or "unspecified_by_model"

        signal_obj = TradeSignal(
            coin=coin,
            signal=signal,
            quantity=quantity,
            leverage=leverage,
            stop_loss=stop_loss,
            profit_target=profit_target,
            confidence=confidence,
            is_add=is_add,
            hold_reason=hold_reason,
            justification=justification,
        )
        normalized_coin = normalize_coin(signal_obj.coin)
        if normalized_coin in seen_coins:
            logging.warning("Bloc dupliqué ignoré pour le coin %s.", normalized_coin)
            continue
        seen_coins.add(normalized_coin)
        parsed_signals.append(
            {
                "coin": signal_obj.coin,
                "signal": signal_obj.signal,
                "quantity": signal_obj.quantity,
                "leverage": signal_obj.leverage,
                "stop_loss": signal_obj.stop_loss,
                "profit_target": signal_obj.profit_target,
                "confidence": signal_obj.confidence,
                "is_add": signal_obj.is_add,
                "hold_reason": signal_obj.hold_reason,
                "justification": signal_obj.justification,
            }
        )

    return parsed_signals


def validate_signal_coverage(signals: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    expected_norm = {normalize_coin(coin) for coin in get_expected_exchange_coins()}
    got_norm = {
        normalize_signal_coin_for_coverage(str(signal.get("coin", "")))
        for signal in signals
        if signal.get("coin")
    }
    missing = expected_norm - got_norm
    return expected_norm, missing


def init_hyperliquid_testnet(wallet_address: str, private_key: str) -> Exchange:
    account = Account.from_key(private_key)
    if wallet_address.lower() != account.address.lower():
        logging.warning(
            "HL_WALLET_ADDRESS (%s) diffère de l'adresse dérivée de HL_PRIVATE_KEY (%s). "
            "Le bot utilisera HL_WALLET_ADDRESS pour le compte Hyperliquid.",
            wallet_address,
            account.address,
        )

    # Workaround SDK: sur certains environnements testnet, spotMeta peut être incohérent
    # et provoquer un IndexError dans Info.__init__. Le bot trade ici en perp uniquement.
    empty_spot_meta: dict[str, list[Any]] = {"universe": [], "tokens": []}
    try:
        exchange = Exchange(
            account,
            constants.TESTNET_API_URL,
            account_address=wallet_address,
            spot_meta=empty_spot_meta,
        )
    except Exception as exc:
        raise RuntimeError(
            "Échec d'initialisation Hyperliquid Testnet. Vérifiez l'accès réseau vers "
            f"{constants.TESTNET_API_URL}, la validité du wallet, et la disponibilité du testnet."
        ) from exc
    return exchange


def log_order_response(action: str, coin: str, response: Any) -> None:
    if not isinstance(response, dict):
        logging.info("%s %s -> réponse: %s", action, coin, response)
        return

    if response.get("status") != "ok":
        logging.error("%s %s -> échec: %s", action, coin, response)
        return

    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
    if not statuses:
        logging.info("%s %s -> succès: %s", action, coin, response)
        return

    for status in statuses:
        filled = status.get("filled")
        if filled:
            logging.info(
                "%s %s -> rempli: oid=%s size=%s avg_px=%s",
                action,
                coin,
                filled.get("oid"),
                filled.get("totalSz"),
                filled.get("avgPx"),
            )
            continue
        if "error" in status:
            logging.error("%s %s -> erreur exchange: %s", action, coin, status["error"])
            continue
        logging.info("%s %s -> statut: %s", action, coin, status)


def get_open_position_norm_coins(exchange: Exchange) -> set[str]:
    account_address = getattr(exchange, "account_address", None)
    wallet = getattr(exchange, "wallet", None)
    if not account_address and wallet is not None:
        account_address = getattr(wallet, "address", None)

    if not account_address:
        return set()

    try:
        state = exchange.info.user_state(str(account_address), dex=HL_DEX)
    except Exception:
        logging.exception("Impossible de lire les positions ouvertes pour %s.", account_address)
        return set()

    asset_positions = state.get("assetPositions", []) if isinstance(state, dict) else []
    norm_coins: set[str] = set()
    for item in asset_positions:
        position = item.get("position", {}) if isinstance(item, dict) else {}
        coin = str(position.get("coin", "")).strip()
        qty = parse_float(str(position.get("szi", "")))
        if coin and qty is not None and abs(qty) > 0:
            norm_coins.add(normalize_coin(coin))
    return norm_coins


def execute_signals(
    exchange: Exchange,
    signals: list[dict[str, Any]],
    max_slippage: float,
) -> None:
    open_position_norm_coins = get_open_position_norm_coins(exchange)

    for signal in signals:
        action = str(signal.get("signal", "")).strip().lower()
        coin = str(signal.get("coin", "")).strip().upper()
        quantity = signal.get("quantity")
        leverage = signal.get("leverage")
        hold_reason = parse_text(str(signal.get("hold_reason", "")).strip())
        justification = parse_text(str(signal.get("justification", "")).strip())
        logging.info("Signal reçu: %s", signal)

        if not coin:
            logging.warning("Signal ignoré (coin vide): %s", signal)
            continue

        exchange_coin = resolve_exchange_coin_symbol(exchange, coin)
        if not exchange_coin:
            logging.warning("Signal ignoré (coin non résolu): %s", signal)
            continue
        if exchange_coin != coin:
            logging.info("Coin résolu pour exchange: %s -> %s", coin, exchange_coin)

        try:
            if action in {"buy_to_enter", "add"}:
                if not isinstance(quantity, (int, float)) or float(quantity) <= 0:
                    logging.error(
                        "Signal ignoré (%s %s): quantité absente ou invalide (%s).",
                        action,
                        coin,
                        quantity,
                    )
                    continue

                if isinstance(leverage, (int, float)) and int(leverage) > 0:
                    lev_resp = exchange.update_leverage(int(leverage), exchange_coin)
                    log_order_response("update_leverage", exchange_coin, lev_resp)
                else:
                    logging.warning("Levier absent pour %s, aucun ajustement de levier.", exchange_coin)

                order_resp = exchange.market_open(
                    exchange_coin, True, float(quantity), None, max_slippage
                )
                log_order_response("market_open_buy", exchange_coin, order_resp)

            elif action in {"sell", "close"}:
                order_resp = exchange.market_close(exchange_coin)
                log_order_response("market_close", exchange_coin, order_resp)

            elif action == "hold":
                reason = hold_reason or justification or "unspecified_by_model"
                if normalize_coin(exchange_coin) in open_position_norm_coins:
                    logging.info(
                        "Aucune action pour %s (signal hold sur position existante). Raison: %s",
                        exchange_coin,
                        reason,
                    )
                else:
                    logging.info(
                        "Aucune action pour %s (signal hold sans position ouverte). Raison: %s",
                        exchange_coin,
                        reason,
                    )

            else:
                logging.warning(
                    "Signal inconnu '%s' pour %s, aucune action exécutée.", action, exchange_coin
                )

        except Exception as exc:
            logging.exception(
                "Erreur lors de l'exécution du signal %s pour %s: %s", action, exchange_coin, exc
            )


def run_cycle(
    system_prompt_path: Path,
    user_prompt_path: Path,
    xai_timeout_sec: int,
    xai_max_retries: int,
    xai_retry_backoff_sec: float,
    max_slippage: float,
) -> None:
    logging.info("Démarrage d'un cycle de trading.")

    xai_api_key = require_env("XAI_API_KEY")
    wallet_address = require_env("HL_WALLET_ADDRESS")
    private_key = require_env("HL_PRIVATE_KEY")
    exchange = init_hyperliquid_testnet(wallet_address=wallet_address, private_key=private_key)
    try:
        market_news_block, missing_news_tickers = fetch_market_news()
        if market_news_block:
            logging.info("News marché récupérées pour le prompt (%d lignes).", market_news_block.count("\n") + 1)
        else:
            logging.warning("Aucune news marché récupérée via RSS Yahoo Finance.")
        if missing_news_tickers:
            logging.warning(
                "News manquantes pour %d/%d tickers monitorés: %s",
                len(missing_news_tickers),
                len(MARKET_NEWS_TICKERS),
                ", ".join(missing_news_tickers),
            )
        else:
            logging.info("Couverture news complète: %d/%d tickers.", len(MARKET_NEWS_TICKERS), len(MARKET_NEWS_TICKERS))

        system_prompt = read_text_file(system_prompt_path)
        base_user_prompt = read_text_file(user_prompt_path)
        base_user_prompt = inject_runtime_markers(base_user_prompt)
        base_user_prompt = inject_market_news_into_user_prompt(base_user_prompt, market_news_block)

        user_prompt = base_user_prompt
        try:
            live_context = build_live_account_context(exchange, wallet_address, base_user_prompt)
            user_prompt = inject_live_context_into_user_prompt(base_user_prompt, live_context)
            logging.info("Snapshot live Hyperliquid injecté dans le prompt utilisateur.")
        except Exception:
            logging.exception(
                "Impossible de récupérer le snapshot live Hyperliquid. "
                "Fallback sur User_prompt.txt brut."
            )

        write_text_file(LATEST_PROMPT_PATH, user_prompt)

        xai_text = call_xai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=xai_api_key,
            timeout_sec=xai_timeout_sec,
            max_retries=xai_max_retries,
            retry_backoff_sec=xai_retry_backoff_sec,
        )
        write_text_file(LATEST_RESPONSE_PATH, xai_text)
        logging.info("Réponse xAI reçue (%d caractères).", len(xai_text))
        logging.debug("Réponse xAI brute:\n%s", xai_text)

        signals = parse_signals(xai_text)
        if not signals:
            raise RuntimeError("Aucun signal exploitable trouvé dans la réponse xAI.")

        expected_norm, missing_norm = validate_signal_coverage(signals)
        if REQUIRE_FULL_SIGNAL_COVERAGE and missing_norm:
            raise RuntimeError(
                "Réponse xAI incomplète: "
                f"{len(missing_norm)}/{len(expected_norm)} tickers manquants ({', '.join(sorted(missing_norm))})."
            )

        append_decisions_history(signals, datetime.now(SCHEDULE_TZ), wallet_address)
        execute_signals(exchange=exchange, signals=signals, max_slippage=max_slippage)
        logging.info("Cycle de trading terminé.")
    finally:
        nav_now = get_current_nav(exchange, wallet_address)
        append_nav_history(nav_now, datetime.now(SCHEDULE_TZ))


def start_scheduler(job_callable: Any) -> None:
    scheduler = BlockingScheduler(timezone=SCHEDULE_TZ)
    job = scheduler.add_job(
        job_callable,
        trigger="cron",
        day_of_week="mon-fri",
        hour=15,
        minute=45,
        id="xai_hyperliquid_daily_1545_et",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    next_run_time: Optional[datetime] = None

    # Compatibilité APScheduler: un job "tentative" peut ne pas exposer next_run_time.
    try:
        next_run_time = getattr(job, "next_run_time")
    except AttributeError:
        next_run_time = None

    if not next_run_time:
        trigger = getattr(job, "trigger", None)
        get_next_fire_time = getattr(trigger, "get_next_fire_time", None)
        if callable(get_next_fire_time):
            try:
                next_run_time = get_next_fire_time(None, datetime.now(SCHEDULE_TZ))
            except Exception:
                next_run_time = None

    if next_run_time:
        logging.info(
            "Scheduler actif. Prochaine exécution: %s (timezone America/New_York).",
            next_run_time,
        )
    else:
        logging.info("Scheduler actif. Timezone: America/New_York.")
    scheduler.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot de trading xAI -> Hyperliquid Testnet")
    parser.add_argument(
        "--system-prompt",
        default=os.getenv("SYSTEM_PROMPT_PATH", "System_prompt.txt"),
        help="Chemin vers System_prompt.txt (default: SYSTEM_PROMPT_PATH ou ./System_prompt.txt)",
    )
    parser.add_argument(
        "--user-prompt",
        default=os.getenv("USER_PROMPT_PATH", "User_prompt.txt"),
        help="Chemin vers User_prompt.txt (default: USER_PROMPT_PATH ou ./User_prompt.txt)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Exécute un cycle immédiatement puis s'arrête (sans scheduler).",
    )
    parser.add_argument(
        "--xai-timeout",
        type=int,
        default=120,
        help="Timeout en secondes pour l'appel xAI (default: 120).",
    )
    parser.add_argument(
        "--xai-max-retries",
        type=int,
        default=5,
        help="Nombre max de tentatives xAI (default: 5).",
    )
    parser.add_argument(
        "--xai-retry-backoff",
        type=float,
        default=2.0,
        help="Backoff de base en secondes pour les retries xAI (default: 2.0).",
    )
    parser.add_argument(
        "--max-slippage",
        type=float,
        default=0.01,
        help="Slippage max pour market_open Hyperliquid (default: 0.01).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Niveau de logs (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    system_prompt_path = Path(args.system_prompt).expanduser().resolve()
    user_prompt_path = Path(args.user_prompt).expanduser().resolve()

    def job() -> None:
        run_cycle(
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            xai_timeout_sec=args.xai_timeout,
            xai_max_retries=args.xai_max_retries,
            xai_retry_backoff_sec=args.xai_retry_backoff,
            max_slippage=args.max_slippage,
        )

    if args.run_once:
        job()
        return 0

    logging.info(
        "Bot en mode scheduler: exécution du lundi au vendredi à 15:45 America/New_York."
    )
    start_scheduler(job)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.info("Arrêt manuel du bot.")
        raise SystemExit(0)
