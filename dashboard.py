#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from hyperliquid.utils import constants

SCHEDULE_TZ = ZoneInfo("America/New_York")
HL_DEX = "xyz"

ROOT_DIR = Path(__file__).resolve().parent
NAV_HISTORY_PATH = ROOT_DIR / "nav_history.csv"
LATEST_PROMPT_PATH = ROOT_DIR / "latest_prompt.txt"
LATEST_RESPONSE_PATH = ROOT_DIR / "latest_response.txt"
DECISIONS_HISTORY_PATH = ROOT_DIR / "decisions_history.jsonl"
LOG_GLOB = "*.log"
DASHBOARD_STATE_PATH = ROOT_DIR / "dashboard_state.json"

TOKEN_PER_WORD = 1.3
PRICE_INPUT_PER_1M = 2.00
PRICE_OUTPUT_PER_1M = 6.00

WINDOWS: dict[str, Optional[timedelta]] = {
    "1h": timedelta(hours=1),
    "5h": timedelta(hours=5),
    "1j": timedelta(days=1),
    "7j": timedelta(days=7),
    "30j": timedelta(days=30),
    "90j": timedelta(days=90),
    "All time": None,
}

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
BLOCK_HEADER_RE = re.compile(r"(?m)^\s*[A-Z][A-Z0-9_-]*\s*:\s*\[?[A-Za-z0-9._/\-]+\]?\s*$")


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def parse_int(value: Any) -> Optional[int]:
    parsed = parse_float(value)
    return int(round(parsed)) if parsed is not None else None


def normalize_coin(raw_coin: str) -> str:
    coin = raw_coin.strip().strip("[]")
    if ":" in coin:
        coin = coin.split(":")[-1]
    return re.sub(r"[^A-Z0-9._-]", "", coin.upper().strip("[]"))


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
        match = re.match(r"^[A-Z][A-Z0-9_-]*\s*:\s*(\[?[A-Za-z0-9._/\-]+\]?)\s*$", stripped)
        if match:
            return normalize_coin(match.group(1))
        if stripped.upper() in FIELD_LABELS:
            break
    return None


def split_blocks(raw_text: str) -> list[str]:
    starts = [m.start() for m in BLOCK_HEADER_RE.finditer(raw_text)]
    if starts:
        blocks: list[str] = []
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(raw_text)
            block = raw_text[start:end].strip()
            if block:
                blocks.append(block)
        return blocks
    return [chunk.strip() for chunk in re.split(r"\n\s*\n+", raw_text) if chunk.strip()]


def parse_signals_from_text(raw_text: str) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in split_blocks(raw_text):
        coin_value = extract_field(block, "COIN")
        header_coin = extract_header_coin(block)
        coin = normalize_coin(coin_value) if coin_value else header_coin
        if not coin:
            continue
        norm_coin = normalize_coin(coin)
        if norm_coin in seen:
            continue
        seen.add(norm_coin)
        signals.append(
            {
                "coin": coin,
                "signal": (extract_field(block, "SIGNAL") or "hold").strip().lower(),
                "quantity": parse_float(extract_field(block, "QUANTITY")),
                "leverage": parse_int(extract_field(block, "LEVERAGE")),
                "stop_loss": parse_float(extract_field(block, "STOP LOSS")),
                "profit_target": parse_float(extract_field(block, "PROFIT TARGET")),
                "confidence": parse_float(extract_field(block, "CONFIDENCE")),
                "hold_reason": (extract_field(block, "HOLD_REASON") or "").strip() or None,
                "justification": (extract_field(block, "JUSTIFICATION") or "").strip() or None,
            }
        )
    return signals


def estimate_api_cost(prompt_text: str, response_text: str) -> dict[str, float]:
    in_words = len(re.findall(r"\S+", prompt_text))
    out_words = len(re.findall(r"\S+", response_text))
    in_tokens = in_words * TOKEN_PER_WORD
    out_tokens = out_words * TOKEN_PER_WORD
    in_cost = (in_tokens / 1_000_000) * PRICE_INPUT_PER_1M
    out_cost = (out_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M
    total_cost = in_cost + out_cost
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "input_cost_usd": in_cost,
        "output_cost_usd": out_cost,
        "total_cost_usd": total_cost,
        "total_cost_cent": total_cost * 100.0,
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_user_state(wallet_address: str) -> dict[str, Any]:
    url = f"{constants.TESTNET_API_URL}/info"
    payload = {"type": "clearinghouseState", "user": wallet_address, "dex": HL_DEX}
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Reponse Hyperliquid inattendue (clearinghouseState).")
    return data


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_mids() -> dict[str, float]:
    url = f"{constants.TESTNET_API_URL}/info"
    payload = {"type": "allMids", "dex": HL_DEX}
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        px = parse_float(value)
        if px is not None:
            parsed[str(key)] = px
    return parsed


def extract_nav_from_user_state(user_state: dict[str, Any]) -> Optional[float]:
    margin_summary = user_state.get("marginSummary", {})
    cross_summary = user_state.get("crossMarginSummary", {})
    nav = parse_float(margin_summary.get("accountValue"))
    if nav is None:
        nav = parse_float(cross_summary.get("accountValue"))
    return nav


def build_positions_df(user_state: dict[str, Any], mids: dict[str, float]) -> pd.DataFrame:
    asset_positions = user_state.get("assetPositions", []) if isinstance(user_state, dict) else []
    rows: list[dict[str, Any]] = []

    for item in asset_positions:
        position = item.get("position", {}) if isinstance(item, dict) else {}
        coin = str(position.get("coin", "")).strip()
        qty = parse_float(position.get("szi"))
        if not coin or qty is None or abs(qty) == 0:
            continue

        current_price = mids.get(coin)
        if current_price is None:
            norm_coin = normalize_coin(coin)
            for key, value in mids.items():
                if normalize_coin(key) == norm_coin:
                    current_price = value
                    break

        lev_value = None
        leverage_data = position.get("leverage", {})
        if isinstance(leverage_data, dict):
            lev_value = parse_int(leverage_data.get("value"))

        rows.append(
            {
                "coin": coin,
                "quantity": qty,
                "entry_price": parse_float(position.get("entryPx")),
                "current_price": current_price,
                "liquidation_price": parse_float(position.get("liquidationPx")),
                "unrealized_pnl": parse_float(position.get("unrealizedPnl")),
                "leverage": lev_value,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "coin",
                "quantity",
                "entry_price",
                "current_price",
                "liquidation_price",
                "unrealized_pnl",
                "leverage",
            ]
        )
    return pd.DataFrame(rows)


def load_nav_history() -> pd.DataFrame:
    if not NAV_HISTORY_PATH.exists():
        return pd.DataFrame(columns=["timestamp_et", "nav"])
    try:
        df = pd.read_csv(NAV_HISTORY_PATH)
    except Exception:
        return pd.DataFrame(columns=["timestamp_et", "nav"])

    if "timestamp_et" not in df.columns or "nav" not in df.columns:
        return pd.DataFrame(columns=["timestamp_et", "nav"])

    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"], errors="coerce", utc=True)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["timestamp_et", "nav"]).sort_values("timestamp_et")
    if df.empty:
        return pd.DataFrame(columns=["timestamp_et", "nav"])
    df["timestamp_et"] = df["timestamp_et"].dt.tz_convert(SCHEDULE_TZ)
    return df


def load_dashboard_state() -> dict[str, Any]:
    if not DASHBOARD_STATE_PATH.exists():
        return {}
    try:
        raw = json.loads(DASHBOARD_STATE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_dashboard_state(state: dict[str, Any]) -> None:
    DASHBOARD_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_reset_anchor() -> Optional[datetime]:
    state = load_dashboard_state()
    raw = state.get("stats_reset_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SCHEDULE_TZ)
    return dt.astimezone(SCHEDULE_TZ)


def set_reset_anchor_now() -> datetime:
    now = datetime.now(SCHEDULE_TZ)
    state = load_dashboard_state()
    state["stats_reset_at"] = now.isoformat()
    save_dashboard_state(state)
    return now


def apply_reset_anchor(df: pd.DataFrame, anchor: Optional[datetime]) -> pd.DataFrame:
    if anchor is None or df.empty:
        return df
    return df[df["timestamp_et"] >= anchor].copy()


def compute_window_stats(df: pd.DataFrame, now: datetime) -> dict[str, Optional[dict[str, float]]]:
    stats: dict[str, Optional[dict[str, float]]] = {}
    for label, delta in WINDOWS.items():
        window_df = df if delta is None else df[df["timestamp_et"] >= (now - delta)]
        if window_df.empty:
            stats[label] = None
            continue
        first_nav = float(window_df["nav"].iloc[0])
        last_nav = float(window_df["nav"].iloc[-1])
        delta_nav = last_nav - first_nav
        pct = None if first_nav == 0 else (delta_nav / first_nav) * 100.0
        stats[label] = {
            "first_nav": first_nav,
            "last_nav": last_nav,
            "delta_nav": delta_nav,
            "pct": pct if pct is not None else float("nan"),
            "points": float(len(window_df)),
        }
    return stats


def choose_ai_window(stats: dict[str, Optional[dict[str, float]]]) -> str:
    candidates: list[tuple[float, float, str]] = []
    for label in WINDOWS:
        row = stats.get(label)
        if not row:
            continue
        pct = row.get("pct")
        if pct is None or pd.isna(pct):
            continue
        points = row.get("points", 0.0)
        candidates.append((abs(float(pct)), float(points), label))
    if not candidates:
        return "All time"
    candidates.sort(reverse=True)
    return candidates[0][2]


def filter_nav_for_window(df: pd.DataFrame, label: str, now: datetime) -> pd.DataFrame:
    delta = WINDOWS.get(label)
    if delta is None:
        return df
    return df[df["timestamp_et"] >= (now - delta)].copy()


def load_decisions_history(limit: int = 500) -> list[dict[str, Any]]:
    if not DECISIONS_HISTORY_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with DECISIONS_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    if len(records) <= limit:
        return records
    return records[-limit:]


def fallback_latest_cycle_from_response() -> Optional[dict[str, Any]]:
    raw = read_text(LATEST_RESPONSE_PATH).strip()
    if not raw:
        return None
    signals = parse_signals_from_text(raw)
    if not signals:
        return None
    summary: dict[str, int] = {}
    for signal in signals:
        action = str(signal.get("signal", "unknown")).strip().lower() or "unknown"
        summary[action] = summary.get(action, 0) + 1
    timestamp = datetime.fromtimestamp(LATEST_RESPONSE_PATH.stat().st_mtime, tz=SCHEDULE_TZ)
    return {
        "timestamp_et": timestamp.isoformat(),
        "wallet": os.getenv("HL_WALLET_ADDRESS", "").strip(),
        "dex": HL_DEX,
        "signal_count": len(signals),
        "summary": summary,
        "signals": signals,
    }


def get_latest_cycle() -> Optional[dict[str, Any]]:
    history = load_decisions_history()
    if history:
        return history[-1]
    return fallback_latest_cycle_from_response()


def list_log_files() -> list[Path]:
    return sorted(ROOT_DIR.glob(LOG_GLOB))


def load_logs_df(files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    level_pattern = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR)\s*\|")
    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            match = level_pattern.search(line)
            level = match.group(1) if match else "UNKNOWN"
            rows.append(
                {
                    "file": file_path.name,
                    "path": str(file_path),
                    "line_no": idx,
                    "level": level,
                    "text": line,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["file", "path", "line_no", "level", "text"])
    return pd.DataFrame(rows)


def apply_log_filters(df: pd.DataFrame, selected_level: str, search_query: str) -> pd.DataFrame:
    filtered = df
    if selected_level != "Tous":
        filtered = filtered[filtered["level"] == selected_level]
    if search_query.strip():
        needle = search_query.strip().lower()
        filtered = filtered[
            filtered["text"].str.lower().str.contains(needle, regex=False)
            | filtered["file"].str.lower().str.contains(needle, regex=False)
        ]
    return filtered.copy()


def export_logs_text(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(f"[{row['file']}:{row['line_no']}] {row['text']}")
    return "\n".join(lines)


def delete_filtered_lines(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    grouped: dict[str, set[int]] = {}
    for _, row in df.iterrows():
        path = str(row["path"])
        line_no = int(row["line_no"])
        grouped.setdefault(path, set()).add(line_no)

    deleted_lines = 0
    touched_files = 0
    for path_str, line_numbers in grouped.items():
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            kept: list[str] = []
            for idx, line in enumerate(lines, start=1):
                if idx in line_numbers:
                    deleted_lines += 1
                    continue
                kept.append(line)
            path.write_text("".join(kept), encoding="utf-8")
            touched_files += 1
        except Exception:
            continue
    return deleted_lines, touched_files


def delete_selected_files(files: list[Path]) -> int:
    removed = 0
    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()
                removed += 1
        except Exception:
            continue
    return removed


def main() -> None:
    st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
    st.title("Trading Bot Dashboard")

    st.sidebar.header("Controle")
    selected_window = st.sidebar.selectbox(
        "Fenetre graphe",
        ["AI choice", "1h", "5h", "1j", "7j", "30j", "90j", "All time"],
        index=0,
    )

    reset_anchor = get_reset_anchor()
    if reset_anchor:
        st.sidebar.caption(f"Stats reset le: {reset_anchor.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    else:
        st.sidebar.caption("Stats reset: jamais")

    if st.sidebar.button("Reset stats baseline"):
        now = set_reset_anchor_now()
        st.cache_data.clear()
        st.sidebar.success(f"Baseline reset: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        st.rerun()

    tab_home, tab_logs = st.tabs(["Home", "Logs"])

    with tab_home:
        wallet_address = os.getenv("HL_WALLET_ADDRESS", "").strip()
        user_state: dict[str, Any] = {}
        mids: dict[str, float] = {}
        nav_live: Optional[float] = None
        positions_df = pd.DataFrame()

        if not wallet_address:
            st.warning("HL_WALLET_ADDRESS non defini. Les donnees live Hyperliquid ne peuvent pas etre chargees.")
        else:
            try:
                user_state = fetch_user_state(wallet_address)
                mids = fetch_all_mids()
                nav_live = extract_nav_from_user_state(user_state)
                positions_df = build_positions_df(user_state, mids)
            except Exception as exc:
                st.error(f"Erreur lors du chargement Hyperliquid: {exc}")

        nav_history_df = load_nav_history()
        scoped_nav_df = apply_reset_anchor(nav_history_df, get_reset_anchor())
        now = datetime.now(SCHEDULE_TZ)
        window_stats = compute_window_stats(scoped_nav_df, now)
        ai_window = choose_ai_window(window_stats)
        active_window = ai_window if selected_window == "AI choice" else selected_window
        graph_df = filter_nav_for_window(scoped_nav_df, active_window, now)

        prompt_text = read_text(LATEST_PROMPT_PATH)
        response_text = read_text(LATEST_RESPONSE_PATH)
        api_cost = estimate_api_cost(prompt_text, response_text)

        nav_display = nav_live
        if nav_display is None and not scoped_nav_df.empty:
            nav_display = float(scoped_nav_df["nav"].iloc[-1])

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "NAV actuel",
            f"{nav_display:.2f} USDC" if nav_display is not None else "N/A",
        )
        metric_cols[1].metric("Positions ouvertes", str(len(positions_df.index)))
        metric_cols[2].metric("Cout API dernier cycle", f"{api_cost['total_cost_cent']:.4f} c")

        perf_cols = st.columns(len(WINDOWS))
        for idx, label in enumerate(WINDOWS.keys()):
            row = window_stats.get(label)
            if not row or pd.isna(row["pct"]):
                perf_cols[idx].metric(label, "N/A", "N/A")
                continue
            perf_cols[idx].metric(label, f"{float(row['pct']):+.2f}%", f"{float(row['delta_nav']):+.2f} USDC")

        main_col, right_col = st.columns([2.2, 1.2], gap="large")

        with main_col:
            st.subheader("Evolution du wallet")
            if selected_window == "AI choice":
                st.caption(f"Fenetre AI choice retenue: {ai_window}")
            else:
                st.caption(f"Fenetre active: {active_window}")

            if graph_df.empty:
                st.info("Pas assez de donnees NAV pour tracer un graphe sur cette fenetre.")
            else:
                chart_df = graph_df.copy()
                fig = px.line(chart_df, x="timestamp_et", y="nav", markers=True)
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Positions en cours (live Hyperliquid Testnet)")
            if positions_df.empty:
                st.info("Aucune position ouverte.")
            else:
                st.dataframe(positions_df, use_container_width=True, height=280)

        with right_col:
            latest_cycle = get_latest_cycle()
            st.subheader("Decisions IA")
            if latest_cycle:
                timestamp_et = latest_cycle.get("timestamp_et", "")
                if isinstance(timestamp_et, str) and timestamp_et:
                    st.caption(f"Dernier cycle: {timestamp_et}")
                signals = latest_cycle.get("signals", [])
                if isinstance(signals, list) and signals:
                    orders_df = pd.DataFrame(signals)
                    orders_cols = [
                        col
                        for col in [
                            "coin",
                            "signal",
                            "quantity",
                            "leverage",
                            "stop_loss",
                            "profit_target",
                            "confidence",
                        ]
                        if col in orders_df.columns
                    ]
                    if orders_cols:
                        st.markdown("**Ordres (y compris hold)**")
                        st.dataframe(orders_df[orders_cols], use_container_width=True, height=220)

                    reflection_cols = [col for col in ["coin", "hold_reason", "justification"] if col in orders_df.columns]
                    if reflection_cols:
                        st.markdown("**Reflexion IA par actif**")
                        st.dataframe(orders_df[reflection_cols], use_container_width=True, height=260)
                else:
                    st.info("Aucun signal trouve pour le dernier cycle.")
            else:
                st.info("Aucun historique de decisions disponible.")

            with st.expander("Voir le User Prompt envoye"):
                st.text_area(
                    "latest_prompt.txt",
                    value=prompt_text or "(vide)",
                    height=180,
                    disabled=True,
                )
            with st.expander("Voir la reponse brute de l'IA"):
                st.text_area(
                    "latest_response.txt",
                    value=response_text or "(vide)",
                    height=180,
                    disabled=True,
                )

    with tab_logs:
        st.subheader("Logs")
        log_files = list_log_files()
        if not log_files:
            st.info("Aucun fichier .log trouve.")
            return

        name_to_path = {path.name: path for path in log_files}
        selected_names = st.multiselect(
            "Fichiers logs",
            options=list(name_to_path.keys()),
            default=list(name_to_path.keys()),
        )
        selected_paths = [name_to_path[name] for name in selected_names]

        level_filter = st.selectbox("Niveau", ["Tous", "DEBUG", "INFO", "WARNING", "ERROR"], index=0)
        search_query = st.text_input("Recherche dans les logs")

        logs_df = load_logs_df(selected_paths)
        filtered_logs_df = apply_log_filters(logs_df, level_filter, search_query)

        st.caption(f"Lignes affichees: {len(filtered_logs_df.index)}")
        if filtered_logs_df.empty:
            st.info("Aucune ligne de log correspondant aux filtres.")
        else:
            st.dataframe(
                filtered_logs_df[["file", "line_no", "level", "text"]],
                use_container_width=True,
                height=460,
            )

        export_blob = export_logs_text(filtered_logs_df)
        st.download_button(
            label="Exporter les logs filtres",
            data=export_blob.encode("utf-8"),
            file_name=f"logs_export_{datetime.now(SCHEDULE_TZ).strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            disabled=not export_blob,
        )

        confirm_delete = st.checkbox("Confirmer suppression", value=False)
        delete_cols = st.columns(2)
        if delete_cols[0].button("Supprimer lignes filtrees", disabled=filtered_logs_df.empty):
            if not confirm_delete:
                st.warning("Coche 'Confirmer suppression' pour autoriser l'action.")
            else:
                deleted_lines, touched_files = delete_filtered_lines(filtered_logs_df)
                st.cache_data.clear()
                st.success(f"Lignes supprimees: {deleted_lines} sur {touched_files} fichier(s).")
                st.rerun()

        if delete_cols[1].button("Supprimer fichiers selectionnes", disabled=not selected_paths):
            if not confirm_delete:
                st.warning("Coche 'Confirmer suppression' pour autoriser l'action.")
            else:
                removed = delete_selected_files(selected_paths)
                st.cache_data.clear()
                st.success(f"Fichiers supprimes: {removed}.")
                st.rerun()


if __name__ == "__main__":
    main()
