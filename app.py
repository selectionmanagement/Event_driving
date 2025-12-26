import datetime as dt
import re
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


EVENTS_CSV_PATH = "events_crypto_liquidation_shocks.csv"
SIMPLE_EVENTS_CSV_PATH = "events_simple.csv"

COLUMN_LETTERS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]

EVENT_COLUMNS = [
    "event_id",
    "date_utc",
    "category",
    "scope",
    "total_liquidation_usd",
    "long_liquidation_usd",
    "short_liquidation_usd",
    "traders_liquidated",
    "funding_rate",
    "title",
    "source_url",
]

INTERVAL_MS = {
    "1d": 86_400_000,
    "4h": 14_400_000,
    "1h": 3_600_000,
}

CATEGORY_COLORS = {
    "liquidation_shock": "#FF6B6B",
    "regulation": "#4ECDC4",
    "macro": "#FFD166",
    "default": "#A0A7B4",
}

EVENT_MARKER_COLOR = "#FF5D8F"
EVENT_LABEL_COLOR = "#EAF2FF"
EVENT_LABEL_SIZE = 10


def read_events_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[EVENT_COLUMNS].copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["scope"] = df["scope"].fillna("market").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["source_url"] = df["source_url"].fillna("").astype(str)
    numeric_cols = [
        "total_liquidation_usd",
        "long_liquidation_usd",
        "short_liquidation_usd",
        "traders_liquidated",
        "funding_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "total_liquidation_usd" not in df.columns:
        df["total_liquidation_usd"] = pd.NA
    if "funding_rate" not in df.columns:
        df["funding_rate"] = pd.NA
    df["severity"] = 3
    df.loc[df["total_liquidation_usd"] >= 5e9, "severity"] = 5
    df.loc[
        (df["total_liquidation_usd"] >= 2e9) & (df["severity"] < 5), "severity"
    ] = 4
    df.loc[df["funding_rate"] >= 30, "severity"] = 5
    df.loc[(df["funding_rate"] >= 20) & (df["severity"] < 5), "severity"] = 4
    return df


def format_money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.0f}"


def format_rate(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}%"


def format_signed_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.2f}%"


def format_price(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return pd.isna(value)


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def read_simple_events_csv(path: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, header=None, dtype=str)
    except FileNotFoundError:
        return pd.DataFrame(columns=["event_id", "date_utc", "note", "event_price"])

    if raw.empty:
        return pd.DataFrame(columns=["event_id", "date_utc", "note", "event_price"])

    raw = raw.iloc[:, : len(COLUMN_LETTERS)]
    raw.columns = COLUMN_LETTERS[: raw.shape[1]]
    raw = ensure_columns(raw, COLUMN_LETTERS)

    date_series = raw["A"].astype(str).str.strip()
    date_series = date_series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    parsed_dayfirst = pd.to_datetime(date_series, dayfirst=True, errors="coerce", utc=True)
    parsed_monthfirst = pd.to_datetime(date_series, dayfirst=False, errors="coerce", utc=True)
    raw["date_utc"] = parsed_dayfirst.fillna(parsed_monthfirst)
    raw["note"] = raw["B"].fillna("").astype(str).str.strip()
    price_series = raw["C"].astype(str).str.strip()
    price_series = price_series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    price_series = price_series.str.replace(",", "", regex=False)
    price_series = price_series.str.replace("$", "", regex=False)
    raw["event_price"] = pd.to_numeric(price_series, errors="coerce")
    raw = raw[raw["date_utc"].notna()]

    raw["event_id"] = [f"simple_{idx + 1:03d}" for idx in range(len(raw))]
    return raw[["event_id", "date_utc", "note", "event_price"] + COLUMN_LETTERS]


def fetch_klines(
    symbol: str, interval: str, start_ms: int, end_ms: int, api_base: str
) -> pd.DataFrame:
    url = f"{api_base}/api/v3/klines"
    if "fapi" in api_base:
        url = f"{api_base}/fapi/v1/klines"
    limit = 1000
    interval_ms = INTERVAL_MS.get(interval, INTERVAL_MS["1d"])
    rows = []
    current = start_ms
    while current <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        next_start = last_open + interval_ms
        if next_start <= current:
            break
        current = next_start
        if len(data) < limit:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("open_time")
    df["close_change_pct"] = df["close"].pct_change() * 100
    df["date"] = df["open_time"].dt.date
    return df


def align_events(events_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    events_df = events_df.copy()
    if events_df.empty:
        events_df["aligned_time"] = pd.to_datetime([], utc=True)
        events_df["price_at_event"] = pd.Series(dtype="float64")
        events_df["close_change_pct"] = pd.Series(dtype="float64")
        return events_df
    if price_df.empty:
        events_df["aligned_time"] = events_df["date_utc"]
        events_df["price_at_event"] = pd.NA
        events_df["close_change_pct"] = pd.NA
        return events_df
    price_times = price_df["open_time"].to_numpy()
    event_times = events_df["date_utc"].to_numpy()
    idx = price_times.searchsorted(event_times, side="left")
    idx = idx.clip(0, len(price_times) - 1)
    prev_idx = (idx - 1).clip(0, len(price_times) - 1)
    choose_prev = (
        abs(event_times - price_times[prev_idx]) <= abs(event_times - price_times[idx])
    )
    final_idx = idx.copy()
    final_idx[choose_prev] = prev_idx[choose_prev]
    events_df["aligned_time"] = price_df["open_time"].iloc[final_idx].to_numpy()
    events_df["price_at_event"] = price_df["close"].iloc[final_idx].to_numpy()
    events_df["close_change_pct"] = price_df["close_change_pct"].iloc[final_idx].to_numpy()
    return events_df


def pick_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["default"])


def generate_event_id(prefix: str, df: pd.DataFrame) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    numbers = []
    for event_id in df["event_id"].fillna("").astype(str):
        match = pattern.match(event_id)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers or [0]) + 1
    return f"{prefix}_{next_num:03d}"


def build_chart(
    price_df: pd.DataFrame,
    events_df: pd.DataFrame,
    label_events: pd.DataFrame,
    show_labels: bool,
    simple_mode: bool,
) -> go.Figure:
    fig = go.Figure()
    if not price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=price_df["open_time"],
                y=price_df["close"],
                mode="lines",
                line={"color": "#E9C46A", "width": 2},
                name="Price",
            )
        )
    if events_df.empty:
        fig.update_layout(
            height=620,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
        )
        return fig

    label_field = "note" if simple_mode else "title"
    events_df = events_df.copy()

    if simple_mode:
        hover_texts = []
        extra_cols = COLUMN_LETTERS[3:]
        for _, row in events_df.iterrows():
            lines = []
            date_value = row.get("A")
            if not is_blank(date_value):
                lines.append(f"date: {date_value}")
            else:
                date_fallback = row.get("date_utc")
                if pd.notna(date_fallback):
                    lines.append(f"date: {date_fallback.strftime('%d/%m/%Y')}")
            note_value = row.get("B")
            if not is_blank(note_value):
                lines.append(f"note: {note_value}")
            price_value = row.get("C")
            if not is_blank(price_value):
                lines.append(f"price: {price_value}")
            for col in extra_cols:
                col_value = row.get(col)
                if not is_blank(col_value):
                    lines.append(f"{col}: {col_value}")
            hover_texts.append("<br>".join(lines) if lines else "n/a")

        events_df["hover_text"] = hover_texts
        fig.add_trace(
            go.Scatter(
                x=events_df["aligned_time"],
                y=events_df["price_at_event"],
                mode="markers",
                marker={
                    "size": 6 + events_df["severity"] * 1.5,
                    "color": EVENT_MARKER_COLOR,
                    "line": {"width": 0.5, "color": "#0B0F14"},
                    "opacity": 0.85,
                },
                name="Events",
                text=events_df["note"],
                customdata=events_df["hover_text"],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    else:
        hover_data = list(
            zip(
                events_df["title"].fillna(""),
                events_df["aligned_time"],
                events_df["total_liquidation_usd"].map(format_money),
                events_df["long_liquidation_usd"].map(format_money),
                events_df["short_liquidation_usd"].map(format_money),
                events_df["traders_liquidated"].map(format_number),
                events_df["funding_rate"].map(format_rate),
                events_df["close_change_pct"].map(format_signed_pct),
                events_df["source_url"].replace("", "n/a"),
            )
        )
        events_df["hover_title"] = [item[0] or "Untitled event" for item in hover_data]
        events_df["hover_date"] = [item[1] for item in hover_data]
        events_df["hover_total"] = [item[2] for item in hover_data]
        events_df["hover_long"] = [item[3] for item in hover_data]
        events_df["hover_short"] = [item[4] for item in hover_data]
        events_df["hover_traders"] = [item[5] for item in hover_data]
        events_df["hover_funding"] = [item[6] for item in hover_data]
        events_df["hover_close_change"] = [item[7] for item in hover_data]
        events_df["hover_source"] = [item[8] for item in hover_data]

        fig.add_trace(
            go.Scatter(
                x=events_df["aligned_time"],
                y=events_df["price_at_event"],
                mode="markers",
                marker={
                    "size": 6 + events_df["severity"] * 1.5,
                    "color": EVENT_MARKER_COLOR,
                    "line": {"width": 0.5, "color": "#0B0F14"},
                    "opacity": 0.85,
                },
                name="Events",
                text=events_df["title"],
                customdata=events_df[
                    [
                        "hover_total",
                        "hover_long",
                        "hover_short",
                        "hover_traders",
                        "hover_funding",
                        "hover_close_change",
                        "hover_source",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Total liq: %{customdata[0]}<br>"
                    "Long: %{customdata[1]}<br>"
                    "Short: %{customdata[2]}<br>"
                    "Traders: %{customdata[3]}<br>"
                    "Funding: %{customdata[4]}<br>"
                    "Close change: %{customdata[5]}<br>"
                    "Source: %{customdata[6]}<extra></extra>"
                ),
            )
        )

    if show_labels and not label_events.empty:
        labeled = label_events.copy()
        label_texts = []
        label_positions = []
        position_cycle = [
            "top center",
            "bottom center",
            "top left",
            "bottom right",
            "top right",
            "bottom left",
        ]
        for _, row in labeled.iterrows():
            title = row.get(label_field)
            if pd.isna(title):
                title = ""
            else:
                title = str(title)
            if simple_mode:
                label_texts.append(title)
            else:
                pct = format_signed_pct(row.get("close_change_pct"))
                if pct == "n/a":
                    label_texts.append(title)
                else:
                    label_texts.append(f"{title} ({pct})")
            label_positions.append(position_cycle[len(label_positions) % len(position_cycle)])
        fig.add_trace(
            go.Scatter(
                x=labeled["aligned_time"],
                y=labeled["price_at_event"],
                mode="markers+text",
                text=label_texts,
                textposition=label_positions,
                textfont={"color": EVENT_LABEL_COLOR, "size": EVENT_LABEL_SIZE},
                marker={
                    "size": 6 + labeled["severity"] * 1.5,
                    "color": EVENT_MARKER_COLOR,
                    "opacity": 0.9,
                },
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=620,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": -0.1},
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",
        font={"family": "Space Grotesk", "color": "#E6E1D8"},
    )
    fig.update_yaxes(
        type="log",
        title="Price (log scale)",
        gridcolor="#1F2937",
        linecolor="#2A3440",
        zerolinecolor="#2A3440",
    )
    fig.update_xaxes(title="Date", gridcolor="#1F2937", linecolor="#2A3440")
    return fig


def parse_optional_number(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_keyword_filter(value: str) -> List[str]:
    return [token.strip() for token in re.split(r"[,\n]+", value) if token.strip()]


def filter_events_by_keywords(
    events_df: pd.DataFrame, keywords: List[str], columns: List[str]
) -> pd.DataFrame:
    if not keywords or events_df.empty:
        return events_df
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    mask = pd.Series(False, index=events_df.index)
    for col in columns:
        if col in events_df.columns:
            mask |= (
                events_df[col]
                .fillna("")
                .astype(str)
                .str.contains(pattern, case=False, regex=True)
            )
    return events_df[mask].copy()


def main() -> None:
    st.set_page_config(page_title="Crypto Event Dashboard", layout="wide")

    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap");
        :root {
            --ink: #E6E1D8;
            --muted: #A4ACB8;
            --paper: #0B0F14;
            --paper-2: #10161D;
            --accent: #E9C46A;
            --accent-2: #4ECDC4;
            --border: #1F2937;
        }
        html, body, [class*="css"] { font-family: "Space Grotesk", sans-serif; color: var(--ink); }
        .stApp {
            background: radial-gradient(1200px circle at 10% 10%, #0B0F14 0%, #0F1720 50%, #0A1219 100%);
        }
        h1, h2, h3 {
            font-family: "Fraunces", serif;
            letter-spacing: 0.4px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F141B 0%, #0B0F14 100%);
            border-right: 1px solid var(--border);
        }
        .stButton > button {
            background-color: var(--accent);
            color: #11151A;
            border: none;
        }
        .stButton > button:hover {
            background-color: #D8B65A;
            color: #11151A;
        }
        div[data-testid="stMetric"] {
            background: #0F141B;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px;
        }
        .stDataFrame {
            background: #0F141B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Crypto Event-Driven Dashboard")
    st.caption("Event overlays are manual; price data is fetched on demand and never saved.")

    if "price_df" not in st.session_state:
        st.session_state["price_df"] = pd.DataFrame()
    if "manual_events" not in st.session_state:
        st.session_state["manual_events"] = pd.DataFrame(columns=EVENT_COLUMNS)

    with st.sidebar:
        st.header("Event Data")
        events_csv_path = st.text_input("Event CSV path", value=SIMPLE_EVENTS_CSV_PATH)
        st.caption("Columns: A-Z (A=date dd/mm/yyyy, B=note, C=price; others optional)")
        keyword_filter = st.text_input(
            "Event keyword filter",
            value="",
            help="Comma/newline separated keywords (case-insensitive).",
        )

    simple_mode = True
    events_df = read_simple_events_csv(events_csv_path)
    events_df = ensure_columns(events_df, ["event_id", "date_utc", "note", "event_price"])
    events_df["title"] = events_df["note"]
    events_df["category"] = "manual"
    events_df["scope"] = "market"
    events_df["source_url"] = ""
    events_df = ensure_columns(events_df, EVENT_COLUMNS + ["note", "event_price"] + COLUMN_LETTERS)
    keyword_terms = parse_keyword_filter(keyword_filter)
    if keyword_terms:
        events_df = filter_events_by_keywords(events_df, keyword_terms, ["note", "title"])

    min_event_date = events_df["date_utc"].min()
    default_start = min_event_date.date() if pd.notna(min_event_date) else dt.date(2019, 1, 1)
    default_end = dt.datetime.utcnow().date()

    with st.sidebar:
        st.header("Market Data")
        symbol = st.selectbox(
            "Symbol",
            options=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            index=0,
        )
        market_type = "Spot"
        interval = st.selectbox("Interval", options=list(INTERVAL_MS.keys()), index=0)
        start_date = st.date_input("Start date (UTC)", value=default_start)
        end_date = st.date_input("End date (UTC)", value=default_end)
        fetch = st.button("Fetch price data")

        st.header("Marker Labels")
        top_n_labels = st.slider("Label top-N events", min_value=0, max_value=30, value=12)
        show_labels = st.checkbox("Show labels", value=True)


    if fetch:
        if end_date < start_date:
            st.error("End date must be on or after start date.")
        else:
            start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
            end_dt = dt.datetime.combine(end_date, dt.time.max, tzinfo=dt.timezone.utc)
            api_base = "https://api.binance.com" if market_type == "Spot" else "https://fapi.binance.com"
            try:
                price_df = fetch_klines(
                    symbol=symbol.strip().upper(),
                    interval=interval,
                    start_ms=int(start_dt.timestamp() * 1000),
                    end_ms=int(end_dt.timestamp() * 1000),
                    api_base=api_base,
                )
                st.session_state["price_df"] = price_df
            except requests.RequestException as exc:
                st.error(f"Failed to fetch data: {exc}")

    events_df = compute_severity(events_df)

    price_df = st.session_state["price_df"]
    if price_df.empty:
        st.warning("No price data loaded yet. Use the sidebar to fetch.")
    elif "close_change_pct" not in price_df.columns:
        price_df = price_df.copy()
        price_df["close_change_pct"] = price_df["close"].pct_change() * 100
        st.session_state["price_df"] = price_df

    events_df = align_events(events_df, price_df)
    if simple_mode and "event_price" in events_df.columns:
        events_df["price_at_event"] = events_df["event_price"].where(
            events_df["event_price"].notna(),
            events_df["price_at_event"],
        )
    if events_df.empty:
        st.warning("No events loaded. Check CSV path and column A date format.")
    elif events_df["price_at_event"].isna().all():
        st.warning("No event markers: check column C price values or fetch market data.")
    events_df = events_df.sort_values("date_utc")
    events_df["close_change_display"] = events_df["close_change_pct"].map(format_signed_pct)

    if top_n_labels > 0 and not events_df.empty:
        if simple_mode:
            label_events = events_df.sort_values("date_utc", ascending=False).head(top_n_labels)
        else:
            label_events = events_df.sort_values(
                ["severity", "total_liquidation_usd"], ascending=False
            ).head(top_n_labels)
    else:
        label_events = events_df.iloc[0:0]

    fig = build_chart(
        price_df,
        events_df,
        label_events,
        show_labels and top_n_labels > 0,
        simple_mode,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Events")
    if simple_mode:
        display_df = events_df[
            [
                "event_id",
                "date_utc",
                "note",
                "event_price",
                "close_change_display",
            ]
        ].rename(columns={"close_change_display": "close_change_pct"})
    else:
        display_df = events_df[
            [
                "event_id",
                "date_utc",
                "category",
                "scope",
                "total_liquidation_usd",
                "funding_rate",
                "close_change_display",
                "title",
                "source_url",
            ]
        ].rename(columns={"close_change_display": "close_change_pct"})
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
