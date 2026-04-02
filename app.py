import re
from itertools import takewhile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────────
_BASE = "https://docs.google.com/spreadsheets/d/1JlGDTULosvO6UHIpnwa6ZDjP2HLmwTjZ6dRJ5uS3DQo/export?format=csv&gid="
URLS = {
    "Matches":          _BASE + "0",
    "Match Level Elo":  _BASE + "314728595",
    "Standings":        _BASE + "511969979",
    "Single Opponents": _BASE + "1079417723",
}

st.set_page_config(page_title="Darts League", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  [data-testid="metric-container"] { background:#1e1e2e; border-radius:8px; padding:12px; }
  .stTabs [data-baseweb="tab"] { font-size:15px; font-weight:600; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    matches = pd.read_csv(URLS["Matches"])
    matches = matches[matches["Game"].notna()].copy()
    for col in ["Legs Team 1", "Legs Team 2", "Game"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches = matches[matches["Game"].notna()].copy()
    for col in ["Team 1 Before", "Team 2 Before", "Team 1 After", "Team 2 After",
                "T1 Change", "T2 Change", "T1 180", "T2 180"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    elo_hist = pd.read_csv(URLS["Match Level Elo"], header=1)
    player_names = [c for c in elo_hist.columns if c != "Match" and str(c).strip() not in ("", "nan")]
    elo_hist["Match"] = pd.to_numeric(elo_hist["Match"], errors="coerce")
    for p in player_names:
        elo_hist[p] = pd.to_numeric(
            elo_hist[p].astype(str).replace(["9999", "nan", ""], np.nan), errors="coerce"
        )
    elo_hist = elo_hist.dropna(subset=["Match"])

    standings = pd.read_csv(URLS["Standings"], header=1)
    standings = standings[standings["NAMES"].notna() & (standings["NAMES"].str.strip() != "")].copy()
    standings["Elo"] = pd.to_numeric(standings["Elo"], errors="coerce")
    standings["Rank"] = pd.to_numeric(standings["Rank"], errors="coerce")

    h2h = pd.read_csv(URLS["Single Opponents"])
    h2h = h2h[h2h.iloc[:, 0].notna() & (h2h.iloc[:, 0].astype(str).str.strip() != "")].copy()

    return matches, elo_hist, standings, h2h, player_names


# ── Helpers ──────────────────────────────────────────────────────────────────

def elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def parse_wl(s) -> tuple[int, int]:
    m = re.match(r"(\d+)-(\d+)", str(s or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def get_current_elos(elo_hist: pd.DataFrame, players: list[str]) -> dict[str, float]:
    result = {}
    for p in players:
        if p not in elo_hist.columns:
            continue
        series = elo_hist[p].dropna()
        if not series.empty:
            result[p] = float(series.iloc[-1])
    return result


def get_elo_changes_l2(
    matches: pd.DataFrame, elo_hist: pd.DataFrame, players: list[str], n_days: int = 2
) -> dict[str, float]:
    dates = matches["Date"].dropna()
    dates = dates[dates.astype(str).str.strip() != ""]
    unique_dates = list(dict.fromkeys(dates.tolist()))
    if not unique_dates:
        return {p: 0.0 for p in players}
    cutoff_date = unique_dates[-n_days] if len(unique_dates) > n_days else unique_dates[0]
    cutoff_games = matches[matches["Date"] == cutoff_date]["Game"].dropna()
    if cutoff_games.empty:
        return {p: 0.0 for p in players}
    cutoff_game = int(cutoff_games.min())
    result = {}
    for p in players:
        if p not in elo_hist.columns:
            result[p] = 0.0
            continue
        current = elo_hist[p].dropna()
        before  = elo_hist[elo_hist["Match"] < cutoff_game][p].dropna()
        if current.empty:
            result[p] = 0.0
        elif before.empty:
            result[p] = float(current.iloc[-1] - 800)
        else:
            result[p] = float(current.iloc[-1] - before.iloc[-1])
    return result


def player_matches(df: pd.DataFrame, player: str) -> pd.DataFrame:
    mask = (
        (df["Player A"] == player)
        | (df["Player B"] == player)
        | (df["Placer C"] == player)
        | (df["Player D"] == player)
    )
    return df[mask]


def result_for(row, player: str) -> str:
    return "W" if player in str(row["Winner"]) else "L"


def color_result(val):
    return "color:#2ecc71;font-weight:bold" if val == "W" else "color:#e74c3c;font-weight:bold"


def get_results_list(df: pd.DataFrame, player: str) -> list[str]:
    pm = player_matches(df, player).sort_values("Game")
    return [result_for(r, player) for _, r in pm.iterrows()]


def form_emoji(results: list[str], n: int = 5) -> str:
    tail = results[-n:]
    return " ".join("🟢" if r == "W" else "🔴" for r in tail)


def get_streaks(results: list[str]) -> tuple[str, int]:
    """Returns (current_streak e.g. '3W', longest_win_streak)."""
    if not results:
        return "–", 0
    val = results[-1]
    cur_len = sum(1 for _ in takewhile(lambda x: x == val, reversed(results)))
    current = f"{cur_len}{val}"
    max_ws = cur_ws = 0
    for r in results:
        cur_ws = cur_ws + 1 if r == "W" else 0
        max_ws = max(max_ws, cur_ws)
    return current, max_ws


def style_streak(val):
    if isinstance(val, str) and val.endswith("W"):
        return "color:#2ecc71;font-weight:bold"
    if isinstance(val, str) and val.endswith("L"):
        return "color:#e74c3c;font-weight:bold"
    return ""


def _style_l2(val):
    if isinstance(val, float):
        return "color:#2ecc71" if val > 0 else ("color:#e74c3c" if val < 0 else "")
    return ""


PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c",
    "#e67e22","#e91e63","#00bcd4","#8bc34a","#ff5722","#607d8b",
    "#795548","#ffc107","#673ab7","#03a9f4","#4caf50","#ff9800","#9e9e9e",
]


# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Loading data from Google Sheets…"):
    matches, elo_hist, standings, h2h, all_players = load_data()

current_elos   = get_current_elos(elo_hist, all_players)
elo_changes    = get_elo_changes_l2(matches, elo_hist, all_players, n_days=2)
ranked_players = sorted(current_elos, key=lambda p: -current_elos[p])

# Pre-compute results lists for all players (used in multiple tabs)
all_results = {p: get_results_list(matches, p) for p in ranked_players}

# ── Header ────────────────────────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([5, 1])
hdr_l.title("🎯 Darts League Dashboard")
if hdr_r.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

tabs = st.tabs([
    "🏆 Leaderboard", "👤 Player Profile", "⚔️ Head to Head",
    "🎲 Matchup Predictor", "📋 Match History",
    "💥 Upset Log", "📊 Player Comparison",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    total_180s = int(matches["T1 180"].fillna(0).sum() + matches["T2 180"].fillna(0).sum()) \
                 if "T1 180" in matches.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Players", len(ranked_players))
    c2.metric("Total Matches", int(matches["Game"].max() or 0))
    c3.metric("Singles", len(matches[matches["Single/Team"] == "S"]))
    c4.metric("Team Matches", len(matches[matches["Single/Team"] == "T"]))
    c5.metric("Total 180s 🎯", total_180s)

    st.markdown("---")

    # Build leaderboard
    lb_rows = []
    for p in ranked_players:
        s_row  = standings[standings["NAMES"] == p]
        elo    = current_elos[p]
        change = elo_changes.get(p, 0.0)
        g_wl   = s_row["G W-L"].values[0] if not s_row.empty else "0-0"
        l_wl   = s_row["L W-L"].values[0] if not s_row.empty else "0-0"
        rank   = s_row["Rank"].values[0]   if not s_row.empty else "–"
        w, l   = parse_wl(g_wl)
        res    = all_results[p]
        streak, longest_ws = get_streaks(res)
        lb_rows.append({
            "Rank":       rank,
            "Player":     p,
            "ELO":        int(round(elo)),
            "L2 Δ":       change,
            "Streak":     streak,
            "Best W Str": longest_ws,
            "Form":       form_emoji(res, 5),
            "Record":     g_wl,
            "Win%":       f"{w/(w+l)*100:.0f}%" if (w + l) > 0 else "–",
            "Legs":       l_wl,
        })

    lb_df = pd.DataFrame(lb_rows)

    # ELO bar chart
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in lb_df["L2 Δ"]]
    fig = go.Figure(go.Bar(
        x=lb_df["Player"], y=lb_df["ELO"],
        marker_color=colors,
        text=lb_df["ELO"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>ELO: %{y}<extra></extra>",
    ))
    fig.add_hline(y=800, line_dash="dot", line_color="gray",
                  annotation_text="Starting ELO (800)", annotation_position="top right")
    fig.update_layout(
        title="Current ELO Ratings  (green = gained, red = lost over last 2 match days)",
        yaxis_title="ELO", xaxis_title="",
        height=420, margin=dict(t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        lb_df.style
            .map(_style_l2,      subset=["L2 Δ"])
            .map(style_streak,   subset=["Streak"]),
        use_container_width=True, height=560,
    )

    st.markdown("---")

    st.subheader("ELO History — All Players")
    fig2 = go.Figure()
    for i, p in enumerate(ranked_players):
        if p not in elo_hist.columns:
            continue
        s = elo_hist[["Match", p]].dropna(subset=[p])
        if s.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=s["Match"], y=s[p], mode="lines", name=p,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            hovertemplate=f"<b>{p}</b>  Match %{{x}}  ELO %{{y:.0f}}<extra></extra>",
        ))
    fig2.add_hline(y=800, line_dash="dot", line_color="gray", opacity=0.4)
    fig2.update_layout(
        xaxis_title="Match Number", yaxis_title="ELO",
        height=520, hovermode="x unified",
        legend=dict(orientation="v", x=1.01),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 180s Leaderboard ─────────────────────────────────────────────────────
    if "T1 180" in matches.columns and "T2 180" in matches.columns:
        st.markdown("---")
        st.subheader("🎯 180s Leaderboard")

        s180 = matches[["Single/Team", "Player A", "Player B", "Placer C", "Player D",
                         "T1 180", "T2 180", "Date", "Game"]].copy()

        player_180s = {p: 0.0 for p in all_players}

        for _, row in s180.iterrows():
            t1_180 = row["T1 180"] if pd.notna(row["T1 180"]) else 0
            t2_180 = row["T2 180"] if pd.notna(row["T2 180"]) else 0
            mtype  = row["Single/Team"]

            if mtype == "S":
                # Singles: Player A = Team 1, Player B = Team 2
                pa, pb = str(row["Player A"]), str(row["Player B"])
                if pa in player_180s:
                    player_180s[pa] += t1_180
                if pb in player_180s:
                    player_180s[pb] += t2_180
            else:
                # Teams: split evenly between the two partners
                t1_players = [str(row["Player A"]), str(row["Placer C"])]
                t2_players = [str(row["Player B"]), str(row["Player D"])]
                for pp in t1_players:
                    if pp in player_180s:
                        player_180s[pp] += t1_180 / 2
                for pp in t2_players:
                    if pp in player_180s:
                        player_180s[pp] += t2_180 / 2

        s180_df = (
            pd.DataFrame([{"Player": p, "180s": player_180s[p]} for p in ranked_players])
            .sort_values("180s", ascending=False)
            .reset_index(drop=True)
        )
        s180_df["180s"] = s180_df["180s"].round(1)

        # Games played per player for per-game rate
        gpd = {}
        for p in all_players:
            gpd[p] = len(player_matches(matches, p))
        s180_df["Games"] = s180_df["Player"].map(gpd)
        s180_df["Per Game"] = (s180_df["180s"] / s180_df["Games"].replace(0, np.nan)).round(2)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            fig180 = go.Figure(go.Bar(
                x=s180_df["Player"], y=s180_df["180s"],
                marker_color="#f39c12",
                text=s180_df["180s"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>180s: %{y}<extra></extra>",
            ))
            fig180.update_layout(
                title="Total 180s per Player",
                xaxis_title="", yaxis_title="180s",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig180, use_container_width=True)

        with col_r:
            fig_pg = go.Figure(go.Bar(
                x=s180_df.sort_values("Per Game", ascending=False)["Player"],
                y=s180_df.sort_values("Per Game", ascending=False)["Per Game"],
                marker_color="#9b59b6",
                text=s180_df.sort_values("Per Game", ascending=False)["Per Game"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Per game: %{y}<extra></extra>",
            ))
            fig_pg.update_layout(
                title="180s per Game",
                xaxis_title="", yaxis_title="Avg per game",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pg, use_container_width=True)

        st.dataframe(s180_df, use_container_width=True, hide_index=True)

        # Most 180s in a single match
        matches["Total 180s"] = matches["T1 180"].fillna(0) + matches["T2 180"].fillna(0)
        best_match = matches.nlargest(5, "Total 180s")[
            ["Date", "Game", "Single/Team", "Team 1", "Team 2",
             "T1 180", "T2 180", "Total 180s", "Winner"]
        ].reset_index(drop=True)
        st.markdown("**Most 180s in a single match (top 5)**")
        st.dataframe(best_match, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    sel    = st.selectbox("Select player", ranked_players, key="profile_sel")
    s_row  = standings[standings["NAMES"] == sel]
    elo    = current_elos.get(sel, 800)
    rank   = int(s_row["Rank"].values[0]) if not s_row.empty and not pd.isna(s_row["Rank"].values[0]) else "–"
    g_wl   = s_row["G W-L"].values[0] if not s_row.empty else "0-0"
    l_wl   = s_row["L W-L"].values[0] if not s_row.empty else "0-0"
    w, l   = parse_wl(g_wl)
    change = elo_changes.get(sel, 0.0)
    res    = all_results[sel]
    streak, longest_ws = get_streaks(res)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ELO", f"{elo:.0f}", delta=f"{change:+.0f} (L2)" if change else None)
    c2.metric("Rank", rank)
    c3.metric("Overall Record", g_wl)
    c4.metric("Win %", f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "–")
    c5.metric("Streak", streak)
    c6.metric("Best Win Streak", longest_ws)

    # Form strip
    st.markdown(
        "**Last 5:** " + " ".join(
            f'<span style="color:{"#2ecc71" if r=="W" else "#e74c3c"};font-size:22px">●</span>'
            for r in res[-5:]
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        if sel in elo_hist.columns:
            series = elo_hist[["Match", sel]].dropna(subset=[sel]).sort_values("Match").copy()
            if not series.empty:
                series["Peak"] = series[sel].cummax()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series["Match"], y=series[sel],
                    mode="lines+markers", name="ELO",
                    line=dict(color="#3498db", width=2), marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=series["Match"], y=series["Peak"],
                    mode="lines", name="Peak ELO",
                    line=dict(color="#f39c12", width=1.5, dash="dot"),
                ))
                fig.add_hline(y=800, line_dash="dot", line_color="gray", opacity=0.5)
                fig.update_layout(
                    title=f"{sel} — ELO History",
                    xaxis_title="Match Number", yaxis_title="ELO",
                    height=360, legend=dict(orientation="h", y=1.1),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if w + l > 0:
            fig = go.Figure(go.Pie(
                labels=["Wins", "Losses"], values=[w, l],
                marker_colors=["#2ecc71", "#e74c3c"],
                hole=0.5, textinfo="label+percent",
            ))
            fig.update_layout(
                title="Overall W / L", height=340, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── ELO vs Opponent Strength ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Performance vs Opponent ELO")
    singles = matches[matches["Single/Team"] == "S"].copy()
    singles["Team 1 Before"] = pd.to_numeric(singles["Team 1 Before"], errors="coerce")
    singles["Team 2 Before"] = pd.to_numeric(singles["Team 2 Before"], errors="coerce")

    opp_rows = []
    for _, row in singles.iterrows():
        t1, t2 = str(row.get("Team 1", "")), str(row.get("Team 2", ""))
        if sel == t1:
            opp_elo = row["Team 2 Before"]
            won = 1 if sel in str(row["Winner"]) else 0
        elif sel == t2:
            opp_elo = row["Team 1 Before"]
            won = 1 if sel in str(row["Winner"]) else 0
        else:
            continue
        if pd.notna(opp_elo) and opp_elo > 0:
            opp_rows.append({"Opp ELO": opp_elo, "Won": won,
                             "Result": "Win" if won else "Loss"})

    if opp_rows:
        opp_df = pd.DataFrame(opp_rows)
        fig = go.Figure()
        for result, color in [("Win", "#2ecc71"), ("Loss", "#e74c3c")]:
            sub = opp_df[opp_df["Result"] == result]
            fig.add_trace(go.Scatter(
                x=sub["Opp ELO"], y=[result] * len(sub),
                mode="markers",
                marker=dict(color=color, size=12, opacity=0.7),
                name=result,
            ))
        # Win rate by ELO bucket
        opp_df["ELO Bucket"] = (opp_df["Opp ELO"] // 50) * 50
        bucket_wr = opp_df.groupby("ELO Bucket")["Won"].agg(["mean", "count"]).reset_index()
        bucket_wr.columns = ["ELO Bucket", "Win%", "Games"]
        bucket_wr["Win%"] *= 100

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bucket_wr["ELO Bucket"], y=bucket_wr["Win%"],
            text=bucket_wr.apply(lambda r: f"{r['Win%']:.0f}% ({r['Games']}g)", axis=1),
            textposition="outside",
            marker_color=["#2ecc71" if v >= 50 else "#e74c3c" for v in bucket_wr["Win%"]],
        ))
        fig2.add_hline(y=50, line_dash="dot", line_color="gray")
        fig2.update_layout(
            title=f"{sel} — Win% by Opponent ELO Range",
            xaxis_title="Opponent ELO (bucketed by 50)", yaxis_title="Win %",
            yaxis=dict(range=[0, 110]),
            height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough singles data with ELO records to build this chart.")

    # Recent matches
    st.markdown("---")
    st.subheader("Recent Matches")
    pm = player_matches(matches, sel)
    if not pm.empty:
        recent = pm.tail(20).copy()
        recent["Result"] = recent.apply(lambda r: result_for(r, sel), axis=1)
        display_cols = ["Date", "Game", "Single/Team", "SF/Final",
                        "Team 1", "Team 2", "Legs Team 1", "Legs Team 2", "Winner", "Result"]
        st.dataframe(
            recent[display_cols].reset_index(drop=True).style.map(
                color_result, subset=["Result"]
            ),
            use_container_width=True, height=380,
        )

    # H2H summary
    st.subheader("Singles H2H vs All Opponents")
    p_h2h = h2h[h2h.iloc[:, 0] == sel]
    if not p_h2h.empty:
        h2h_rows = []
        for opp in all_players:
            if opp == sel or opp not in p_h2h.columns:
                continue
            wl_str = p_h2h[opp].values[0]
            w2, l2 = parse_wl(wl_str)
            if w2 + l2 == 0:
                continue
            h2h_rows.append({
                "Opponent": opp, "W-L": wl_str, "W": w2, "L": l2,
                "Win%": f"{w2/(w2+l2)*100:.0f}%",
                "Opp ELO": int(round(current_elos.get(opp, 800))),
            })
        if h2h_rows:
            st.dataframe(
                pd.DataFrame(h2h_rows).sort_values("W", ascending=False),
                use_container_width=True, hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEAD TO HEAD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", ranked_players, key="h2h_p1")
    p2 = c2.selectbox("Player 2", [p for p in ranked_players if p != p1], key="h2h_p2")

    elo1, elo2 = current_elos.get(p1, 800), current_elos.get(p2, 800)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{p1} ELO", f"{elo1:.0f}")
    c2.metric("ELO Advantage", f"{elo1 - elo2:+.0f}", help="Positive = Player 1 higher")
    c3.metric(f"{p2} ELO", f"{elo2:.0f}")

    p1_h2h = h2h[h2h.iloc[:, 0] == p1]
    if not p1_h2h.empty and p2 in p1_h2h.columns:
        wl_str = p1_h2h[p2].values[0]
        w1, l1 = parse_wl(wl_str)
        total  = w1 + l1
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{p1} wins", w1)
        c2.metric("Meetings", total)
        c3.metric(f"{p2} wins", l1)
        if total > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[w1], y=["Singles"], orientation="h",
                                 name=p1, marker_color="#3498db",
                                 text=f"{p1}  {w1}", textposition="inside"))
            fig.add_trace(go.Bar(x=[l1], y=["Singles"], orientation="h",
                                 name=p2, marker_color="#e74c3c",
                                 text=f"{p2}  {l1}", textposition="inside"))
            fig.update_layout(
                barmode="stack", height=120, showlegend=False,
                margin=dict(t=5, b=5),
                xaxis=dict(showticklabels=False, showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No singles head-to-head record found.")

    fig = go.Figure()
    for p, color in [(p1, "#3498db"), (p2, "#e74c3c")]:
        if p not in elo_hist.columns:
            continue
        s = elo_hist[["Match", p]].dropna(subset=[p])
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["Match"], y=s[p], mode="lines", name=p,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{p}</b>  ELO %{{y:.0f}}<extra></extra>",
        ))
    fig.add_hline(y=800, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="ELO Over Time", xaxis_title="Match Number", yaxis_title="ELO",
        height=380, hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Match History")
    singles = matches[matches["Single/Team"] == "S"]
    h2h_matches = singles[
        ((singles["Player A"] == p1) & (singles["Player B"] == p2)) |
        ((singles["Player A"] == p2) & (singles["Player B"] == p1))
    ]
    if not h2h_matches.empty:
        st.dataframe(
            h2h_matches[["Date", "Game", "SF/Final", "Team 1", "Team 2",
                         "Legs Team 1", "Legs Team 2", "Winner"]].reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("No singles matches between these two players recorded.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MATCHUP PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    match_type = st.radio("Format", ["Singles (1v1)", "Teams (2v2)"], horizontal=True)

    if match_type == "Singles (1v1)":
        c1, c2 = st.columns(2)
        pp1 = c1.selectbox("Player 1", ranked_players, key="pred_p1")
        pp2 = c2.selectbox("Player 2", [p for p in ranked_players if p != pp1], key="pred_p2")
        e1, e2 = current_elos.get(pp1, 800), current_elos.get(pp2, 800)
        prob   = elo_win_prob(e1, e2)

        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(f"## {pp1}")
            st.metric("ELO", f"{e1:.0f}")
            st.metric("Win Probability", f"{prob*100:.1f}%")
        c2.markdown("<br><br><br><h2 style='text-align:center'>VS</h2>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"## {pp2}")
            st.metric("ELO", f"{e2:.0f}")
            st.metric("Win Probability", f"{(1-prob)*100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            title={"text": f"{pp1} win probability"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [0,  40], "color": "#2d1b1b"},
                    {"range": [40, 60], "color": "#2d2a1b"},
                    {"range": [60, 100], "color": "#1b2d1e"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.8, "value": 50},
            },
            number={"suffix": "%", "valueformat": ".1f"},
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        r = h2h[h2h.iloc[:, 0] == pp1]
        if not r.empty and pp2 in r.columns:
            wl = r[pp2].values[0]
            w, l = parse_wl(wl)
            if w + l > 0:
                st.info(f"Head-to-Head record: **{pp1}** {w}–{l} **{pp2}**")

    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Team 1")
            t1a = st.selectbox("Player A", ranked_players, key="t1a")
            t1b = st.selectbox("Player B", [p for p in ranked_players if p != t1a], key="t1b")
        with c2:
            st.subheader("Team 2")
            remaining = [p for p in ranked_players if p not in (t1a, t1b)]
            t2a = st.selectbox("Player A", remaining, key="t2a")
            t2b = st.selectbox("Player B", [p for p in remaining if p != t2a], key="t2b")

        avg1    = np.mean([current_elos.get(t1a, 800), current_elos.get(t1b, 800)])
        avg2    = np.mean([current_elos.get(t2a, 800), current_elos.get(t2b, 800)])
        prob_t1 = elo_win_prob(avg1, avg2)

        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(f"## {t1a} & {t1b}")
            st.metric("Avg Team ELO", f"{avg1:.0f}")
            st.metric("Win Probability", f"{prob_t1*100:.1f}%")
        c2.markdown("<br><br><br><h2 style='text-align:center'>VS</h2>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"## {t2a} & {t2b}")
            st.metric("Avg Team ELO", f"{avg2:.0f}")
            st.metric("Win Probability", f"{(1-prob_t1)*100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_t1 * 100,
            title={"text": "Team 1 win probability"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [0,  40], "color": "#2d1b1b"},
                    {"range": [40, 60], "color": "#2d2a1b"},
                    {"range": [60, 100], "color": "#1b2d1e"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.8, "value": 50},
            },
            number={"suffix": "%", "valueformat": ".1f"},
        ))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame([
            {"Player": t1a, "Team": "Team 1", "ELO": int(current_elos.get(t1a, 800))},
            {"Player": t1b, "Team": "Team 1", "ELO": int(current_elos.get(t1b, 800))},
            {"Player": t2a, "Team": "Team 2", "ELO": int(current_elos.get(t2a, 800))},
            {"Player": t2b, "Team": "Team 2", "ELO": int(current_elos.get(t2b, 800))},
        ]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MATCH HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    f1, f2, f3 = st.columns(3)
    type_f   = f1.selectbox("Type", ["All", "Singles", "Teams"])
    player_f = f2.selectbox("Player", ["All"] + sorted(all_players))
    stage_f  = f3.selectbox("Stage", ["All"] + sorted(
        [s for s in matches["SF/Final"].dropna().unique()
         if str(s).strip() not in ("", "nan")]
    ))

    filtered = matches.copy()
    if type_f == "Singles":
        filtered = filtered[filtered["Single/Team"] == "S"]
    elif type_f == "Teams":
        filtered = filtered[filtered["Single/Team"] == "T"]
    if player_f != "All":
        pm_mask = (
            (filtered["Player A"] == player_f)
            | (filtered["Player B"] == player_f)
            | (filtered["Placer C"] == player_f)
            | (filtered["Player D"] == player_f)
        )
        filtered = filtered[pm_mask]
    if stage_f != "All":
        filtered = filtered[filtered["SF/Final"] == stage_f]

    st.caption(f"{len(filtered)} matches")
    st.dataframe(
        filtered[["Date", "Game", "Single/Team", "SF/Final",
                  "Team 1", "Team 2", "Legs Team 1", "Legs Team 2", "Winner"]
        ].sort_values("Game", ascending=False).reset_index(drop=True),
        use_container_width=True, height=700,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — UPSET LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Biggest Upsets — Lower ELO Player Won")

    type_filter = st.radio("Match type", ["Singles", "Teams", "Both"], horizontal=True)

    upset_rows = []
    for _, row in matches.iterrows():
        t1_before = row.get("Team 1 Before")
        t2_before = row.get("Team 2 Before")
        if pd.isna(t1_before) or pd.isna(t2_before):
            continue
        if t1_before == t2_before:
            continue
        mtype = str(row.get("Single/Team", ""))
        if type_filter == "Singles" and mtype != "S":
            continue
        if type_filter == "Teams" and mtype != "T":
            continue

        winner = str(row.get("Winner", ""))
        team1  = str(row.get("Team 1", ""))
        team2  = str(row.get("Team 2", ""))
        fav    = team1 if t1_before > t2_before else team2
        dog    = team2 if t1_before > t2_before else team1
        fav_elo = max(t1_before, t2_before)
        dog_elo = min(t1_before, t2_before)
        gap     = fav_elo - dog_elo

        # Upset = underdog won
        if winner == dog:
            prob_upset = 1 - elo_win_prob(fav_elo, dog_elo)
            upset_rows.append({
                "Date":       row.get("Date", ""),
                "Game":       int(row.get("Game", 0)),
                "Type":       mtype,
                "Stage":      row.get("SF/Final", ""),
                "Favourite":  fav,
                "Underdog":   dog,
                "Fav ELO":    int(fav_elo),
                "Dog ELO":    int(dog_elo),
                "ELO Gap":    int(gap),
                "Upset Prob": f"{prob_upset*100:.1f}%",
                "Score":      f"{int(row['Legs Team 1'] or 0)}-{int(row['Legs Team 2'] or 0)}",
            })

    if upset_rows:
        upset_df = pd.DataFrame(upset_rows).sort_values("ELO Gap", ascending=False).reset_index(drop=True)

        # Top metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Upsets", len(upset_df))
        c2.metric("Biggest ELO Gap", f"{upset_df['ELO Gap'].max()} pts")
        biggest = upset_df.iloc[0]
        c3.metric("Biggest Upset", f"{biggest['Underdog']} over {biggest['Favourite']}")

        # ELO gap distribution
        fig = px.histogram(
            upset_df, x="ELO Gap", nbins=20,
            title="Distribution of Upset Margins (ELO Gap)",
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(
            height=300, xaxis_title="ELO Gap",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(upset_df, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No upsets found for this filter.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PLAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Compare Players Side by Side")

    selected = st.multiselect(
        "Select 2–6 players", ranked_players,
        default=ranked_players[:4], key="comp_sel",
    )

    if len(selected) < 2:
        st.warning("Select at least 2 players.")
    else:
        # Stats table
        comp_rows = []
        for p in selected:
            s_row  = standings[standings["NAMES"] == p]
            elo    = current_elos.get(p, 800)
            change = elo_changes.get(p, 0.0)
            g_wl   = s_row["G W-L"].values[0] if not s_row.empty else "0-0"
            l_wl   = s_row["L W-L"].values[0] if not s_row.empty else "0-0"
            rank   = s_row["Rank"].values[0]   if not s_row.empty else "–"
            w, l   = parse_wl(g_wl)
            lw, ll = parse_wl(l_wl)
            res    = all_results[p]
            streak, longest_ws = get_streaks(res)

            # Peak ELO
            peak = float(elo_hist[p].dropna().max()) if p in elo_hist.columns else elo

            comp_rows.append({
                "Player":       p,
                "Rank":         rank,
                "ELO":          int(round(elo)),
                "Peak ELO":     int(round(peak)),
                "L2 Δ":         round(change, 1),
                "Record":       g_wl,
                "Win%":         round(w / (w + l) * 100, 1) if (w + l) > 0 else 0,
                "Legs":         l_wl,
                "Leg Win%":     round(lw / (lw + ll) * 100, 1) if (lw + ll) > 0 else 0,
                "Streak":       streak,
                "Best W Str":   longest_ws,
                "Form (L5)":    form_emoji(res, 5),
            })

        comp_df = pd.DataFrame(comp_rows)

        st.dataframe(
            comp_df.style
                .map(style_streak, subset=["Streak"])
                .map(_style_l2,    subset=["L2 Δ"])
                .highlight_max(subset=["ELO", "Win%", "Leg Win%", "Best W Str", "Peak ELO"],
                               color="#1e3a1e")
                .highlight_min(subset=["ELO", "Win%", "Leg Win%"],
                               color="#3a1e1e"),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")

        # ELO comparison chart
        fig = go.Figure()
        for i, p in enumerate(selected):
            if p not in elo_hist.columns:
                continue
            s = elo_hist[["Match", p]].dropna(subset=[p]).sort_values("Match")
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s["Match"], y=s[p], mode="lines", name=p,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                hovertemplate=f"<b>{p}</b>  ELO %{{y:.0f}}<extra></extra>",
            ))
        fig.add_hline(y=800, line_dash="dot", line_color="gray", opacity=0.4)
        fig.update_layout(
            title="ELO History — Selected Players",
            xaxis_title="Match Number", yaxis_title="ELO",
            height=420, hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Win% radar chart
        categories = ["Win%", "Leg Win%", "Best W Str"]
        fig_radar = go.Figure()
        for i, row in comp_df.iterrows():
            # Normalise best win streak to 0-100 scale
            max_bws = comp_df["Best W Str"].max() or 1
            vals = [
                row["Win%"],
                row["Leg Win%"],
                row["Best W Str"] / max_bws * 100,
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself", name=row["Player"],
                line=dict(color=PALETTE[i % len(PALETTE)]),
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Player Comparison Radar",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
