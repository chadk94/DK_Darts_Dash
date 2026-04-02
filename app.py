import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    # ── Matches ──────────────────────────────────────────────────────────────
    matches = pd.read_csv(URLS["Matches"])
    matches = matches[matches["Game"].notna()].copy()
    for col in ["Legs Team 1", "Legs Team 2", "Game"]:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches = matches[matches["Game"].notna()].copy()

    # ── Match Level ELO ───────────────────────────────────────────────────────
    # Row 0 = "First Singles" info → skip; row 1 = "Match" + player names → use as header
    elo_hist = pd.read_csv(URLS["Match Level Elo"], header=1)
    player_names = [c for c in elo_hist.columns if c != "Match" and str(c).strip() not in ("", "nan")]
    elo_hist["Match"] = pd.to_numeric(elo_hist["Match"], errors="coerce")
    for p in player_names:
        elo_hist[p] = pd.to_numeric(
            elo_hist[p].astype(str).replace(["9999", "nan", ""], np.nan),
            errors="coerce",
        )
    elo_hist = elo_hist.dropna(subset=["Match"])

    # ── Standings ─────────────────────────────────────────────────────────────
    # Row 0 = group labels → skip; row 1 = actual col names → use as header
    standings = pd.read_csv(URLS["Standings"], header=1)
    standings = standings[standings["NAMES"].notna() & (standings["NAMES"].str.strip() != "")].copy()
    standings["Elo"] = pd.to_numeric(standings["Elo"], errors="coerce")
    standings["Rank"] = pd.to_numeric(standings["Rank"], errors="coerce")

    # ── Head-to-head matrix ───────────────────────────────────────────────────
    h2h = pd.read_csv(URLS["Single Opponents"])
    h2h = h2h[h2h.iloc[:, 0].notna() & (h2h.iloc[:, 0].astype(str).str.strip() != "")].copy()

    return matches, elo_hist, standings, h2h, player_names


# ── Helpers ──────────────────────────────────────────────────────────────────

def elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def parse_wl(s) -> tuple[int, int]:
    """'3-2' → (3, 2).  Anything unparseable → (0, 0)."""
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


def get_elo_changes_l10(
    matches: pd.DataFrame, elo_hist: pd.DataFrame, players: list[str], n_days: int = 10
) -> dict[str, float]:
    """ELO change over the last N distinct match days."""
    dates = matches["Date"].dropna()
    dates = dates[dates.astype(str).str.strip() != ""]
    unique_dates = list(dict.fromkeys(dates.tolist()))  # ordered, deduplicated

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
            result[p] = float(current.iloc[-1] - 800)  # started inside the window
        else:
            result[p] = float(current.iloc[-1] - before.iloc[-1])
    return result


def player_matches(matches: pd.DataFrame, player: str) -> pd.DataFrame:
    mask = (
        (matches["Player A"] == player)
        | (matches["Player B"] == player)
        | (matches["Placer C"] == player)
        | (matches["Player D"] == player)
    )
    return matches[mask]


def result_for(row, player: str) -> str:
    return "W" if player in str(row["Winner"]) else "L"


def color_result(val):
    return "color:#2ecc71;font-weight:bold" if val == "W" else "color:#e74c3c;font-weight:bold"


# ── Load ──────────────────────────────────────────────────────────────────────

with st.spinner("Loading data from Google Sheets…"):
    matches, elo_hist, standings, h2h, all_players = load_data()

current_elos = get_current_elos(elo_hist, all_players)
elo_changes  = get_elo_changes_l10(matches, elo_hist, all_players, n_days=2)
ranked_players = sorted(current_elos, key=lambda p: -current_elos[p])

# ── Header ────────────────────────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([5, 1])
hdr_l.title("🎯 Darts League Dashboard")
if hdr_r.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

tabs = st.tabs(
    ["🏆 Leaderboard", "👤 Player Profile", "⚔️ Head to Head",
     "🎲 Matchup Predictor", "📋 Match History"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Players", len(ranked_players))
    c2.metric("Total Matches", int(matches["Game"].max() or 0))
    c3.metric("Singles", len(matches[matches["Single/Team"] == "S"]))
    c4.metric("Team Matches", len(matches[matches["Single/Team"] == "T"]))

    st.markdown("---")

    # Build leaderboard table
    lb_rows = []
    for p in ranked_players:
        s_row = standings[standings["NAMES"] == p]
        elo = current_elos[p]
        if not s_row.empty:
            change = elo_changes.get(p, 0.0)
            g_wl  = s_row["G W-L"].values[0]
            l_wl  = s_row["L W-L"].values[0]
            rank  = s_row["Rank"].values[0]
        else:
            change, g_wl, l_wl, rank = 0.0, "0-0", "0-0", "–"
        w, l = parse_wl(g_wl)
        lb_rows.append({
            "Rank": rank,
            "Player": p,
            "ELO": int(round(elo)),
            "L2 Δ": change,
            "Record": g_wl,
            "Win%": f"{w/(w+l)*100:.0f}%" if (w + l) > 0 else "–",
            "Legs": l_wl,
        })

    lb_df = pd.DataFrame(lb_rows)

    # ELO bar chart
    colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in lb_df["L2 Δ"]]
    fig = go.Figure(go.Bar(
        x=lb_df["Player"], y=lb_df["ELO"],
        marker_color=colors,
        text=lb_df["ELO"],
        textposition="outside",
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

    # Standings table
    def _style_change(val):
        if isinstance(val, float):
            return "color:#2ecc71" if val > 0 else ("color:#e74c3c" if val < 0 else "")
        return ""

    st.dataframe(
        lb_df.style.map(_style_change, subset=["L2 Δ"]),
        use_container_width=True, height=530,
    )

    st.markdown("---")

    # All-player ELO history
    st.subheader("ELO History — All Players")
    fig2 = go.Figure()
    palette = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c",
        "#e67e22","#e91e63","#00bcd4","#8bc34a","#ff5722","#607d8b",
        "#795548","#ffc107","#673ab7","#03a9f4","#4caf50","#ff9800","#9e9e9e",
    ]
    for i, p in enumerate(ranked_players):
        if p not in elo_hist.columns:
            continue
        series = elo_hist[["Match", p]].dropna(subset=[p])
        if series.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=series["Match"], y=series[p],
            mode="lines", name=p,
            line=dict(color=palette[i % len(palette)], width=2),
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    sel = st.selectbox("Select player", ranked_players, key="profile_sel")
    s_row = standings[standings["NAMES"] == sel]

    elo  = current_elos.get(sel, 800)
    rank = int(s_row["Rank"].values[0]) if not s_row.empty and not pd.isna(s_row["Rank"].values[0]) else "–"
    g_wl = s_row["G W-L"].values[0] if not s_row.empty else "0-0"
    l_wl = s_row["L W-L"].values[0] if not s_row.empty else "0-0"
    sg_wl = s_row["G W-L"].values[0] if not s_row.empty else "0-0"   # overall used for singles section too

    # Try to grab singles-specific record (cols 16-17 in standings)
    try:
        sg_wl_raw = s_row["G W-L"].values[0] if "GW" not in s_row.columns else \
                    f"{int(s_row['GW'].values[0])}-{int(s_row['GL'].values[0])}"
    except Exception:
        sg_wl_raw = "–"

    w, l = parse_wl(g_wl)
    lw, ll = parse_wl(l_wl)

    change = elo_changes.get(sel, 0.0)

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ELO", f"{elo:.0f}", delta=f"{change:+.0f} (L2)" if change else None)
    c2.metric("Rank", rank)
    c3.metric("Overall Record", g_wl)
    c4.metric("Win %", f"{w/(w+l)*100:.0f}%" if (w+l) > 0 else "–")
    c5.metric("Legs W-L", l_wl)

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # ELO over time
        if sel in elo_hist.columns:
            series = elo_hist[["Match", sel]].dropna(subset=[sel]).sort_values("Match").copy()
            if not series.empty:
                series["Peak"] = series[sel].cummax()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series["Match"], y=series[sel],
                    mode="lines+markers", name="ELO",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=series["Match"], y=series["Peak"],
                    mode="lines", name="Peak ELO",
                    line=dict(color="#f39c12", width=1.5, dash="dot"),
                ))
                fig.add_hline(y=800, line_dash="dot", line_color="gray",
                              annotation_text="Start", opacity=0.5)
                fig.update_layout(
                    title=f"{sel} — ELO History",
                    xaxis_title="Match Number", yaxis_title="ELO",
                    height=360, legend=dict(orientation="h", y=1.1),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # W/L donut
        if w + l > 0:
            fig = go.Figure(go.Pie(
                labels=["Wins", "Losses"], values=[w, l],
                marker_colors=["#2ecc71", "#e74c3c"],
                hole=0.5, textinfo="label+percent",
            ))
            fig.update_layout(
                title="Overall W / L",
                height=340, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Recent matches
    st.subheader(f"Recent Matches")
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

    # Singles H2H summary table
    st.subheader("Singles Head-to-Head vs All Opponents")
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
                "Opponent": opp,
                "W-L": wl_str,
                "W": w2,
                "L": l2,
                "Win%": f"{w2/(w2+l2)*100:.0f}%",
                "Opp ELO": int(round(current_elos.get(opp, 800))),
            })
        if h2h_rows:
            h2h_tbl = pd.DataFrame(h2h_rows).sort_values("W", ascending=False)
            st.dataframe(h2h_tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEAD TO HEAD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", ranked_players, key="h2h_p1")
    p2 = c2.selectbox("Player 2", [p for p in ranked_players if p != p1], key="h2h_p2")

    elo1, elo2 = current_elos.get(p1, 800), current_elos.get(p2, 800)
    prob1 = elo_win_prob(elo1, elo2)

    # ELO comparison metrics
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{p1} ELO", f"{elo1:.0f}")
    c2.metric("ELO Advantage", f"{elo1 - elo2:+.0f}", help="Positive = Player 1 higher")
    c3.metric(f"{p2} ELO", f"{elo2:.0f}")

    # H2H record
    p1_h2h = h2h[h2h.iloc[:, 0] == p1]
    if not p1_h2h.empty and p2 in p1_h2h.columns:
        wl_str = p1_h2h[p2].values[0]
        w1, l1 = parse_wl(wl_str)
        total = w1 + l1

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

    # ELO progression side by side
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

    # Individual match results
    st.subheader("Match History")
    singles = matches[matches["Single/Team"] == "S"]
    h2h_matches = singles[
        ((singles["Player A"] == p1) & (singles["Player B"] == p2)) |
        ((singles["Player A"] == p2) & (singles["Player B"] == p1))
    ]
    if not h2h_matches.empty:
        disp = h2h_matches[["Date", "Game", "SF/Final", "Team 1", "Team 2",
                             "Legs Team 1", "Legs Team 2", "Winner"]].reset_index(drop=True)
        st.dataframe(disp, use_container_width=True)
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
        prob = elo_win_prob(e1, e2)

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
            mode="gauge+number",
            value=prob * 100,
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

        # H2H note
        r = h2h[h2h.iloc[:, 0] == pp1]
        if not r.empty and pp2 in r.columns:
            wl = r[pp2].values[0]
            w, l = parse_wl(wl)
            if w + l > 0:
                st.info(f"Head-to-Head record: **{pp1}** {w}–{l} **{pp2}**")

    else:  # 2v2
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

        avg1 = np.mean([current_elos.get(t1a, 800), current_elos.get(t1b, 800)])
        avg2 = np.mean([current_elos.get(t2a, 800), current_elos.get(t2b, 800)])
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
            mode="gauge+number",
            value=prob_t1 * 100,
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

        # Show individual ELOs
        elo_breakdown = pd.DataFrame([
            {"Player": t1a, "Team": "Team 1", "ELO": int(current_elos.get(t1a, 800))},
            {"Player": t1b, "Team": "Team 1", "ELO": int(current_elos.get(t1b, 800))},
            {"Player": t2a, "Team": "Team 2", "ELO": int(current_elos.get(t2a, 800))},
            {"Player": t2b, "Team": "Team 2", "ELO": int(current_elos.get(t2b, 800))},
        ])
        st.dataframe(elo_breakdown, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MATCH HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    f1, f2, f3 = st.columns(3)
    type_f   = f1.selectbox("Type", ["All", "Singles", "Teams"])
    player_f = f2.selectbox("Player", ["All"] + sorted(all_players))
    stage_f  = f3.selectbox("Stage", ["All"] + sorted(
        [s for s in matches["SF/Final"].dropna().unique() if str(s).strip() not in ("", "nan")]
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
        use_container_width=True,
        height=700,
    )
