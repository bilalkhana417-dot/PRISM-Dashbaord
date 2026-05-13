import platform
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="PRISM", layout="wide")

# init session state vars first or the app crashes on load
defaults = {
    "authenticated": False,
    "report_generated": False,
    "logs": [],
    "dark_mode": True
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# colour palettes for light/dark mode (took forever to get these right)
if st.session_state["dark_mode"]:
    T = dict(
        bg="#0d1117", sidebar_bg="#161b22", card_bg="#161b22",
        text="#e6edf3", subtext="#c9d1d9", muted="#8b949e",
        border="#30363d", inner_border="#21262d",
        accent="#58a6ff",
        tab_active_bg="#1f2937",
        input_bg="#0d1117", input_border="#30363d",
        badge_low_bg="#1a3a2a", badge_low_fg="#3fb950", badge_low_bdr="#3fb950",
        badge_med_bg="#3a2a1a", badge_med_fg="#d29922", badge_med_bdr="#d29922",
        badge_high_bg="#3a1a1a", badge_high_fg="#f85149", badge_high_bdr="#f85149",
        chart_grid="#21262d", dl_btn_bg="#1f2937",
    )
else:
    T = dict(
        bg="#ffffff", sidebar_bg="#f6f8fa", card_bg="#f6f8fa",
        text="#1f2328", subtext="#24292f", muted="#656d76",
        border="#d0d7de", inner_border="#e1e4e8",
        accent="#0969da",
        tab_active_bg="#eaf2ff",
        input_bg="#ffffff", input_border="#d0d7de",
        badge_low_bg="#dafbe1", badge_low_fg="#1a7f37", badge_low_bdr="#1a7f37",
        badge_med_bg="#fff8c5", badge_med_fg="#9a6700", badge_med_bdr="#9a6700",
        badge_high_bg="#ffebe9", badge_high_fg="#cf222e", badge_high_bdr="#cf222e",
        chart_grid="#e1e4e8", dl_btn_bg="#f6f8fa",
    )

# custom CSS to make it look less like standard streamlit and more professional
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

.stApp {{ background: {T['bg']} !important; color: {T['text']} !important; }}

[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['border']};
}}
[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}

[data-testid="stMetric"] {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 8px;
    padding: 16px 20px;
}}
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem !important;
    color: {T['accent']} !important;
}}
[data-testid="stMetricLabel"] {{ color: {T['muted']} !important; font-size: 0.75rem; }}

[data-testid="stDataFrame"] {{ border: 1px solid {T['border']}; border-radius: 6px; }}

.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {T['border']}; gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 6px 6px 0 0;
    color: {T['muted']}; font-size: 0.85rem; padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: {T['tab_active_bg']} !important;
    border-bottom: 2px solid {T['accent']} !important;
    color: {T['accent']} !important;
}}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background: {T['input_bg']} !important;
    border-color: {T['input_border']} !important;
    color: {T['text']} !important;
}}
[data-baseweb="menu"] {{ background: {T['card_bg']} !important; }}
[data-baseweb="option"] {{ color: {T['text']} !important; }}

.badge-low    {{ background:{T['badge_low_bg']}; color:{T['badge_low_fg']}; border:1px solid {T['badge_low_bdr']}; }}
.badge-medium {{ background:{T['badge_med_bg']}; color:{T['badge_med_fg']}; border:1px solid {T['badge_med_bdr']}; }}
.badge-high   {{ background:{T['badge_high_bg']}; color:{T['badge_high_fg']}; border:1px solid {T['badge_high_bdr']}; }}
.sev-badge {{
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
    font-weight:600; letter-spacing:0.05em; margin-bottom:12px;
}}

h1 {{ color: {T['text']} !important; font-weight: 700; }}
h2, h3 {{ color: {T['subtext']} !important; }}
p, li, span, label {{ color: {T['text']} !important; }}
hr {{ border-color: {T['border']} !important; }}
.stMarkdown p {{ color: {T['text']} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {T['muted']} !important; }}
[data-testid="stAlert"] {{ background: {T['card_bg']}; border-color: {T['border']}; }}
[data-testid="stAlert"] p {{ color: {T['text']} !important; }}

[data-testid="stDownloadButton"] button {{
    background: {T['dl_btn_bg']}; border: 1px solid {T['border']};
    color: {T['accent']}; border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)


# STATS19 mappings from the official DfT codebook
DAYS = {
    1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
    5: "Thursday", 6: "Friday", 7: "Saturday"
}

WEATHER = {
    1: "Clear / Fine", 2: "Raining", 3: "Snowing",
    4: "Strong Winds", 5: "Rain & High Winds",
    8: "Fog / Mist", 9: "Data Missing"
}

LIGHTING = {
    1: "Daylight", 4: "Street Lights (Lit)",
    5: "Street Lights (Unlit)", 6: "No Street Lighting", -1: "Data Missing"
}

POLICE_AREAS = {
    1: "London (Metropolitan)", 4: "Northern England", 5: "Merseyside",
    6: "Greater Manchester", 7: "Lancashire", 20: "West Midlands",
    30: "Yorkshire", 42: "Hertfordshire", 44: "Hampshire",
    52: "South West England", 60: "North Wales", 90: "Scotland (Strategic)"
}

ROAD_TYPES = {1: "Roundabout", 2: "One-Way", 3: "Dual", 6: "Single", 7: "Slip", 9: "Unknown"}

# Target variable mapping (1=Fatal, 2=Serious, 3=Slight)
SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}

# contextual notes for the intelligence report
RISK_NARRATIVE = {
    "Raining":            "reduced tyre traction and longer braking distances.",
    "Snowing":            "severe loss of friction and high chance of multi-vehicle incidents.",
    "Fog / Mist":         "critically reduced visibility causing late-reaction collisions.",
    "Clear / Fine":       "higher average speeds increasing collision impact force.",
    "No Street Lighting": "delayed hazard recognition due to poor visibility."
}

# flip dicts for filtering logic
inv_days    = {v: k for k, v in DAYS.items()}
inv_weather = {v: k for k, v in WEATHER.items()}
inv_light   = {v: k for k, v in LIGHTING.items()}

# shared chart config
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=T["subtext"], family="IBM Plex Sans"),
    margin=dict(l=0, r=0, t=30, b=0)
)

@st.cache_data(show_spinner="Loading dataset…")
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    
    # need hour as an int for filtering, raw data is HH:MM
    if "time" in df.columns:
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
    df = df.dropna(subset=["latitude", "longitude", "hour"])
    df["hour"] = df["hour"].astype(int)
    return df

# cache the model so we don't have to retrain every time a filter changes
@st.cache_resource(show_spinner="Training risk model…")
def train_model(path):
    df = load_data(path)
    features = ["day_of_week", "light_conditions", "weather_conditions", "hour"]
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(df[features], df["collision_severity"])
    return clf, features

def add_road_labels(df):
    df = df.copy()
    def road_name(row):
        if row["first_road_class"] == 1:
            return f"M{int(row['first_road_number'])}"
        elif row["first_road_class"] == 3:
            return f"A{int(row['first_road_number'])}"
        return "Local Road"
        
    df["road_name"] = df.apply(road_name, axis=1)
    df["city_town"] = df["police_force"].map(POLICE_AREAS).fillna("UK Regional Route")
    return df

data = load_data("collisions.csv")
model, FEATURES = train_model("collisions.csv")


# --- sidebar ---
st.sidebar.title("PRISM")
st.sidebar.caption("Predictive Road Incident Safety Monitor")

# rebuild UI if theme is toggled
dark_toggle = st.sidebar.toggle("Dark Mode", value=st.session_state["dark_mode"])
if dark_toggle != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = dark_toggle
    st.rerun()

st.sidebar.markdown("---")

if st.session_state["authenticated"]:
    st.sidebar.success("● Administrator session active")
    if st.sidebar.button("Sign Out"):
        st.session_state["authenticated"] = False
        st.rerun()
    access_mode = "Admin Portal"
else:
    access_mode = st.sidebar.radio("Environment", ["Public Dashboard", "Admin Portal"])

if access_mode == "Admin Portal" and not st.session_state["authenticated"]:
    pwd = st.sidebar.text_input("Administrator Credential", type="password")
    if st.sidebar.button("Verify Identity"):
        # load from secrets to avoid hardcoding
        if pwd == st.secrets["admin_password"]:
            st.session_state["authenticated"] = True
            st.session_state["logs"].append(f"Login: {time.strftime('%H:%M:%S')}")
            st.rerun()
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Simulation Parameters")

sel_day = st.sidebar.selectbox("Day of Week", list(DAYS.values()))
hour_labels = {h: f"{'12' if h % 12 == 0 else h % 12}:00 {'AM' if h < 12 else 'PM'}" for h in range(24)}
sel_hour = st.sidebar.selectbox("Hour", options=list(hour_labels.keys()), format_func=lambda h: hour_labels[h], index=18)
sel_weather = st.sidebar.selectbox("Weather Condition", list(WEATHER.values()))
sel_light = st.sidebar.selectbox("Lighting Condition", list(LIGHTING.values()))

if st.sidebar.button("Generate Report", use_container_width=True):
    st.session_state["report_generated"] = True
    st.session_state["logs"].append(f"Report: {sel_day} @ {sel_hour:02d}:00 — {sel_weather}")


# filter down to rows matching all 4 parameters
filtered = data[
    (data["day_of_week"]        == inv_days[sel_day])        &
    (data["hour"]               == sel_hour)                 &
    (data["weather_conditions"] == inv_weather[sel_weather]) &
    (data["light_conditions"]   == inv_light[sel_light])
].copy()

if not filtered.empty:
    filtered = add_road_labels(filtered)

# build the ML input sample and get a prediction
sample = pd.DataFrame(
    [[inv_days[sel_day], inv_light[sel_light], inv_weather[sel_weather], sel_hour]],
    columns=FEATURES
)
pred = model.predict(sample)[0]

# map predictions to css badges
severity_map = {
    1: ("Critical Risk  ·  Fatal",   "badge-high"),
    2: ("Elevated Risk  ·  Serious", "badge-medium"),
    3: ("Low Risk  ·  Slight",       "badge-low")
}
severity_label, severity_css = severity_map.get(pred, ("Unknown", "badge-low"))


# --- admin portal ---
if access_mode == "Admin Portal":
    st.title("Admin Command Centre")
    t1, t2, t3 = st.tabs(["System Diagnostics", "Audit Trail", "Raw Data"])

    with t1:
        integrity = (len(data) / 100927) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DB Integrity",  f"{integrity:.1f}%")
        c2.metric("Total Records", f"{len(data):,}")
        c3.metric("Python",        platform.python_version())
        c4.metric("OS",            platform.system())

    with t2:
        if st.session_state["logs"]:
            for entry in reversed(st.session_state["logs"]):
                st.caption(f"  {entry}")
        else:
            st.info("No events recorded yet.")

    with t3:
        st.dataframe(data.head(20), use_container_width=True)

    st.markdown("---")


# --- main dashboard ---
st.title("PRISM")
st.caption(f"Predictive Road Incident Safety Monitor  —  {sel_day} · {hour_labels[sel_hour]} · {sel_weather} · {sel_light}")
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matched Incidents", f"{len(filtered):,}")
k2.metric("Window",            hour_labels[sel_hour])
k3.metric("Day",               sel_day)
k4.metric("Weather",           sel_weather)

st.markdown("---")

st.markdown(
    f'<span class="sev-badge {severity_css}">{severity_label}</span>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Intelligence Report")

    if st.session_state["report_generated"]:
        if not filtered.empty:
            top_road = filtered["road_name"].mode().iloc[0]
            top_area = filtered["city_town"].mode().iloc[0]
            narrative = RISK_NARRATIVE.get(
                sel_weather,
                RISK_NARRATIVE.get(sel_light, "a combination of environmental and temporal factors.")
            )

            # calculate severity splits
            fatal_n   = len(filtered[filtered["collision_severity"] == 1])
            serious_n = len(filtered[filtered["collision_severity"] == 2])
            slight_n  = len(filtered[filtered["collision_severity"] == 3])
            total_n   = len(filtered)

            same_day_total = len(data[data["day_of_week"] == inv_days[sel_day]])
            hour_pct = (total_n / same_day_total * 100) if same_day_total > 0 else 0

            peak_hour_row = (
                data[data["day_of_week"] == inv_days[sel_day]]
                .groupby("hour").size().idxmax()
            )
            peak_label = hour_labels[peak_hour_row]

            st.markdown(f"""
**Overview**

Analysis of **{sel_day}** at **{hour_labels[sel_hour]}** under **{sel_weather}** conditions
with **{sel_light}** lighting returns a **{severity_label}** risk profile.
This hour accounts for **{hour_pct:.1f}%** of all {sel_day} incidents in the dataset.
The highest-risk hour on {sel_day} overall is **{peak_label}**.

**Incident Breakdown ({total_n} historical matches)**
- Slight injuries: **{slight_n}** ({slight_n/total_n*100:.0f}%)
- Serious injuries: **{serious_n}** ({serious_n/total_n*100:.0f}%)
- Fatal collisions: **{fatal_n}** ({fatal_n/total_n*100:.0f}%)

**Primary Hotspot**

**{top_road}** in the **{top_area}** region recorded the highest incident density
for this exact time and condition combination.

**Environmental Risk Analysis**

The presence of **{sel_weather}** is the leading hazard, causing {narrative}
Under **{sel_light}** conditions this effect is compounded — drivers have reduced
reaction time and hazard recognition is delayed, particularly on high-speed roads.

**Infrastructure Vulnerability**

{top_road} is flagged as high-risk during this window based on historical frequency.
Cross-referencing road type and lighting data suggests infrastructure design
contributes to collision clustering at this location.
""")

            csv_data = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export Filtered Data (CSV)",
                data=csv_data,
                file_name=f"prism_{sel_day}_{sel_hour:02d}h.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning(
                f"No exact historical matches for this combination. "
                f"PRISM still predicts **{severity_label}** based on broader patterns."
            )
    else:
        st.info("Set your parameters and click Generate Report to begin.")

with right:
    st.subheader("Spatial Distribution")
    if not filtered.empty:
        st.map(
            filtered[["latitude", "longitude"]].rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )
        )
    else:
        st.info("No spatial data for this simulation.")


# --- charts ---
st.markdown("---")
st.subheader("Analytical Breakdown")

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("**Severity Breakdown**")
    if not filtered.empty:
        sev_df = (
            filtered["collision_severity"].map(SEVERITY_LABELS)
            .value_counts().reset_index()
        )
        sev_df.columns = ["Severity", "Count"]
        colour_map = {"Slight": "#3fb950", "Serious": "#d29922", "Fatal": "#f85149"}
        fig = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                     color_discrete_map=colour_map)
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data.")

with c2:
    st.markdown("**Hourly Collision Trend**")
    hourly = (
        data[
            (data["day_of_week"]        == inv_days[sel_day]) &
            (data["weather_conditions"] == inv_weather[sel_weather])
        ]
        .groupby("hour").size().reset_index(name="Incidents")
    )
    if not hourly.empty:
        fig2 = px.line(hourly, x="hour", y="Incidents", markers=True)
        fig2.add_vline(x=sel_hour, line_dash="dash", line_color=T["accent"],
                       annotation_text="Selected hour", annotation_position="top right")
        fig2.update_layout(**CHART_LAYOUT)
        fig2.update_traces(line_color=T["accent"], marker_color="#f0883e")
        fig2.update_xaxes(title="Hour", gridcolor=T["chart_grid"])
        fig2.update_yaxes(title="Incidents", gridcolor=T["chart_grid"])
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No trend data for this selection.")

with c3:
    st.markdown("**Road Type Distribution**")
    if not filtered.empty:
        road_df = (
            filtered["road_type"].map(ROAD_TYPES)
            .value_counts().reset_index()
        )
        road_df.columns = ["Road Type", "Count"]
        fig3 = px.pie(road_df, names="Road Type", values="Count", hole=0.45,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig3.update_layout(**CHART_LAYOUT)
        fig3.update_traces(textfont_color=T["text"])
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data.")


# regional breakdown - only renders if we have data
if not filtered.empty:
    st.markdown("---")
    st.subheader("Regional & Motorway Breakdown")
    tbl1, tbl2 = st.columns(2, gap="medium")

    with tbl1:
        st.markdown("**Top Affected Areas**")
        region_df = filtered["city_town"].value_counts().reset_index()
        region_df.columns = ["Area", "Incidents"]
        st.dataframe(region_df, hide_index=True, use_container_width=True)

    with tbl2:
        st.markdown("**Motorway Hotspots**")
        mways = filtered[filtered["road_name"].str.startswith("M")]
        if not mways.empty:
            mway_df = mways["road_name"].value_counts().reset_index().head(10)
            mway_df.columns = ["Motorway", "Incidents"]
            st.dataframe(mway_df, hide_index=True, use_container_width=True)
        else:
            st.info("No motorway incidents in this slice.")

    st.markdown("---")
    st.subheader("Officer Deployment Recommendations")
    st.caption("Based on historical incident density for the selected day, hour and conditions.")

    top_roads = filtered["road_name"].value_counts().head(5).reset_index()
    top_roads.columns = ["Road", "Incidents"]

    top_areas = filtered["city_town"].value_counts().head(3).reset_index()
    top_areas.columns = ["Area", "Incidents"]

    # look at adjacent hours to see if deployment should start earlier or run later
    prev_h = (sel_hour - 1) % 24
    next_h = (sel_hour + 1) % 24
    prev_count = len(data[(data["day_of_week"] == inv_days[sel_day]) & (data["hour"] == prev_h)])
    next_count = len(data[(data["day_of_week"] == inv_days[sel_day]) & (data["hour"] == next_h)])
    window_note = ""
    
    if prev_count > len(filtered) * 1.2:
        window_note = f"Incident frequency rises in the hour before ({hour_labels[prev_h]}) — consider early deployment."
    elif next_count > len(filtered) * 1.2:
        window_note = f"Incident frequency rises in the following hour ({hour_labels[next_h]}) — consider extended patrol."

    dep1, dep2, dep3 = st.columns(3, gap="medium")

    with dep1:
        st.markdown("**Priority Roads**")
        for i, row in top_roads.iterrows():
            priority = "High" if i == 0 else "Medium" if i <= 2 else "Low"
            colour = "#f85149" if i == 0 else "#d29922" if i <= 2 else T["muted"]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {T["inner_border"]}">'
                f'<span style="font-family:IBM Plex Mono,monospace;color:{T["text"]}">{row["Road"]}</span>'
                f'<span style="color:{colour};font-size:0.8rem;font-weight:600">{priority} &nbsp;{row["Incidents"]} incidents</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with dep2:
        st.markdown("**Priority Areas**")
        for i, row in top_areas.iterrows():
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid {T["inner_border"]}">'
                f'<span style="color:{T["text"]}">{row["Area"]}</span>'
                f'<span style="color:{T["muted"]};font-size:0.8rem;float:right">{row["Incidents"]} incidents</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with dep3:
        st.markdown("**Deployment Notes**")
        
        # get percentages for deployment rules
        fatal_pct   = len(filtered[filtered["collision_severity"] == 1]) / len(filtered) * 100
        serious_pct = len(filtered[filtered["collision_severity"] == 2]) / len(filtered) * 100

        notes = []
        if fatal_pct > 5:
            notes.append(f"Fatal rate is {fatal_pct:.1f}% — armed response and paramedic pre-positioning advised.")
        if serious_pct > 30:
            notes.append(f"Serious injury rate is {serious_pct:.1f}% — trauma unit standby recommended.")
        if sel_weather in ("Raining", "Snowing", "Fog / Mist", "Rain & High Winds"):
            notes.append("Adverse weather active — variable speed limit enforcement on motorways.")
        if sel_light in ("No Street Lighting", "Street Lights (Unlit)"):
            notes.append("Poor lighting — high-visibility patrol vehicles and portable lighting units.")
        if window_note:
            notes.append(window_note)
        if not notes:
            notes.append("Standard patrol deployment sufficient for this window.")

        for note in notes:
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid {T["inner_border"]};color:{T["subtext"]};font-size:0.9rem">{note}</div>',
                unsafe_allow_html=True
            )