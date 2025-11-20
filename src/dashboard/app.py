import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import smtplib
from email.message import EmailMessage
from smtplib import SMTPAuthenticationError
import sqlite3
from pathlib import Path
from datetime import datetime
from PIL import Image  # for logo
from typing import Optional
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except Exception:
    create_engine = None
    text = None
    Engine = None
import os

# ===========================================================
# CONFIG
# ===========================================================

REGION_FILES = {
    "Delhi": "data/regions/delhi_sos.csv",
    "Kerala": "data/regions/kerala_sos.csv",
    "Maharashtra": "data/regions/maharashtra_sos.csv",
    "Assam": "data/regions/assam_sos.csv",
}

# Admin credentials
ADMIN_USERNAME = "control_center"     # change if you want
ADMIN_PASSWORD = "praansetu123"       # change or keep

# Email config
# IMPORTANT:
# 1. Turn on 2-Step Verification in your Google account
# 2. Create an "App Password"
# 3. Put that 16-character app password below
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "singhnitant28@gmail.com"            # your email
SENDER_PASSWORD = "gmzk aluc rphr zjsy"          # <--- PUT REAL APP PASSWORD HERE
ADMIN_ALERT_EMAIL = "singhnitant28@gmail.com"       # where you receive alerts

# Offline DB (SQLite) config
DB_PATH = Path("data/praansetu.db")

# In future, you can wire any online DB here
USE_LOCAL_DB = True
# Remote DB: if you set `REMOTE_DB_URL` below or export `DATABASE_URL`/`REMOTE_DB_URL` env var,
# the app will try to use PostgreSQL via SQLAlchemy.
USE_REMOTE_DB = False         # placeholder flag (can be enabled by setting REMOTE_DB_URL or env var)
REMOTE_DB_URL = ""            # placeholder for PostgreSQL connection string

# Allow configuring remote DB via environment variable for safer credentials handling
if not REMOTE_DB_URL:
    REMOTE_DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("REMOTE_DB_URL", "")
if REMOTE_DB_URL:
    USE_REMOTE_DB = True

# Logo path
# Updated to match the actual file present in the workspace
LOGO_PATH = "assets/praansetu1_logo.png"

# Language options
LANG_OPTIONS = {
    "English": "en",
    "हिन्दी": "hi",
    "मराठी": "mr",
    "தமிழ் / മലയാളம்": "ta_ml",
}

# Basic translation dictionary for key UI strings
TEXTS = {
    "title": {
        "en": "PraanSetu – The Bridge of Life",
        "hi": "प्राणसेतु – जीवन की सेतु",
        "mr": "प्राणसेतु – जीवनाची कडी",
        "ta_ml": "ப்ராண் சேது – வாழ்க்கையின் பாலம்",
    },
    "citizen_portal": {
        "en": "Citizen Portal – Request Help",
        "hi": "नागरिक पोर्टल – मदद के लिए अनुरोध",
        "mr": "नागरिक पोर्टल – मदतीसाठी विनंती",
        "ta_ml": "குடிமக்கள் தளம் – உதவி கோருங்கள்",
    },
    "send_sos": {
        "en": "Send SOS with Location",
        "hi": "लोकेशन के साथ SOS भेजें",
        "mr": "स्थानासह SOS पाठवा",
        "ta_ml": "இடத்துடன் SOS அனுப்பவும்",
    },
    "send_plain": {
        "en": "Send Simple Message",
        "hi": "साधारण संदेश भेजें",
        "mr": "साधा संदेश पाठवा",
        "ta_ml": "எளிய செய்தியை அனுப்பவும்",
    },
    "about_contact": {
        "en": "About & Contact",
        "hi": "परिचय और संपर्क",
        "mr": "माहिती आणि संपर्क",
        "ta_ml": "தகவல் & தொடர்பு",
    },
}


def T(key: str, default: str = "") -> str:
    lang = st.session_state.get("lang", "en")
    return TEXTS.get(key, {}).get(lang, default or key)


def render_header():
    """Top header with logo + product title + subtitle."""
    # Use a horizontal flex layout so the logo and text are vertically centered
    cols = st.columns([0.16, 0.84])
    with cols[0]:
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, use_container_width=True)
        except Exception:
            st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

    # Center-align and enlarge the title and subtitle for better visibility
    with cols[1]:
        st.markdown(
            """
            <div style="display:flex; align-items:center; height:100%;">
              <div style="padding-left:8px;">
                <div style="font-size:36px; font-weight:800; color:#F97316; letter-spacing:0.03em; line-height:1; text-align:left;">
                  PraanSetu
                </div>
                <div style="font-size:16px; color:#CBD5F5; margin-top:4px; text-align:left;">
                  AI-powered Disaster Intelligence & Last-Mile SOS Platform
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# City metadata: state -> city -> lat/lon
CITY_METADATA = {
    "Delhi": {
        "Delhi City Center (Connaught Place)": {"lat": 28.6139, "lon": 77.2090},
        "East Delhi (Mayur Vihar side)": {"lat": 28.6085, "lon": 77.2940},
        "West Delhi (Tilak Nagar side)": {"lat": 28.6400, "lon": 77.0780},
        "South Delhi (South Extension)": {"lat": 28.5675, "lon": 77.2275},
        "North Delhi (Rohini side)": {"lat": 28.7090, "lon": 77.1120},
    },
    "Kerala": {
        "Chengannur": {"lat": 9.3180, "lon": 76.6150},
        "Kuttanad": {"lat": 9.3900, "lon": 76.4100},
        "Aluva": {"lat": 10.1070, "lon": 76.3510},
        "Munnar Region": {"lat": 10.0890, "lon": 77.0620},
        "Kochi (Ernakulam)": {"lat": 9.9620, "lon": 76.3010},
    },
    "Maharashtra": {
        "Mumbai Island City": {"lat": 18.9388, "lon": 72.8354},
        "Thane Creek Area": {"lat": 19.2183, "lon": 72.9781},
        "Pune City": {"lat": 18.5204, "lon": 73.8567},
    },
    "Assam": {
        "Guwahati City": {"lat": 26.1445, "lon": 91.7362},
        "Dibrugarh Town": {"lat": 27.4728, "lon": 94.9110},
        "Jorhat": {"lat": 26.7509, "lon": 94.2037},
    },
}

# ===========================================================
# OFFLINE DB (SQLite)
# ===========================================================

@st.cache_resource(show_spinner=False)
def get_db_connection():
    """Create or return a single SQLite connection."""
    # If remote DB configured and available, return a SQLAlchemy engine
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        # create engine and ensure tables exist remotely
        engine = create_engine(REMOTE_DB_URL, future=True)
        # create tables if not exist using SQL - works for Postgres and MySQL
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
        CREATE TABLE IF NOT EXISTS live_sos (
            id SERIAL PRIMARY KEY,
            state TEXT,
            city TEXT,
            location_name TEXT,
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,
            severity TEXT,
            message TEXT,
            channel TEXT,
            created_at TIMESTAMP
        );
                    """
                )
            )
            conn.execute(
                text(
                    """
        CREATE TABLE IF NOT EXISTS plain_messages (
            id SERIAL PRIMARY KEY,
            contact TEXT,
            message TEXT,
            created_at TIMESTAMP,
            status TEXT
        );
                    """
                )
            )
            conn.execute(
                text(
                    """
        CREATE TABLE IF NOT EXISTS sos_images (
            id SERIAL PRIMARY KEY,
            live_sos_id INTEGER,
            filename TEXT,
            created_at TIMESTAMP
        );
                    """
                )
            )
        return engine

    # Fallback to local sqlite
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Create tables if not exist
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_sos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            city TEXT,
            location_name TEXT,
            lat REAL,
            lon REAL,
            severity TEXT,
            message TEXT,
            channel TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plain_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact TEXT,
            message TEXT,
            created_at TEXT,
            status TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sos_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            live_sos_id INTEGER,
            filename TEXT,
            created_at TEXT
        );
        """
    )
    conn.commit()
    return conn


def db_insert_live_sos(row: dict):
    """Insert a live SOS into the offline DB."""
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine: Engine = get_db_connection()
        insert_sql = text(
            """
        INSERT INTO live_sos (state, city, location_name, lat, lon, severity, message, channel, created_at)
        VALUES (:state, :city, :location_name, :lat, :lon, :severity, :message, :channel, :created_at)
        RETURNING id
        """
        )
        params = {
            "state": row.get("state"),
            "city": row.get("city"),
            "location_name": row.get("location_name"),
            "lat": float(row.get("lat")) if row.get("lat") is not None else None,
            "lon": float(row.get("lon")) if row.get("lon") is not None else None,
            "severity": row.get("severity"),
            "message": row.get("message"),
            "channel": row.get("channel", "Web"),
            "created_at": datetime.utcnow(),
        }
        with engine.begin() as conn:
            res = conn.execute(insert_sql, params)
            new_id = int(res.scalar())
        return new_id

    # local sqlite path
    conn = get_db_connection()
    cur = conn.execute(
        """
        INSERT INTO live_sos (state, city, location_name, lat, lon,
                              severity, message, channel, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("state"),
            row.get("city"),
            row.get("location_name"),
            float(row.get("lat")),
            float(row.get("lon")),
            row.get("severity"),
            row.get("message"),
            row.get("channel", "Web"),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def db_insert_sos_image(live_sos_id: int, filename: str):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO sos_images (live_sos_id, filename, created_at)
        VALUES (?, ?, ?)
        """,
        (int(live_sos_id), filename, datetime.utcnow().isoformat()),
    )
    conn.commit()


def db_get_sos_images() -> pd.DataFrame:
    conn = get_db_connection()
    return pd.read_sql_query("SELECT * FROM sos_images ORDER BY created_at DESC", conn)


def save_image_file(uploaded_file, live_id: int) -> str:
    """Save uploaded file bytes to data/sos_images and return the saved path."""
    images_dir = DB_PATH.parent / "sos_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{live_id}_{safe_name}"
    target = images_dir / fname
    # uploaded_file may be a Streamlit UploadedFile
    data = uploaded_file.read()
    with open(target, "wb") as f:
        f.write(data)
    return str(target)


def db_update_sos_image(record_id: int, new_filename: str):
    conn = get_db_connection()
    conn.execute(
        "UPDATE sos_images SET filename = ? WHERE id = ?",
        (new_filename, int(record_id)),
    )
    conn.commit()


def db_delete_sos_image(record_id: int):
    conn = get_db_connection()
    conn.execute("DELETE FROM sos_images WHERE id = ?", (int(record_id),))
    conn.commit()


def db_get_live_sos() -> pd.DataFrame:
    """Return all live SOS from DB."""
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        return pd.read_sql_query("SELECT * FROM live_sos ORDER BY created_at DESC", engine)
    conn = get_db_connection()
    return pd.read_sql_query("SELECT * FROM live_sos ORDER BY created_at DESC", conn)


def db_update_live_sos_severity(record_id: int, new_severity: str):
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("UPDATE live_sos SET severity = :sev WHERE id = :id"), {"sev": new_severity, "id": int(record_id)})
        return
    conn = get_db_connection()
    conn.execute(
        "UPDATE live_sos SET severity = ? WHERE id = ?",
        (new_severity, record_id),
    )
    conn.commit()


def db_delete_live_sos(record_id: int):
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM live_sos WHERE id = :id"), {"id": int(record_id)})
        return
    conn = get_db_connection()
    conn.execute("DELETE FROM live_sos WHERE id = ?", (record_id,))
    conn.commit()


def db_insert_plain_message(contact: str, message: str):
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO plain_messages (contact, message, created_at, status) VALUES (:c, :m, :t, :s)"), {"c": contact, "m": message, "t": datetime.utcnow(), "s": "new"})
        return
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO plain_messages (contact, message, created_at, status)
        VALUES (?, ?, ?, ?)
        """,
        (contact, message, datetime.utcnow().isoformat(), "new"),
    )
    conn.commit()


def db_get_plain_messages() -> pd.DataFrame:
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        return pd.read_sql_query("SELECT * FROM plain_messages ORDER BY created_at DESC", engine)
    conn = get_db_connection()
    return pd.read_sql_query(
        "SELECT * FROM plain_messages ORDER BY created_at DESC", conn
    )


def db_delete_plain_message(record_id: int):
    if USE_REMOTE_DB and REMOTE_DB_URL and create_engine is not None:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM plain_messages WHERE id = :id"), {"id": int(record_id)})
        return
    conn = get_db_connection()
    conn.execute("DELETE FROM plain_messages WHERE id = ?", (record_id,))
    conn.commit()


# ===========================================================
# DATA & RISK LOGIC
# ===========================================================

@st.cache_data(show_spinner=False)
def load_region_sos(region: str) -> pd.DataFrame:
    """
    Load historical SOS data for a given region (CSV).
    Used as baseline, then combined with live DB data.
    """
    path = REGION_FILES.get(region)
    if path is None:
        raise ValueError(f"No dataset configured for region: {region}")

    df = pd.read_csv(path)

    # required columns
    expected_cols = {"id", "message", "location_name", "lat", "lon", "severity", "source"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV for region {region} is missing columns: {missing}")

    # optional columns with defaults
    if "state" not in df.columns:
        df["state"] = region
    if "channel" not in df.columns:
        df["channel"] = "Historical"

    df["severity_norm"] = df["severity"].str.lower().map(
        {"high": 3, "medium": 2, "low": 1}
    ).fillna(1)

    df["priority_score"] = df["severity_norm"]

    return df


def point_in_polygon(lat, lon, polygon):
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    return (min(lats) <= lat <= max(lats)) and (min(lons) <= lon <= max(lons))


def create_flood_polygon(center_lat: float, center_lon: float):
    delta = 0.01
    return [
        [center_lat + delta, center_lon - delta],
        [center_lat + delta, center_lon + delta],
        [center_lat - delta, center_lon + delta],
        [center_lat - delta, center_lon - delta],
    ]


def zone_from_risk(risk_level: str) -> str:
    """
    Map internal risk level to simple zones for citizens:
    - Red   = highest danger
    - Yellow= medium
    - Green = relatively safer
    """
    if risk_level in ["Extreme", "High", "High (outside flood)"]:
        return "Red"
    elif risk_level == "Medium":
        return "Yellow"
    else:
        return "Green"


def classify_risk(df: pd.DataFrame, flood_polygon):
    in_zone = []
    risk = []

    for _, row in df.iterrows():
        inside = point_in_polygon(row["lat"], row["lon"], flood_polygon)
        in_zone.append(inside)

        sev = row["severity_norm"]
        if inside and sev == 3:
            risk.append("Extreme")
        elif inside and sev >= 2:
            risk.append("High")
        elif inside:
            risk.append("Medium")
        elif not inside and sev == 3:
            risk.append("High (outside flood)")
        else:
            risk.append("Lower")

    df["in_flood_zone"] = in_zone
    df["risk_level"] = risk
    df["zone"] = df["risk_level"].apply(zone_from_risk)
    return df


def simple_stats(sos_df: pd.DataFrame) -> dict:
    total = len(sos_df)
    high = (sos_df["severity"].str.lower() == "high").sum()
    medium = (sos_df["severity"].str.lower() == "medium").sum()
    low = (sos_df["severity"].str.lower() == "low").sum()

    extreme_risk = (sos_df["risk_level"] == "Extreme").sum()
    high_risk = (sos_df["risk_level"].str.contains("High")).sum()
    live_count = (sos_df["source"] == "Live SOS").sum()

    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "extreme_risk": extreme_risk,
        "high_risk": high_risk,
        "live_count": live_count,
    }


# ===========================================================
# MAP
# ===========================================================

def create_map(sos_df: pd.DataFrame, flood_polygon, selected_location: str):
    center_lat = sos_df["lat"].mean()
    center_lon = sos_df["lon"].mean()

    # Default satellite view
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="Esri.WorldImagery",
        name="Satellite",
    )

    # Add gray base as alternative
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Gray Canvas",
        name="Gray Base",
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

    # Flood polygon
    folium.Polygon(
        locations=flood_polygon,
        tooltip="Estimated Flood Zone",
        fill=True,
        fill_opacity=0.35,
        color="blue",
        weight=2,
    ).add_to(fmap)

    color_map = {
        "Extreme": "red",
        "High": "orange",
        "High (outside flood)": "darkorange",
        "Medium": "yellow",
        "Lower": "green",
    }

    selected_lat = None
    selected_lon = None
    if selected_location and selected_location != "All locations":
        sel = sos_df[sos_df["location_name"] == selected_location]
        if not sel.empty:
            selected_lat = sel.iloc[0]["lat"]
            selected_lon = sel.iloc[0]["lon"]
            fmap.location = [selected_lat, selected_lon]
            fmap.zoom_start = 14

    for _, row in sos_df.iterrows():
        risk = row["risk_level"]
        color = color_map.get(risk, "blue")
        radius = 7

        if selected_lat is not None and row["lat"] == selected_lat and row["lon"] == selected_lon:
            radius = 11  # highlight selected

        popup_text = (
            f"<b>{row['location_name']}</b><br>"
            f"{row['message']}<br>"
            f"Severity: <b>{row['severity']}</b><br>"
            f"Risk Level: <b>{row['risk_level']}</b><br>"
            f"Zone: <b>{row['zone']}</b><br>"
            f"Source: <b>{row.get('source', 'Unknown')}</b>"
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="black",
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup_text,
        ).add_to(fmap)

    return fmap


# ===========================================================
# EMAIL
# ===========================================================

def send_email(to_email: str, subject: str, body: str):
    if not EMAIL_ENABLED:
        return "Email disabled (EMAIL_ENABLED=False)"

    if "YOUR_APP_PASSWORD_HERE" in SENDER_PASSWORD:
        return "Email disabled: configure SENDER_PASSWORD with a real Gmail app password."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return "OK"
    except SMTPAuthenticationError:
        return "AUTH_ERROR: Gmail rejected the username/password. Use a valid Gmail app password."
    except Exception as e:
        return f"ERROR: {e}"


def send_new_sos_email(row: pd.Series):
    body = (
        "New Live SOS received in PraanSetu:\n\n"
        f"Region: {row.get('state', st.session_state.get('region', 'Unknown'))}\n"
        f"Location: {row['location_name']}\n"
        f"Severity: {row['severity']}\n"
        f"Message: {row['message']}\n"
        f"Lat/Lon: {row['lat']}, {row['lon']}\n"
        f"Channel: {row.get('channel', 'Web')}\n"
        f"Source: {row.get('source', 'Live SOS')}\n"
    )
    return send_email(ADMIN_ALERT_EMAIL, "PraanSetu – New Live SOS", body)


def send_plain_message_email(contact: str, message: str):
    body = (
        "New Plain Message (No GPS) received in PraanSetu:\n\n"
        f"Contact: {contact}\n"
        f"Message: {message}\n"
    )
    return send_email(ADMIN_ALERT_EMAIL, "PraanSetu – Plain SOS Message", body)


# ===========================================================
# WEATHER (DUMMY NOW, API LATER)
# ===========================================================

def get_dummy_weather(state: str):
    sample = {
        "Delhi": {
            "temp": "32°C",
            "rain": "Heavy rain expected in next 3 hours",
            "risk": "High local flooding risk in low-lying colonies",
        },
        "Kerala": {
            "temp": "27°C",
            "rain": "Moderate to heavy showers over next 12 hours",
            "risk": "River levels rising; watch for embankment overflow",
        },
        "Maharashtra": {
            "temp": "29°C",
            "rain": "Intermittent heavy showers",
            "risk": "Urban flooding possible near creeks and low areas",
        },
        "Assam": {
            "temp": "26°C",
            "rain": "Continuous moderate rain",
            "risk": "River and embankment flooding risk near Brahmaputra",
        },
    }
    return sample.get(state, {"temp": "-", "rain": "-", "risk": "-"})


def get_weather_for_location(lat: float, lon: float, state: str):
    """
    Placeholder: in deployment, this would call IMD / national APIs / OpenWeather
    with (lat, lon) to get real-time forecast.

    For now, it just uses a state-level dummy mapping.
    """
    return get_dummy_weather(state)


# ===========================================================
# RESCUER BRIEFING
# ===========================================================

def build_rescuer_brief(sos_df: pd.DataFrame, extra: str) -> str:
    region = st.session_state.get("region", "Unknown")
    extreme = sos_df[sos_df["risk_level"] == "Extreme"]
    high = sos_df[sos_df["risk_level"].str.contains("High")]

    lines = []
    lines.append("PraanSetu – Rescuer Operational Brief\n")
    lines.append(f"Region: {region}\n")
    lines.append("Priority Zones:\n")

    if not extreme.empty:
        lines.append("1. EXTREME RISK ZONES (Flooded + High Severity):")
        for _, r in extreme.iterrows():
            lines.append(f"   - {r['location_name']} | {r['message']}")
    else:
        lines.append("1. No zones marked as Extreme currently.")

    if not high.empty:
        lines.append("\n2. HIGH RISK ZONES:")
        for _, r in high.iterrows():
            if r["risk_level"] != "Extreme":
                lines.append(f"   - {r['location_name']} | {r['message']}")
    else:
        lines.append("\n2. No additional high-risk zones beyond extreme.")

    safer = sos_df[~sos_df["in_flood_zone"]]
    if not safer.empty:
        lines.append("\n3. Recommended Safer Staging Areas (outside flood zone):")
        for _, r in safer.iterrows():
            lines.append(f"   - {r['location_name']}")
    else:
        lines.append("\n3. No clearly safe staging areas detected outside flood polygon.")

    if extra.strip():
        lines.append("\n4. Additional Instructions from Control Center:")
        lines.append(f"   {extra.strip()}")

    return "\n".join(lines)


# ===========================================================
# ADMIN VIEW
# ===========================================================

def admin_view():
    st.subheader("Control Center – Admin Panel")

    # Choose region (state)
    region = st.selectbox("Select flooded region to monitor:", list(REGION_FILES.keys()))
    st.session_state["region"] = region

    # Load baseline historical data
    base_df = load_region_sos(region)

    # Live DB data for this region
    live_df = db_get_live_sos()
    if not live_df.empty:
        live_df_region = live_df[live_df["state"] == region].copy()
    else:
        live_df_region = pd.DataFrame(columns=[
            "id", "state", "city", "location_name", "lat", "lon",
            "severity", "message", "channel", "created_at"
        ])

    # Combine historical + live
    if not live_df_region.empty:
        # mark live and adjust ids so they do not collide visually
        live_df_region["source"] = "Live SOS"
        live_df_region["id"] = live_df_region["id"] + 10000
        combined = pd.concat([
            base_df[["id", "message", "location_name", "lat", "lon", "severity", "source", "state", "channel"]],
            live_df_region[["id", "message", "location_name", "lat", "lon", "severity", "source", "state", "channel"]],
        ], ignore_index=True)
    else:
        combined = base_df.copy()

    combined["severity_norm"] = combined["severity"].str.lower().map(
        {"high": 3, "medium": 2, "low": 1}
    ).fillna(1)
    combined["priority_score"] = combined["severity_norm"]

    center_lat = combined["lat"].mean()
    center_lon = combined["lon"].mean()
    flood_polygon = create_flood_polygon(center_lat, center_lon)
    sos_df = classify_risk(combined, flood_polygon)

    stats = simple_stats(sos_df)

    left, right = st.columns([2.4, 1])

    # ---------- RIGHT: stats, tables, broadcasts ----------
    with right:
        st.markdown("### Overview (by Region)")
        st.metric("Total SOS", stats["total"])
        st.metric("Extreme Risk Points", stats["extreme_risk"])
        st.metric("High Risk Points", stats["high_risk"])
        st.metric("Live SOS (DB)", len(live_df_region))

        st.markdown("---")
        st.markdown("**High Severity Locations (Quick Jump)**")
        high_only = sos_df[sos_df["severity"].str.lower() == "high"]
        high_locations = ["All locations"] + sorted(high_only["location_name"].unique().tolist())
        selected_location = st.selectbox("Jump to:", high_locations, key="admin_location_jump")

        st.markdown("---")
        st.markdown("**All SOS (Combined Historical + Live)**")
        st.dataframe(
            sos_df[
                [
                    "location_name",
                    "severity",
                    "risk_level",
                    "zone",
                    "state",
                    "channel",
                    "source",
                    "message",
                ]
            ],
            use_container_width=True,
            height=220,
        )

        st.markdown("---")
        st.markdown("**Geo-targeted Advisory by Zone**")
        zone_choice = st.selectbox("Select zone:", ["Red", "Yellow", "Green"])
        advisory_text = st.text_area(
            "Advisory text:",
            placeholder=(
                "Red example: Water rising rapidly. Move to higher floors immediately and avoid low-lying streets.\n"
                "Yellow example: Stay alert and keep documents and medicines ready.\n"
                "Green example: You are in a relatively safer zone, but stay informed."
            ),
        )
        if st.button("Preview Affected Locations"):
            affected = sos_df[sos_df["zone"] == zone_choice]
            st.write(f"{len(affected)} SOS locations currently in {zone_choice} zone.")
            st.dataframe(
                affected[["location_name", "severity", "risk_level", "source"]],
                use_container_width=True,
                height=150,
            )

        if st.button("Set Advisory for Zone"):
            if "zone_advisories" not in st.session_state:
                st.session_state["zone_advisories"] = {}
            st.session_state["zone_advisories"][zone_choice] = advisory_text
            st.success(f"Advisory set for {zone_choice} zone.")

        st.markdown("---")
        st.markdown("**Broadcast Message (Global)**")
        broadcast = st.text_area(
            "Message shown on all citizen pages:",
            value=st.session_state.get("broadcast_message", ""),
            placeholder="Example: Stay calm. We have received your SOS. Rescue teams are being dispatched.",
        )
        if st.button("Update Broadcast Message"):
            st.session_state["broadcast_message"] = broadcast
            st.success("Broadcast message updated.")

    # ---------- LEFT: tabs ----------
    with left:
        tab_map, tab_brief, tab_weather, tab_analytics, tab_db = st.tabs(
            [
                "🗺️ Map Intelligence",
                "📋 Rescuer Briefing",
                "☁️ Weather & Forecast",
                "📊 Alerts Analytics",
                "💾 Database Manager",
            ]
        )

        with tab_map:
            st.markdown("#### Flood Impact & SOS Map")
            fmap = create_map(sos_df, flood_polygon, selected_location)
            st_folium(fmap, width=950, height=550)

            st.info(
                "**Map legend (simplified):**\n"
                "- **Blue shaded area**: Estimated flood-affected zone.\n"
                "- **Red dots**: Extreme danger – flooded + high-severity SOS.\n"
                "- **Orange dots**: High risk – serious SOS in or near flood.\n"
                "- **Yellow dots**: Medium risk – water present, situation tense but not critical.\n"
                "- **Green dots**: Relatively safer / staging areas outside main flood.\n\n"
                "Use the layer control on the map to switch between **Satellite** and **Gray Base** view."
            )

        with tab_brief:
            st.markdown("#### Rescuer Briefing")
            extra = st.text_area(
                "Additional instructions for rescuers:",
                placeholder="Example: Avoid main bridge, water above 1.5m. Use service road from West Delhi towards city center.",
                height=120,
            )
            if st.button("Generate Brief"):
                brief = build_rescuer_brief(sos_df, extra)
                st.text_area(
                    "Operational Brief (copy/share with teams):",
                    value=brief,
                    height=280,
                )

        with tab_weather:
            st.markdown("#### Weather & Flood Risk Outlook (Region-Level)")
            w = get_weather_for_location(center_lat, center_lon, region)
            st.metric("Temperature", w["temp"])
            st.write("Rainfall forecast:", w["rain"])
            st.write("Flood risk summary:", w["risk"])
            st.info(
                "In deployment, this panel calls IMD / national weather APIs using district-level or (lat, lon) "
                "to keep the control room updated every few minutes."
            )

        with tab_analytics:
            st.markdown("#### Alerts Analytics – State / City / Channel Comparison")

            if "state" in sos_df.columns and "channel" in sos_df.columns:
                # State-wise by channel
                state_counts = sos_df.groupby(["state", "channel"])["id"].count().reset_index(name="count")
                st.markdown("**State-wise alerts by channel:**")
                st.dataframe(state_counts, use_container_width=True, height=200)

                # City/location-wise by channel – treat location_name as city-group for now
                city_col = "location_name"
                city_counts = sos_df.groupby([city_col, "channel"])["id"].count().reset_index(name="count")
                st.markdown("**Top locations by alerts (all channels):**")
                top_city_counts = (
                    city_counts.groupby(city_col)["count"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.bar_chart(top_city_counts)

                st.markdown(f"**Detailed location × channel breakdown for {region}:**")
                st.dataframe(city_counts, use_container_width=True, height=220)
            else:
                st.write("State/channel columns missing – add them to CSV for full analytics.")

        # ---------- DB MANAGER ----------
        with tab_db:
            st.markdown("#### Live SOS – Offline DB (SQLite)")
            live_all = db_get_live_sos()
            if not live_all.empty:
                st.dataframe(live_all, use_container_width=True, height=250)
            else:
                st.write("No live SOS stored in offline DB yet.")

            st.markdown("**Edit or Delete Live SOS**")
            col_id, col_sev, col_btns = st.columns([1, 1, 2])
            with col_id:
                edit_id = st.number_input("Record ID", min_value=1, step=1, value=1)
            with col_sev:
                new_sev = st.selectbox("New severity", ["high", "medium", "low"])
            with col_btns:
                if st.button("Update severity"):
                    db_update_live_sos_severity(int(edit_id), new_sev)
                    st.success("Severity updated in offline DB.")
                if st.button("Delete record"):
                    db_delete_live_sos(int(edit_id))
                    st.warning("Record deleted from offline DB.")

            st.markdown("---")
            st.markdown("#### Plain Messages – Offline DB (SMS / No GPS)")
            plain_df = db_get_plain_messages()
            if not plain_df.empty:
                st.dataframe(plain_df, use_container_width=True, height=200)
            else:
                st.write("No plain messages stored in offline DB yet.")

            del_id = st.number_input("Plain message ID to delete", min_value=1, step=1, value=1, key="plain_del_id")
            if st.button("Delete plain message"):
                db_delete_plain_message(int(del_id))
                st.warning("Plain message deleted from offline DB.")

            st.markdown("---")
            st.info(
                "Current DB mode: Offline SQLite at data/praansetu.db.\n"
                "In online mode, these functions can be redirected to any cloud DB (PostgreSQL, MySQL, Firebase, etc.)."
            )
            st.markdown("---")
            st.markdown("#### SOS Images Stored (offline DB)")
            images_df = db_get_sos_images()
            if not images_df.empty:
                st.dataframe(images_df, use_container_width=True, height=200)
                img_ids = images_df["id"].tolist()
                sel_img = st.selectbox("Select image record to preview:", img_ids) if img_ids else None
                if sel_img:
                    row = images_df[images_df["id"] == sel_img].iloc[0]
                    # Show the saved image file
                    try:
                        st.image(row["filename"], width='stretch')
                        st.markdown(f"**Live SOS ID:** {row['live_sos_id']}  \n**File:** {Path(row['filename']).name}")
                    except Exception as e:
                        st.warning(f"Unable to display image: {e}")
                    # Controls: replace or delete this image
                    col_rep, col_del = st.columns([2, 1])
                    with col_rep:
                        replace_file = st.file_uploader("Replace this image:", type=["jpg", "jpeg", "png"], key=f"replace_{sel_img}")
                        if st.button("Replace Image", key=f"replace_btn_{sel_img}"):
                            if replace_file is None:
                                st.warning("Choose a file to replace the image first.")
                            else:
                                try:
                                    # remove old file if exists
                                    try:
                                        if Path(row["filename"]).exists():
                                            os.remove(row["filename"]) 
                                    except Exception:
                                        pass
                                    new_path = save_image_file(replace_file, row["live_sos_id"])
                                    db_update_sos_image(sel_img, new_path)
                                    st.success("Image replaced successfully.")
                                except Exception as e:
                                    st.warning(f"Failed to replace image: {e}")
                    with col_del:
                        if st.button("Delete Image", key=f"del_img_{sel_img}"):
                            try:
                                # remove file from disk
                                try:
                                    if Path(row["filename"]).exists():
                                        os.remove(row["filename"]) 
                                except Exception:
                                    pass
                                db_delete_sos_image(sel_img)
                                st.warning("Image record deleted.")
                            except Exception as e:
                                st.warning(f"Failed to delete image record: {e}")
            else:
                st.write("No SOS images stored in DB yet.")

            st.markdown("---")
            st.markdown("#### Add or Attach Image to Live SOS")
            with st.form("add_image_form"):
                attach_live_id = st.number_input("Live SOS ID to attach image:", min_value=1, step=1, value=1)
                attach_file = st.file_uploader("Image to attach:", type=["jpg", "jpeg", "png"], key="attach_new_img")
                add_img_sub = st.form_submit_button("Add Image")
            if add_img_sub:
                if attach_file is None:
                    st.warning("Please choose an image to upload.")
                else:
                    try:
                        newp = save_image_file(attach_file, attach_live_id)
                        db_insert_sos_image(attach_live_id, newp)
                        st.success("Image saved and linked to live SOS id.")
                    except Exception as e:
                        st.warning(f"Failed to save image: {e}")

            st.markdown("---")
            st.markdown("#### Add Live SOS (Admin)")
            with st.form("admin_add_sos"):
                add_state = st.selectbox("State:", list(CITY_METADATA.keys()))
                add_city = st.text_input("City:", value="")
                add_location = st.text_input("Location name:")
                a_lat = st.number_input("Latitude:", format="%.6f", value=0.0)
                a_lon = st.number_input("Longitude:", format="%.6f", value=0.0)
                a_sev = st.selectbox("Severity:", ["high", "medium", "low"])
                a_msg = st.text_area("Message:")
                attach_image_checkbox = st.checkbox("Attach image?", key="admin_attach_checkbox")
                a_img = None
                if attach_image_checkbox:
                    a_img = st.file_uploader("Image (optional):", type=["jpg", "jpeg", "png"], key="admin_add_img")
                add_sub = st.form_submit_button("Add Live SOS")
            if add_sub:
                try:
                    new_live_id = db_insert_live_sos({
                        "state": add_state,
                        "city": add_city,
                        "location_name": add_location,
                        "lat": a_lat,
                        "lon": a_lon,
                        "severity": a_sev,
                        "message": a_msg,
                        "channel": "Admin",
                    })
                    if a_img is not None:
                        p = save_image_file(a_img, new_live_id)
                        db_insert_sos_image(new_live_id, p)
                    st.success(f"Live SOS added with id {new_live_id}")
                except Exception as e:
                    st.warning(f"Failed to add live SOS: {e}")


# ===========================================================
# CITIZEN VIEW
# ===========================================================

def citizen_view():
    st.subheader(T("citizen_portal", "Citizen Portal – Request Help"))

    broadcast = st.session_state.get("broadcast_message", "")
    if broadcast.strip():
        st.info(f"Control Center Message: {broadcast.strip()}")

    # Zone-level advisories (simple global display for now)
    zone_advisories = st.session_state.get("zone_advisories", {})
    if zone_advisories:
        for zone, msg in zone_advisories.items():
            if msg.strip():
                st.warning(f"[{zone} Zone Advisory] {msg}")

    tab_sos, tab_plain, tab_info = st.tabs(
        [f"🚨 {T('send_sos', 'Send SOS with Location')}",
         f"✉️ {T('send_plain', 'Send Simple Message')}",
         f"ℹ️ {T('about_contact', 'About & Contact')}"]
    )

    # ---------- SOS with location ----------
    with tab_sos:
        state = st.selectbox("Select your state/region:", list(CITY_METADATA.keys()))
        st.session_state["region"] = state

        cities = list(CITY_METADATA[state].keys())
        city = st.selectbox("Select your city:", cities)
        city_info = CITY_METADATA[state][city]

        # show local weather for this city (using placeholder function)
        w = get_weather_for_location(city_info["lat"], city_info["lon"], state)
        st.markdown("**Local weather snapshot for your city (demo):**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Temp", w["temp"])
        with c2:
            st.write("Rain:", w["rain"])
        with c3:
            st.write("Risk:", w["risk"])

        if "region_sos" not in st.session_state:
            st.session_state["region_sos"] = {}
        region_sos_dict = st.session_state["region_sos"]

        if state in region_sos_dict:
            region_df = region_sos_dict[state]
        else:
            base_df = load_region_sos(state)
            region_df = base_df.copy()
            region_sos_dict[state] = region_df
            st.session_state["region_sos"] = region_sos_dict

        with st.form("citizen_sos_form"):
            location_name = st.text_input(
                "Exact location (street, landmark, building):",
                value=city,
            )
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat = st.number_input(
                    "Latitude (approx):",
                    value=float(city_info["lat"]),
                    format="%.6f",
                )
            with col_lon:
                lon = st.number_input(
                    "Longitude (approx):",
                    value=float(city_info["lon"]),
                    format="%.6f",
                )

            severity = st.selectbox("How urgent is it?", ["high", "medium", "low"])
            contact = st.text_input("Your name / phone (optional but helpful):", "")
            message = st.text_area(
                "Describe your situation:",
                placeholder="Example: Water at chest level and 4 people trapped on roof.",
            )

            image_file = st.file_uploader(
                "Attach a photo (optional):",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of your surroundings or damage if possible.",
            )

            submitted = st.form_submit_button("Send SOS")

        if submitted:
            if not location_name or not message:
                st.warning("Please fill at least location and message.")
            else:
                current_df = region_sos_dict[state]
                new_id = int(current_df["id"].max()) + 1 if not current_df.empty else 1
                sev_map = {"high": 3, "medium": 2, "low": 1}

                full_message = message
                if contact.strip():
                    full_message = f"{message} (Contact: {contact.strip()})"

                new_row = {
                    "id": new_id,
                    "message": full_message,
                    "location_name": location_name,
                    "lat": float(lat),
                    "lon": float(lon),
                    "severity": severity,
                    "severity_norm": sev_map.get(severity, 1),
                    "priority_score": sev_map.get(severity, 1),
                    "source": "Live SOS",
                    "state": state,
                    "channel": "Web",
                }

                # 1) Update in-memory DF (for this session)
                updated_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
                region_sos_dict[state] = updated_df
                st.session_state["region_sos"] = region_sos_dict

                # 2) Store in offline DB and get inserted live_sos id
                live_id = db_insert_live_sos({
                    "state": state,
                    "city": city,
                    "location_name": location_name,
                    "lat": lat,
                    "lon": lon,
                    "severity": severity,
                    "message": full_message,
                    "channel": "Web",
                })

                # 2b) Save uploaded image (if any) and record in DB
                if image_file is not None:
                    try:
                        saved_path = save_image_file(image_file, live_id)
                        db_insert_sos_image(live_id, saved_path)
                    except Exception as e:
                        st.warning(f"Failed to save uploaded image: {e}")

                # 3) Email notification
                res = send_new_sos_email(pd.Series(new_row))
                if EMAIL_ENABLED and "YOUR_APP_PASSWORD_HERE" not in SENDER_PASSWORD:
                    if res == "OK":
                        st.success("SOS sent, stored in offline DB, and email notification triggered to control center.")
                    else:
                        st.warning(f"SOS saved and stored in DB, but email issue: {res}")
                else:
                    st.success("SOS sent and stored in offline DB. (Email is not configured properly yet.)")

    # ---------- Simple message ----------
    with tab_plain:
        st.markdown(
            "If you cannot share your location or use full SOS, send a simple message.\n"
            "Control center will see it in the offline DB inbox and may also receive an email."
        )
        with st.form("plain_msg_form"):
            contact_plain = st.text_input("Your name / phone (optional but recommended):", "")
            msg_plain = st.text_area("Your message:", placeholder="Example: I am in trouble due to flood, please call me.")
            submitted_plain = st.form_submit_button("Send Message")

        if submitted_plain:
            if not msg_plain.strip():
                st.warning("Please write a message.")
            else:
                # Store in offline DB
                db_insert_plain_message(contact_plain, msg_plain)

                # Try email
                res = send_plain_message_email(contact_plain, msg_plain)
                if EMAIL_ENABLED and "YOUR_APP_PASSWORD_HERE" not in SENDER_PASSWORD:
                    if res == "OK":
                        st.success("Message stored in offline DB and emailed to control center.")
                    else:
                        st.warning(f"Message stored in DB, but email issue: {res}")
                else:
                    st.success("Message stored in offline DB. (Email is not configured properly yet.)")

    # ---------- About & Contact ----------
    with tab_info:
        st.markdown("### About PraanSetu")
        st.write(
            "PraanSetu is an AI-powered disaster intelligence assistant focused on last-mile impact. "
            "It combines flood mapping, live SOS messages, weather outlook and early warnings "
            "into one operational dashboard for citizens and control rooms."
        )

        st.markdown("### Contact & Helpline")
        st.write("Example Helpline: **108 / 112**")
        st.write("Control Center Email: **singhnitant28@gmail.com**")
        st.write("Control Center Phone: **+91-7397867643, +91-9359210080")
        st.info(
            "In real deployment, this would show official State Disaster Management Authority contact details "
            "and verified helpline numbers."
        )


# ===========================================================
# MAIN
# ===========================================================

def main():
    st.set_page_config(page_title="PraanSetu – Multi-Role v3", layout="wide")

    # Sidebar logo + language/role
    with st.sidebar:
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, use_container_width=True)
        except Exception:
            st.markdown("### PraanSetu")
        st.markdown("---")

    col_lang, col_role = st.sidebar.columns([1, 1])
    with col_lang:
        lang_label = st.selectbox("Language:", list(LANG_OPTIONS.keys()))
    lang = LANG_OPTIONS[lang_label]
    st.session_state["lang"] = lang

    with col_role:
        role = st.selectbox("Login as:", ["Citizen", "Admin"])

    if "broadcast_message" not in st.session_state:
        st.session_state["broadcast_message"] = ""

    # New header with logo
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    # st.caption(T("title", "PraanSetu – The Bridge of Life"))  # optional subtitle

    if role == "Admin":
        st.sidebar.markdown("### Admin Login")

        # initialize login state if missing
        if "admin_logged_in" not in st.session_state:
            st.session_state["admin_logged_in"] = False

        # Show login form when not logged in
        if not st.session_state.get("admin_logged_in", False):
            username = st.sidebar.text_input("Username:", key="admin_username")
            password = st.sidebar.text_input("Password:", type="password", key="admin_password")

            if st.sidebar.button("Login"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["admin_user"] = username
                    st.sidebar.success("Logged in as admin.")
                else:
                    st.sidebar.error("Invalid username or password.")

        else:
            st.sidebar.markdown(f"Logged in as: {st.session_state.get('admin_user', ADMIN_USERNAME)}")
            if st.sidebar.button("Logout"):
                st.session_state["admin_logged_in"] = False
                if "admin_user" in st.session_state:
                    del st.session_state["admin_user"]

            # When logged in, show admin view
            admin_view()
    else:
        citizen_view()


if __name__ == "__main__":
    main()
