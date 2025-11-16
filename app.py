import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import base64
import plotly.graph_objects as go
# Apply Custom Page Style (Styling)
st.markdown("""
    <style>
        /* Smooth rounded UI */
        .stButton>button, .stSelectbox, .stNumberInput>div>input {
            border-radius: 8px;
        }

        /* Improve metric cards */
        .stMetric label {
            font-size: 16px;
            font-weight: 600;
        }

        /* Center Page Title */
        h1 {
            text-align: center;
            font-weight: 700;
            padding-bottom: 10px;
        }

        /* Tabs style */
        .stTabs [data-baseweb="tab"] {
            padding: 12px 20px;
            font-size: 16px;
        }

        /* Increase spacing */
        .block-container {
            padding-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Load local image and encode to Base64
with open("/Users/shaktiswry/PycharmProjects/fyp 4/kl.jpg", "rb") as f:
    img_bytes = f.read()
b64_img = base64.b64encode(img_bytes).decode()

# Set as background
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{b64_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-color: transparent;
}}

/* Add a semi-transparent white box behind content */
.block-container {{
    background-color: rgba(255, 255, 255, 0.2);  /* adjust opacity for readability */
    border-radius: 10px;
    padding: 2rem;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
h1, h2, h3, h4, h5, h6, p, label {{
    color: white;  /* make text white */
    text-shadow: 1px 1px 4px black;
}}
</style>
""", unsafe_allow_html=True)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="KL Property Price Predictor", page_icon="🏙️", layout="wide")

st.title("🏙️ Kuala Lumpur Condominium Price Intelligence Dashboard")
st.write("Estimate fair market pricing and assess investment viability.")

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
import joblib
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=19rg-26e8KyE6iRWLUlYpYpA-AmUrYmq6"
COLUMNS_URL = "https://drive.google.com/uc?id=1WcnB5zgHYkec4znoSxRsBiJ52ldw_e9F"

MODEL_PATH = "best_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Download model columns if not exists
if not os.path.exists(COLUMNS_PATH):
    gdown.download(COLUMNS_URL, COLUMNS_PATH, quiet=False)

# Load the files
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Load dataset for analytics visuals
data = pd.read_csv("train_preprocessed_unscaled.csv")
# Extract location name from one-hot encoded location columns
location_cols = [c for c in data.columns if c.startswith("location_clean_")]
data["location_clean"] = data[location_cols].idxmax(axis=1).str.replace("location_clean_", "")
df_vis = data.copy()

# ==================================================
# CREATE PRICE COLUMN ONCE (USED IN ALL VISUALS)
# ==================================================
if data["Target"].max() < 20:
    data["price"] = np.expm1(data["Target"])
    df_vis["price"] = np.expm1(df_vis["Target"])
else:
    data["price"] = data["Target"]
    df_vis["price"] = df_vis["Target"]

# ========================================
# ADD COORDINATES FOR MAP VISUALIZATION
# ========================================
location_coordinates = {k.upper(): v for k, v in {
    "KLCC": (3.1573, 101.7122),
    "Bukit Bintang": (3.1468, 101.7113),
    "Bukit Jalil": (3.0587, 101.6917),
    "Mont Kiara": (3.1685, 101.6512),
    "Desa ParkCity": (3.1862, 101.6299),
    "Damansara Heights": (3.1491, 101.6534),
    "Bangsar": (3.1290, 101.6798),
    "Bangsar South": (3.1107, 101.6664),
    "Ampang": (3.1665, 101.7483),
    "Ampang Hilir": (3.1540, 101.7446),
    "Cheras": (3.1068, 101.7229),
    "Setapak": (3.2088, 101.7278),
    "Wangsa Maju": (3.2038, 101.7367),
    "Segambut": (3.1917, 101.6734),
    "Sentul": (3.1823, 101.6888),
    "Jalan Kuching": (3.1922, 101.6721),
    "Jalan Ipoh": (3.1752, 101.6866),
    "Jalan Sultan Ismail": (3.1552, 101.7054),
    "Pantai": (3.1197, 101.6669),
    "Taman Desa": (3.1030, 101.6845),
    "Sri Petaling": (3.0684, 101.6856),
    "Bukit Tunku (Kenny Hills)": (3.1665, 101.6828),
    "Sri Hartamas": (3.1600, 101.6520),
    "Taman Tun Dr Ismail": (3.1461, 101.6255),
    "Brickfields": (3.1292, 101.6861),
    "KL Sentral": (3.1342, 101.6861),
    "KL City": (3.1499, 101.6945),
    "City Centre": (3.1517, 101.6942),
    "Seputeh": (3.1150, 101.6797),
    "Pandan Perdana": (3.1172, 101.7421),
    "Taman Melawati": (3.2126, 101.7471),
    "Titiwangsa": (3.1774, 101.7077),
    "Kepong": (3.2140, 101.6356),
    "Bandar Menjalara": (3.1939, 101.6309),
    "Kuchai Lama": (3.0839, 101.6883),
    "Jinjang": (3.2091, 101.6560),
    "Bandar Tasik Selatan": (3.0720, 101.7096),
    "Salak Selatan": (3.1049, 101.7055),
    "Sungai Besi": (3.0574, 101.7179),
    "Keramat": (3.1689, 101.7277),
    "Country Heights Damansara": (3.1784, 101.6211)
}.items()}

# Convert 1-hot encoded location columns back to actual names
data["location_clean"] = data[location_cols].idxmax(axis=1).str.replace("location_clean_", "")

# Map coordinates
data["lat"] = data["location_clean"].map(lambda x: location_coordinates.get(x, (None, None))[0])
data["lon"] = data["location_clean"].map(lambda x: location_coordinates.get(x, (None, None))[1])

# Remove any rows where coordinates missing
data = data.dropna(subset=["lat", "lon"])

# ==============================
# INPUT FORM (LEFT COLUMN)
# ==============================
st.markdown("### 🏠 Enter Property Details")
st.markdown("Use the form below to describe the property you're evaluating.")
st.markdown("---")

with st.container():
    col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Property Info")
    st.write("Fill in the specifications:")
    size = st.number_input("Size (sqft)", min_value=300, max_value=10000, value=1000)
    rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
    baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    carparks = st.number_input("Carparks", min_value=0, max_value=5, value=1)

    location = st.selectbox("Location", sorted(
        [c.replace("location_clean_", "") for c in model_columns if c.startswith("location_clean_")]))

    property_base = st.selectbox("Property Base", ["CONDOMINIUM", "SERVICED RESIDENCE"])
    property_subtype = st.selectbox("Property Subtype",
                                    ["STANDARD", "INTERMEDIATE", "DUPLEX", "PENTHOUSE", "STUDIO", "CORNER"])
    furnishing = st.selectbox("Furnishing", ["FULLY FURNISHED", "PARTLY FURNISHED", "UNFURNISHED"])

    actual_price = st.number_input("Actual Market Price (RM)", min_value=10000, step=10000)

# ==============================
# PROCESS INPUT INTO MODEL FORMAT & PREDICT
# ==============================
input_dict = {
    "size_sqft": size,
    "rooms_num": rooms,
    "bathrooms_num": baths,
    "carparks_num": carparks,
    f"location_clean_{location}": 1,
    f"property_base_{property_base}": 1,
    f"property_subtype_{property_subtype}": 1,
    f"furnishing_clean_{furnishing}": 1,
}

input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0
for key in input_dict:
    if key in input_df.columns:
        input_df.loc[0, key] = input_dict[key]

predicted_value = model.predict(input_df)[0]

# ==============================
# DYNAMIC FILTERING FOR SIMILAR PROPERTIES (NEW)
# ==============================
# Filter data to find properties similar to user input (e.g., same location, size ±20%, rooms ±1, etc.)
size_tolerance = 0.2  # ±20% size
room_tolerance = 1    # ±1 room
filtered_similar_properties = data[
    (data["location_clean"] == location) &
    (data["size_sqft"].between(size * (1 - size_tolerance), size * (1 + size_tolerance))) &
    (data["rooms_num"].between(max(1, rooms - room_tolerance), rooms + room_tolerance)) &
    (data["bathrooms_num"].between(max(1, baths - room_tolerance), baths + room_tolerance)) &
    (data["carparks_num"] == carparks) &  # Exact match for carparks
    (data[f"property_base_{property_base}"] == 1) &
    (data[f"property_subtype_{property_subtype}"] == 1) &
    (data[f"furnishing_clean_{furnishing}"] == 1)
].copy()

# If no exact matches, fall back to location-only filtering
if filtered_similar_properties.empty:
    filtered_similar_properties = data[data["location_clean"] == location].copy()
    st.warning("No highly similar properties found. Showing all in selected location for comparison.")

# Add user's price per sqft for comparison
user_price_per_sqft = actual_price / size if size > 0 else 0
filtered_similar_properties["price_per_sqft"] = filtered_similar_properties["price"] / filtered_similar_properties["size_sqft"]

# ==============================
# RESULTS (RIGHT COLUMN) WITH SUMMARY INSIGHTS (NEW)
# ==============================
with col2:
    st.markdown("#### Price Evaluation Result")

    predicted_price = float(predicted_value)
    st.metric("Predicted Price (RM)", f"{predicted_price:,.2f}")
    st.metric("Actual Price (RM)", f"{actual_price:,.2f}")

    if predicted_price < actual_price:
        st.success("✅ **Good Deal!** This property is *under-valued* relative to similar units.")
    else:
        st.error("❗ **Not Recommended.** This property is *over-valued* compared to market trends.")

    # NEW: Summary Insights based on filtered data
    st.markdown("---")
    st.markdown("#### 📊 Quick Insights from Similar Properties")
    if not filtered_similar_properties.empty:
        avg_similar_price = filtered_similar_properties["price"].mean()
        avg_similar_pps = filtered_similar_properties["price_per_sqft"].mean()
        count_similar = len(filtered_similar_properties)
        st.write(f"- **Similar Properties Found**: {count_similar}")
        st.write(f"- **Avg Price of Similar Units**: RM {avg_similar_price:,.2f}")
        st.write(f"- **Avg Price per Sqft**: RM {avg_similar_pps:,.2f}")
        st.write(f"- **Your Price per Sqft**: RM {user_price_per_sqft:,.2f} ({'Below Avg' if user_price_per_sqft < avg_similar_pps else 'Above Avg'})")
    else:
        st.write("No similar properties to compare.")

# ==============================
# VISUAL ANALYTICS TABS (UPDATED FOR DYNAMIC FILTERING)
# ==============================
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Location Pricing",
    "🏗️ Size vs Price",
    "📊 Feature Importance",
    "📈 Market Distribution",
    "🗺️ Kuala Lumpur Property Map",
    "💰 Price per Sqft Analysis"
])

px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Tealrose"

# Tab 1: Location Price Comparison (UPDATED: Show all locations, highlight selected)
with tab1:
    # Compute mean price by location
    loc_cols = [col for col in df_vis.columns if col.startswith("location_clean_")]
    location_prices = []
    for col in loc_cols:
        loc_name = col.replace("location_clean_", "")
        avg_price = df_vis.loc[df_vis[col] == 1, "price"].mean()
        location_prices.append([loc_name, avg_price])
    loc_df = pd.DataFrame(location_prices, columns=["Location", "Average Price (RM)"]).dropna()
    loc_df = loc_df.sort_values("Average Price (RM)", ascending=False)  # Sort descending
    loc_df["Selected"] = loc_df["Location"].apply(lambda x: "Selected" if x == location else "Other")

    fig = px.bar(loc_df, x="Location", y="Average Price (RM)",  # Removed .head(20) to show all
                 color="Selected",
                 color_discrete_map={"Selected": "red", "Other": "gray"},
                 title=f"All Locations by Average Price (You Selected: {location})",
                 text_auto=".2s")
    fig.update_layout(xaxis_tickangle=45)  # Rotate labels for readability
    # Add horizontal line for user's predicted price
    fig.add_hline(y=predicted_price, line_dash="dash", line_color="blue", annotation_text="Your Predicted Price")
    st.plotly_chart(fig, use_container_width=True)


# Tab 2: Size vs Price Chart (UPDATED: Use filtered similar properties, add user point)
with tab2:
    fig = px.scatter(filtered_similar_properties, x="size_sqft", y="price", opacity=0.6,
                     trendline="ols",
                     title=f"Size vs Price for Similar Properties ({location})",
                     labels={"size_sqft": "Size (sqft)", "price": "Price (RM)"})
    # NEW: Add user's input as a highlighted point
    fig.add_trace(go.Scatter(x=[size], y=[predicted_price], mode="markers",
                             marker=dict(size=12, color="red", symbol="star"),
                             name="Your Property"))
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Feature Importance Chart (UNCHANGED, but added context)
with tab3:
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(model.feature_importances_, index=model_columns)\
                     .sort_values(ascending=False).head(15)
        fig = px.bar(feat_imp,
                     x=feat_imp.values,
                     y=feat_imp.index,
                     title="Top 15 Most Important Features (Influencing Your Prediction)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance not available for this model.")

# Tab 4: Price Distribution (UPDATED: Use filtered data, add user marker)
with tab4:
    st.subheader("Distribution of Property Prices for Similar Properties")

    # Filter based on similar properties
    min_price = max(50000, int(filtered_similar_properties["price"].quantile(0.01))) if not filtered_similar_properties.empty else 50000
    max_price = int(filtered_similar_properties["price"].quantile(0.99)) if not filtered_similar_properties.empty else 2000000

    price_range = st.slider(
        "Select Price Range (RM)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=50000
    )

    filtered_range = filtered_similar_properties[(filtered_similar_properties["price"] >= price_range[0]) &
                                                 (filtered_similar_properties["price"] <= price_range[1])]

    fig = px.histogram(
        filtered_range,
        x="price",
        nbins=30,
        title=f"Price Distribution for Similar Properties (RM {price_range[0]:,} - RM {price_range[1]:,})"
    )
    fig.update_layout(xaxis_title="Price (RM)", yaxis_title="Count", bargap=0.05)
    # NEW: Add vertical line for user's predicted price
    fig.add_vline(x=predicted_price, line_dash="dash", line_color="red", annotation_text="Your Predicted Price")
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Kuala Lumpur Property Map (UPDATED: Zoom to selected location, show similar properties)
with tab5:
    st.subheader(f"🗺️ Property Map for {location} (Similar Properties Highlighted)")

    # Use filtered similar properties for the map
    map_df = filtered_similar_properties.groupby("location_clean", as_index=False).agg(
        avg_price=("price", "mean"),
        lat=("lat", "first"),
        lon=("lon", "first")
    ).dropna(subset=["lat", "lon"])

    zoom_level = 14
    center_lat, center_lon = location_coordinates.get(location.upper(), (3.1390, 101.6869))

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="avg_price",
        color="avg_price",
        hover_name="location_clean",
        hover_data={"avg_price": ":,.0f"},
        zoom=zoom_level,
        height=600,
        mapbox_style="carto-positron"
    )
    fig.update_layout(mapbox_center={"lat": center_lat, "lon": center_lon})
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Price per Sqft Analysis (UPDATED: Use filtered data, add user marker)
with tab6:
    st.subheader(f"💰 Price per Sqft Analysis for Similar Properties in {location}")

    if filtered_similar_properties.empty:
        st.warning(f"No similar property records found for {location}.")
    else:
        fig = px.box(
            filtered_similar_properties,
            y="price_per_sqft",
            title=f"Price per Sqft Distribution for Similar Properties",
            labels={"price_per_sqft": "Price per Sqft (RM)"}
        )
        # NEW: Add horizontal line for user's price per sqft
        fig.add_hline(y=user_price_per_sqft, line_dash="dash", line_color="blue", annotation_text="Your Price per Sqft")
        st.plotly_chart(fig, use_container_width=True)
