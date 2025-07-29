import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import streamlit.components.v1 as components
import json
import pydeck as pdk


st.set_page_config(
    page_title="🛫 Flight price predictor",
    layout="centered"
)

def render_lottie_web(filepath: str, height=300, width=300):
    with open(filepath, "r") as f:
        lottie_json = f.read()
    
    components.html(f"""
        <lottie-player 
            autoplay 
            loop 
            mode="normal" 
            background="transparent" 
            speed="1" 
            src='data:application/json;base64,{base64.b64encode(lottie_json.encode()).decode()}'
            style="width: {width}px; height: {height}px; margin-bottom: 0px;">
        </lottie-player>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """, height=height, width=width)

lottie_flight_path = "animations/flight_anim.json"
lottie_success_path = "animations/success_anim.json"

def get_base64_of_local_image(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img_base64 = get_base64_of_local_image("flight_background.jpg")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .stButton>button {{
        background-color: steelblue;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: steelblue;
        transform: scale(1.05);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    render_lottie_web(lottie_flight_path, height=300, width=350)

st.markdown(
    "<h1 style='text-align: center; color: turquoise;'>🛩️ Flight price predictor</h1>",
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align: center;'>Estimate domestic Indian flight prices based on travel details 🧳</h4>", unsafe_allow_html=True)
st.markdown("---")

model = joblib.load("best_gradient_boosting_model.pkl")

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "selected_currency" not in st.session_state:
    st.session_state["selected_currency"] = "USD ($)"

with st.form("prediction_form"):
    st.subheader("📋 Enter flight details")
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            "✈️ Airline",
            [
                'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business','Multiple carriers', 'Multiple carriers Premium economy',
                'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy'
            ],
            help="Select the airline you will be flying with. Different airlines have different fare structures."
        )
        source = st.selectbox(
            "🌆 Source City",
            ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'],
            help="Select the city from where your journey starts."
        )
        destination = st.selectbox(
            "🌇 Destination City",
            ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata'],
            help="Select the city where you want to fly to."
        )
        travel_period = st.radio(
            "📅 Travel Period",
            ['Holiday Period', 'Non-Holiday Period'],
            help="Select whether your travel falls during a holiday season or not, as prices vary accordingly."
        )

    with col2:
        journey_date = st.date_input(
            "📆 Journey Date",
            help="Pick the date of your flight journey. Prices can vary depending on the date."
        )
        journey_day = journey_date.day
        stops = st.slider("🛑 Number of Stops",min_value=0,max_value=4,value=1,step=1,help="Select how many stops your flight makes.")


        st.markdown("⏱️ **Flight Duration** (hh:mm)")
        colh, colm = st.columns(2)
        with colh:
            hours = st.number_input(
                "Hours",
                min_value=0,max_value=20,value=2,
                step=1, help="Enter the hours part of your total flight duration."
            )
        with colm:
            minutes = st.number_input(
                "Minutes", min_value=0,max_value=59, value=30,step=1,
                help="Enter the minutes part of your total flight duration."
            )

        duration_mins = hours * 60 + minutes
        dep_time_bin = st.selectbox(
            "🕑 Departure Time",
            ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'],
            help="Select the time window during which your flight departs."
        )
        arr_time_bin = st.selectbox(
            "🕓 Arrival Time",
            ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'],
            help="Select the time window during which your flight arrives."
        )

    predict_btn = st.form_submit_button("📍 Predict Price")

if predict_btn:
    errors = False

    if duration_mins < 30:
        st.error("⚠️ Flight duration must be at least 30 minutes for a valid flight.")
        errors = True

    if source == destination:
        st.error("⚠️ Source and Destination cannot be the same.")
        errors = True

    if errors:
        #clear results and inputs if there are errors
        st.session_state["prediction_result"] = None
        if "last_inputs" in st.session_state:
            del st.session_state["last_inputs"]

    else:
        with st.spinner("🔍 Predicting flight fare..."):
            input_dict = {
                'Journey_Day': journey_day, 'Total_Stops': int(stops), 'Total_Duration_mins': duration_mins,
                f'Airline_{airline}': 1, f'Source_{source}': 1, f'Destination_{destination}': 1,
                f'Dep_Time_of_Day_{dep_time_bin}': 1, f'Arrival_Time_of_Day_{arr_time_bin}': 1,
                f'India_Travel_Period_{travel_period}': 1
            }

            expected_cols = list(model.feature_names_in_)
            input_df = pd.DataFrame([input_dict])
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_cols]

            prediction = model.predict(input_df)[0]
            st.session_state["prediction_result"] = round(prediction, 2)
            st.session_state["last_inputs"] = {
                "source": source, "destination": destination, "stops": stops, "duration_mins": duration_mins,
                "airline": airline, "journey_day": journey_day, "journey_date": journey_date,
                "dep_time_bin": dep_time_bin, "arr_time_bin": arr_time_bin, "travel_period": travel_period
            }

if st.session_state["prediction_result"] is not None and "last_inputs" in st.session_state:
    inputs = st.session_state["last_inputs"]



    st.success(f"💸 Estimated flight price: ₹ {st.session_state['prediction_result']:,}")
    
    #coordinates
    city_coords = {
        "Banglore": [12.9716, 77.5946],"Delhi": [28.6139, 77.2090], "Mumbai": [19.0760, 72.8777],"Chennai": [13.0827, 80.2707],"Kolkata": [22.5726, 88.3639],
        "Cochin": [9.9312, 76.2673], "Hyderabad": [17.3850, 78.4867]
    }

    src_lat, src_lon = city_coords[inputs["source"]]
    dst_lat, dst_lon = city_coords[inputs["destination"]]

    view_state = pdk.ViewState(
        latitude=(src_lat + dst_lat) / 2,
        longitude=(src_lon + dst_lon) / 2,
        zoom=4.5,
        pitch=0
    )

    label_data = pd.DataFrame({
        "name": ["Source: " + inputs["source"], "Destination: " + inputs["destination"]],
        "lat": [src_lat, dst_lat],
        "lon": [src_lon, dst_lon]
    })

    line_layer = pdk.Layer(
        "LineLayer",
        data=pd.DataFrame({
            'from_lat': [src_lat], 'from_lon': [src_lon],
            'to_lat': [dst_lat], 'to_lon': [dst_lon]
        }),
        get_source_position='[from_lon, from_lat]',
        get_target_position='[to_lon, to_lat]',
        get_color='[0, 200, 100]', get_width=4
    )
    layer_src = pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [src_lat], "lon": [src_lon]}),
                          get_position='[lon, lat]', get_color='[0, 128, 255]', get_radius=50000)
    layer_dst = pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [dst_lat], "lon": [dst_lon]}),
                          get_position='[lon, lat]', get_color='[255, 0, 128]', get_radius=50000)
    text_layer = pdk.Layer("TextLayer", data=label_data, get_position='[lon, lat]',
                           get_text='name', get_size=16, get_color=[0, 0, 0], get_angle=0,
                           get_alignment_baseline="'bottom'")

    st.markdown("### 🗺️ Flight route map")
    st.pydeck_chart(pdk.Deck(
        map_style="road",
        initial_view_state=view_state,
        layers=[layer_src, layer_dst, line_layer, text_layer]
    ))

    #currency conversion
    exchange_rates = {
        "USD ($)": 0.012, "EUR (€)": 0.011, "SGD (S$)": 0.016,
        "GBP (£)": 0.0095, "JPY (¥)": 1.8
    }

    #allow dynamic selection without triggering rerun conflict
    new_currency = st.selectbox(
        "Choose currency",
        list(exchange_rates.keys()),
        index=list(exchange_rates.keys()).index(st.session_state["selected_currency"])
    )
    if new_currency != st.session_state["selected_currency"]:
        st.session_state["selected_currency"] = new_currency

    symbols = {"USD ($)": "$", "EUR (€)": "€", "SGD (S$)": "S$", "GBP (£)": "£", "JPY (¥)": "¥"}
    converted_price = round(st.session_state['prediction_result'] * exchange_rates[st.session_state["selected_currency"]], 2)
    st.info(f"💱 Converted price: {symbols[st.session_state['selected_currency']]} {converted_price:,}")

    #tips based on feature importance
    st.markdown("### 💡 Tips to get a cheaper flight:")
    feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    top_features = feature_importances.sort_values(ascending=False).head(15)

    tips_shown = False

    if 'Total_Stops' in top_features.index and int(inputs["stops"]) > 1:
        st.markdown("- 🛑 Try to choose flights with fewer stops; direct flights can be cheaper overall.")
        tips_shown = True

    if 'Total_Duration_mins' in top_features.index and inputs["duration_mins"] > 150:
        st.markdown("- ⏱️ Your flight duration is quite long. Consider shorter flights if available to reduce costs.")
        tips_shown = True

    if 'Airline_Jet Airways' in top_features.index and inputs["airline"] == "Jet Airways":
        st.markdown("- 🛫 Jet Airways tends to be expensive. Consider budget airlines like SpiceJet or IndiGo.")
        tips_shown = True

    if 'Airline_Jet Airways Business' in top_features.index and inputs["airline"] == "Jet Airways Business":
        st.markdown("- 💼 You're flying Business class. Downgrading to Economy could save significant cost.")
        tips_shown = True

    if 'Airline_IndiGo' in top_features.index and inputs["airline"] != "IndiGo":
        st.markdown("- 💡 IndiGo offers budget-friendly fares. Consider switching to it for potential savings.")
        tips_shown = True

    if 'Airline_SpiceJet' in top_features.index and inputs["airline"] != "SpiceJet":
        st.markdown("- 🔥 SpiceJet is often economical. Explore their fares for cost-effective travel.")
        tips_shown = True

    if 'Journey_Day' in top_features.index and inputs["journey_day"] <= 7:
        st.markdown("- 📆 Flying mid- or late-month can sometimes offer better prices than early in the month.")
        tips_shown = True

    if 'Journey_Month' in top_features.index and inputs["journey_date"].month in [4, 5, 10, 12]:
        st.markdown("- 🏖️ Your selected month may be in peak season. Traveling off-season could lower costs.")
        tips_shown = True

    if 'Arrival_Hour' in top_features.index and inputs["arr_time_bin"] not in ['Night', 'Late Night']:
        st.markdown("- 🌙 Late-night arrivals are often cheaper than daytime ones.")
        tips_shown = True

    if 'Dep_Hour' in top_features.index and inputs["dep_time_bin"] not in ['Night', 'Late Night']:
        st.markdown("- 🌃 Late-night departures usually come at lower prices.")
        tips_shown = True

    if 'Destination_Cochin' in top_features.index and inputs["destination"] == "Cochin":
        st.markdown("- 📍 Flying to Cochin? Prices can vary—compare nearby destinations if possible.")
        tips_shown = True

    if 'Destination_Delhi' in top_features.index and inputs["destination"] == "Delhi":
        st.markdown("- 🏙️ Delhi fares can fluctuate—try different dates or times for better rates.")
        tips_shown = True

    if 'Destination_Hyderabad' in top_features.index and inputs["destination"] == "Hyderabad":
        st.markdown("- 🛬 Hyderabad flights might have hidden deals—search across multiple airlines.")
        tips_shown = True

    if 'Source_Delhi' in top_features.index and inputs["source"] == "Delhi":
        st.markdown("- 🏁 Delhi departures may vary in price—explore alternatives like Jaipur if nearby.")
        tips_shown = True

    if 'India_Travel_Period_Non-Holiday Period' in top_features.index and inputs["travel_period"] == 'Holiday Period':
        st.markdown("- 🎉 Holiday period fares are usually higher. If flexible, travel during non-holidays.")
        tips_shown = True

    if not tips_shown:
        st.markdown("🎉 Great! Your flight details look optimised for a good price.")


st.markdown("---")
with st.expander("📊 Show visual insights"):
    df = pd.read_csv("flight_price.csv")

    st.subheader("Price distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Price'], kde=True, ax=ax1, color='skyblue')
    ax1.set_title("Distribution of flight prices")
    st.pyplot(fig1)

    st.subheader("Price variation across airlines")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Airline', y='Price', data=df, ax=ax2)
    ax2.set_title("Price variation across airlines")
    ax2.tick_params(axis='x', rotation=90)
    st.pyplot(fig2)

    st.subheader("Price by number of stops")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Total_Stops', y='Price', data=df, ax=ax3)
    ax3.set_title("Price by number of stops")
    st.pyplot(fig3)

    st.subheader("Price vs Duration")
    def duration_to_minutes(duration):
        hours, minutes = 0, 0
        if 'h' in duration:
            parts = duration.split('h')
            hours = int(parts[0].strip())
            duration = parts[1]
        if 'm' in duration:
            minutes = int(duration.split('m')[0].strip())
        return hours * 60 + minutes

    df['Duration_temp'] = df['Duration'].apply(duration_to_minutes)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Duration_temp', y='Price', data=df, ax=ax4)
    ax4.set_title("Price vs Duration (Minutes)")
    ax4.set_xlabel("Duration (Minutes)")
    st.pyplot(fig4)
    df.drop('Duration_temp', axis=1, inplace=True)

st.markdown("---")
st.markdown("<center>🛫 FlightFarePredictor © 2025 ❤️</center>", unsafe_allow_html=True)
