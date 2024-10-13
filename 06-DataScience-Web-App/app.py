import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
st.title("Data Science Web #1")
st.markdown("## Motor Vehicle Collisions in New York City")
st.markdown("#### Made by Arjun Raghunandanan following the instructor using Python & Streamlit for Coursera Project Network")
# Modify This URL according to where you have stored your csv file or from where you are fetching the csv file
#option 1 : fetch code from online
#DATA_URL = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
#option 2 : download csv file and load it offline (I used this method)
DATA_URL = ("/home/rhyme/Desktop/Project/Motor_Vehicle_Collisions_-_Crashes.csv")
# Function to load the data
@st.cache(persist=True)  # Cache the data to avoid reloading on each run
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE', 'CRASH_TIME']])
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
    return data
# Load the data
data = load_data(100000)
original_data = data.copy()  # Create a copy of the original data for later use
# Display header and slider for selecting the number of injured people
st.header("Where are the most people injured in NYC?")
injured_people = st.slider("Number of People Injured in Vehicle Collisions", 0, 19)
st.map(data.query("injured_persons >= @injured_people")[["latitude", "longitude"]].dropna(how="any"))
# Display header and slider for selecting the hour
st.header("How many collisions occur during a given time of day?")
hour = st.slider("Hour to look at", 0, 23)
filtered_data = data[data['date/time'].dt.hour == hour]
st.markdown("Vehicle Collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = np.average(filtered_data['latitude']), np.average(filtered_data['longitude'])
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=filtered_data[['date/time', 'latitude', 'longitude']],
            get_position=['longitude', 'latitude'],
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))
# Display subheader and histogram chart for breakdown by minute
st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = data[
    (data['date/time'].dt.hour >= hour) & (data['date/time'].dt.hour < (hour + 1))
]
hist = np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
st.plotly_chart(fig)
# Display header and selectbox for selecting the affected type
st.header("Top 5 Dangerous Streets by Affected Type")
select = st.selectbox('Affected Type of People', ['Pedestrians', 'Cyclists', 'Motorists'])
if select == 'Pedestrians':
    st.write(original_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]]
             .sort_values(by=['injured_pedestrians'], ascending=False).dropna(how='any')[:5])
elif select == 'Cyclists':
    st.write(original_data.query("injured_cyclists >= 1")[["on_street_name", "injured_cyclists"]]
             .sort_values(by=['injured_cyclists'], ascending=False).dropna(how='any')[:5])
else:
    st.write(original_data.query("injured_motorists >= 1")[["on_street_name", "injured_motorists"]]
             .sort_values(by=['injured_motorists'], ascending=False).dropna(how='any')[:5])
# Checkbox to display raw data
if st.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(data)