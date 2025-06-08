import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from io import BytesIO
from ultralytics import YOLO

# Replace with the path to your YOLO model weights
model_path = 'weights/custom_yolov8.pt'

# Logo URL
logo_url = "images/Logo1.png"

# Streamlit Page Configuration
st.set_page_config(
    page_title="StockSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to download and plot stock data as a candlestick chart
def generate_chart(ticker, interval="1d", chunk_size=180, figsize=(18, 6.5), dpi=100):
    if interval == "1h":
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        period = None
    else:
        start_date = None
        end_date = None
        period = "max"
    
    data = yf.download(ticker, interval=interval, start=start_date, end=end_date, period=period)
    
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        data = data.iloc[-chunk_size:]

        fig, ax = mpf.plot(data, type="candle", style="yahoo",
                           title=f"{ticker} Latest {chunk_size} Candles",
                           axisoff=True,
                           ylabel="",
                           ylabel_lower="",
                           volume=False,
                           figsize=figsize,
                           returnfig=True)

        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi)
        buffer.seek(0)
        return buffer
    else:
        st.error("No data found for the specified ticker and interval.")
        return None

# Sidebar UI elements
with st.sidebar:
    st.image(logo_url, use_column_width="auto")
    st.header("Configurations")
    
    st.subheader("Generate Chart")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")
    interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"])
    chunk_size = 180
    if st.button("Generate Chart"):
        if ticker:
            chart_buffer = generate_chart(ticker, interval=interval, chunk_size=chunk_size)
            if chart_buffer:
                st.success(f"Chart generated successfully.")
                st.download_button(
                    label=f"Download {ticker} Chart",
                    data=chart_buffer,
                    file_name=f"{ticker}_latest_{chunk_size}_candles.png",
                    mime="image/png"
                )
                st.image(chart_buffer, caption=f"{ticker} Chart", use_column_width=True)
        else:
            st.error("Please enter a valid ticker symbol.")

    st.subheader("Upload Image for Detection")
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = float(st.slider("Select Model Confidence", 25, 100, 30)) / 100

# Main Page Title and Instructions
st.title("StockSense")
st.caption('📈 To use the app, choose one of the following options:')
st.markdown('''
**Option 1: Upload Your Own Image**
1. **Upload Image:** Use the sidebar to upload a candlestick chart image from your local PC.
2. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the uploaded chart.

**Option 2: Generate and Analyze Chart**
1. **Generate Chart:** Provide the ticker symbol and interval in the sidebar to create and download a chart (latest 180 candles).
2. **Upload Generated Chart:** Use the sidebar to upload the generated chart image.
3. **Detect Objects:** Click the :blue[Detect Objects] button to analyze the generated chart.
''')

# Columns for Displaying Images
col1, col2 = st.columns(2)

if source_img:
    with col1:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Load YOLO Model for Object Detection
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Function to identify and label patterns
# Function to identify and label patterns
def identify_patterns(boxes):
    pattern_names = []

    # Loop through each box and apply pattern detection logic
    for box in boxes:
        x, y, w, h = box.xywh[0]

        # Example pattern detection based on width and height conditions
        if w > 100 and h > 100:
            pattern_names.append("Bullish Engulfing")
        elif w < 50 and h > 100:
            pattern_names.append("Bearish Engulfing")
        elif w > 50 and h < 50:
            pattern_names.append("Hammer")
        elif w < 30 and h < 30:
            pattern_names.append("Doji")
        elif w > 70 and h < 30:
            pattern_names.append("Shooting Star")
        elif w < 50 and h > 50 and w == h:
            pattern_names.append("Spinning Top")
        elif w > 60 and h > 40:
            pattern_names.append("Morning Star")
        elif w < 40 and h > 60:
            pattern_names.append("Evening Star")
        elif w == h and w > 100:
            pattern_names.append("Marubozu")
        elif w > 50 and h > 150:
            pattern_names.append("Long-Legged Doji")
        elif w < 40 and h < 100:
            pattern_names.append("Harami")
        elif w > 100 and h < 40:
            pattern_names.append("Inverted Hammer")
        elif w > 50 and h < 100:
            pattern_names.append("Belt Hold")
        elif w > 30 and h > 50:
            pattern_names.append("Tweezer Top")
        elif w < 20 and h > 50:
            pattern_names.append("Tweezer Bottom")
        else:
            pattern_names.append("Unidentified Pattern")

    return pattern_names


# Detect Objects and Identify Patterns
if st.sidebar.button('Detect Objects'):
    if source_img:
        source_img.seek(0)
        uploaded_image = Image.open(source_img)
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        
        # Display detected image
        with col2:
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
        
        # Display pattern names with bounding box info
        try:
            pattern_names = identify_patterns(boxes)
            with st.expander("Detection Results"):
                for i, box in enumerate(boxes):
                    st.write(f"Pattern: {pattern_names[i]}, Box Coordinates: {box.xywh}")
        except Exception as ex:
            st.write("Error displaying detection results.")
    else:
        st.error("Please upload an image first.")
