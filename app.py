import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

# Replace with your actual model path and API key
model_path = 'weights/custom_yolov8.pt'
API_KEY = '6ZH5YJK5IGEMA8PN'  # (Currently unused in code)
logo_url = "images/Logo1.png"  # Local logo path

# Streamlit page config
st.set_page_config(
    page_title="StockSense",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_chart(ticker, interval='1d'):
    try:
        data = yf.download(ticker, period='1y', interval=interval)

        if data.empty:
            st.error("No data found for the selected ticker.")
            return None

        # Flatten multi-level columns if any
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col not in data.columns:
                st.error(f"Column '{col}' missing in data.")
                return None
            # Convert to numeric safely
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with NaNs
        data.dropna(subset=numeric_columns, inplace=True)
        if data.empty:
            st.error("No valid data after cleaning.")
            return None

        buf = BytesIO()
        mpf.plot(
            data,
            type='candle',
            style='charles',
            title=f"{ticker.upper()} Stock Chart",
            volume=True,
            savefig=dict(fname=buf, format='png')
        )
        buf.seek(0)
        return buf

    except Exception as e:
        st.error(f"Error generating chart: {e}")
        return None

# Sidebar UI
with st.sidebar:
    st.image(logo_url, use_column_width=True)
    st.header("Configurations")

    st.subheader("Generate Chart")
    ticker = st.text_input("Enter Ticker Symbol (e.g. AAPL):").strip().upper()
    interval = st.selectbox("Select Interval", ["1d", "1h", "1wk"])
    chunk_size = 180  # fixed, not currently used in generate_chart

    if st.button("Generate Chart"):
        if ticker:
            chart_buf = generate_chart(ticker, interval=interval)
            if chart_buf:
                st.success("Chart generated successfully.")
                st.download_button(
                    label=f"Download {ticker} Chart",
                    data=chart_buf,
                    file_name=f"{ticker}_latest_{chunk_size}_candles.png",
                    mime="image/png"
                )
                st.image(chart_buf, caption=f"{ticker} Chart", use_column_width=True)
        else:
            st.error("Please enter a valid ticker symbol.")

    st.subheader("Upload Image for Detection")
    source_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    confidence = st.slider("Select Model Confidence (%)", 25, 100, 30) / 100.0

# Main page
st.title("StockSense")
st.caption('📈 Use the sidebar to generate charts or upload images for object detection.')

st.markdown('''
**Options:**
- Generate a candlestick chart by entering a ticker and selecting an interval.
- Upload your own chart image to detect candlestick patterns.
''')

col1, col2 = st.columns(2)

if source_img:
    with col1:
        uploaded_image = Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Load YOLO model once
model = None
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Failed to load YOLO model from {model_path}: {e}")

def identify_patterns(boxes):
    patterns = []
    for box in boxes:
        x, y, w, h = box.xywh[0]
        # Simple heuristics based on width and height of bounding boxes
        if w > 100 and h > 100:
            patterns.append("Bullish Engulfing")
        elif w < 50 and h > 100:
            patterns.append("Bearish Engulfing")
        elif w > 50 and h < 50:
            patterns.append("Hammer")
        elif w < 30 and h < 30:
            patterns.append("Doji")
        elif w > 70 and h < 30:
            patterns.append("Shooting Star")
        elif w < 50 and h > 50 and abs(w - h) < 5:
            patterns.append("Spinning Top")
        elif w > 60 and h > 40:
            patterns.append("Morning Star")
        elif w < 40 and h > 60:
            patterns.append("Evening Star")
        elif abs(w - h) < 5 and w > 100:
            patterns.append("Marubozu")
        elif w > 50 and h > 150:
            patterns.append("Long-Legged Doji")
        elif w < 40 and h < 100:
            patterns.append("Harami")
        elif w > 100 and h < 40:
            patterns.append("Inverted Hammer")
        elif w > 50 and h < 100:
            patterns.append("Belt Hold")
        elif w > 30 and h > 50:
            patterns.append("Tweezer Top")
        elif w < 20 and h > 50:
            patterns.append("Tweezer Bottom")
        else:
            patterns.append("Unidentified Pattern")
    return patterns

if st.sidebar.button("Detect Objects"):
    if not model:
        st.error("YOLO model not loaded, cannot perform detection.")
    elif not source_img:
        st.error("Please upload an image first.")
    else:
        source_img.seek(0)
        image = Image.open(source_img)
        results = model.predict(image, conf=confidence)
        boxes = results[0].boxes

        detected_img = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB

        with col2:
            st.image(detected_img, caption="Detected Patterns", use_column_width=True)

        try:
            pattern_names = identify_patterns(boxes)
            with st.expander("Detection Results"):
                for i, box in enumerate(boxes):
                    coords = box.xywh[0].tolist()
                    st.write(f"Pattern: {pattern_names[i]}, Box (x,y,w,h): {coords}")
        except Exception as e:
            st.error(f"Error showing detection results: {e}")
