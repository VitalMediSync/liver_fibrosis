import streamlit as st
from streamlit_image_carousel import ImageCarousel

st.set_page_config(layout="wide")

st.title("Liver Ultrasound Image Gallery")

# Prepare your image data
# You'll need a list of image URLs or paths to local files.
# For local files, ensure they are accessible when the app runs (e.g., in the same repo for Streamlit Cloud)
image_list = [
    "https://placehold.co/600x400/FF0000/FFFFFF?text=Image+1", # Example URL
    "https://placehold.co/600x400/00FF00/000000?text=Image+2",
    "https://placehold.co/600x400/0000FF/FFFFFF?text=Image+3",
    # Add your actual image paths/URLs here
    # For local files in your app's directory:
    # "images/fibrosis_f0_1.png",
    # "images/fibrosis_f1_2.png",
]

# --- Basic Carousel ---
st.header("Basic Image Carousel")
ImageCarousel(image_list)

# --- Carousel with Customization ---
st.header("Customized Carousel")
ImageCarousel(
    image_list,
    height=300,            # Height of the carousel
    width=500,             # Width of the carousel
    loop=True,             # Loop back to the start when end is reached
    autoplay=True,         # Auto-play the slideshow
    autoplay_interval=3000, # Interval in milliseconds (3 seconds)
    # You can also add titles/descriptions for each image if the component supports it,
    # often by passing a list of dicts instead of just strings.
    # Check the component's GitHub repo for full options.
)

st.write("---")
st.write("You can replace the placeholder URLs with your actual ultrasound image paths or URLs.")
