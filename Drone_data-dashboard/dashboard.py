import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tifffile

# Load TIFF data
tiff_data = tifffile.imread("data/lowlowlowres.tif")
# Create a Streamlit app
st.title("Emali Farm Dashboard")

# Display TIFF data
with st.container():
    rotated_tiff_data = np.rot90(tiff_data)
    st.image(
        rotated_tiff_data,
        caption="Emali Avocado Farm",
        width=800,
    )
# Add interactive elements (optional)
contrast = st.slider("Image contrast", min_value=0.0, max_value=2.0, value=1.0)

# Extract pixel values
pixel_values = tiff_data.flatten()

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        # create a histogram

        fig, ax = plt.subplots()
        ax.hist(pixel_values, bins=10)
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Avocado Yield Report")

        # Display histogram in Streamlit app
        st.pyplot(fig)
    with col2:
        # heatmap
        flat_data = tiff_data.reshape(-1, tiff_data.shape[2])

        # Create a heatmap using matplotlib
        fig, ax = plt.subplots()
        im = ax.imshow(flat_data, cmap="hot", interpolation="nearest")
        ax.set_title("Heatmap of Avocado Yield Report")
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        fig.colorbar(im, ax=ax)

        # show plot
        st.pyplot(fig)
