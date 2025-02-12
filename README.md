# Pixify - AI-Powered Photo Mosaic Generator
Pixify is an interactive photo mosaic generator built with Python and Streamlit. It allows users to transform a single image into a beautiful mosaic composed of smaller part images. The app automatically optimizes and arranges images based on color similarity, ensuring a seamless and visually appealing result.

Features
✅ Interactive Setup: Upload a target image and multiple part images effortlessly via an intuitive UI.
✅ Smart Mosaic Generation: Uses K-Means clustering to find the best part images for each tile.
✅ Customizable Settings: Adjust tile size, resolution (DPI), scaling factor, and alpha blending for the perfect mosaic.
✅ Real-Time Preview: Compare the original vs. mosaic with an interactive slider.
✅ Export Options: Download the final mosaic as a high-quality PNG or PDF.

## Repository Structure

```plaintext
pixify/
├── Pixify.py              # Main Streamlit application
├── pictures/              # Example images and UI assets
├── streamlit/             # Streamlit configuration files
├── requirements.txt       # List of Python dependencies
└── README.md              # This file
