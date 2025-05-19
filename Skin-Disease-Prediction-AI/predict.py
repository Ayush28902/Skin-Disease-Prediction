"""
    This script loads the trained model and predicts the skin disease for a batch of images from a directory.
"""

import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# Load your trained model
model = load_model("model/skin_disease_model.h5")

# Define the correct class labels (based on HAM10000)
classes = [
    'Actinic keratoses and intraepithelial carcinoma (akiec)',
    'Basal cell carcinoma (bcc)',
    'Benign keratosis-like lesions (bkl)',
    'Dermatofibroma (df)',
    'Melanoma (mel)',
    'Melanocytic nevi (nv)',
    'Vascular lesions (vasc)'
]


# Folder containing images to predict
image_dir = 'attachments/image_to_predict' 

# Get all image filenames
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Predict each image
for img_name in tqdm(image_files):
    img_path = os.path.join(image_dir, img_name)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Predict
    prediction = model.predict(x)[0]
    sorted_result = sorted(
        [(classes[i], float(prediction[i]) * 100.0) for i in range(len(prediction))],
        key=lambda x: x[1], reverse=True
    )

    # Display results
    print(f"\n📷 Image: {img_name}")
    for rank, (label, prob) in enumerate(sorted_result, 1):
        print(f"Top {rank}: {label} → {prob:.2f}%")
