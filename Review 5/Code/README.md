# ISE_02-University-Project
# 🐄 Cattle Breed Classification using EfficientNet (Streamlit App)

## 📌 Project Overview

This project is an AI-based web application that classifies cattle/buffalo breeds from images using a deep learning model based on EfficientNet. The system allows users to upload an image through a web interface and get predictions along with the top-3 probable breeds.

---

# 🧠 Model Training (Google Colab)

## 🔹 Step 1: Open Google Colab

* Upload the provided `.ipynb` notebook.
* Enable GPU:

  * Go to **Runtime → Change runtime type → GPU**

---

## 🔹 Step 2: Upload Dataset

* Upload your dataset (ZIP file) or mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

* Extract dataset:

```python
import zipfile
with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")
```

---

## 🔹 Step 3: Train the Model

* Run all cells in sequence.
* The model uses:

  * EfficientNet (pretrained)
  * Transfer learning
  * Adam optimizer
  * CrossEntropy loss

---

## 🔹 Step 4: Save the Model

After training:

```python
torch.save(model.state_dict(), "model.pth")
```

---

## 🔹 Step 5: Download Model File

```python
from google.colab import files
files.download("model.pth")
```

👉 This file is required for deployment.

---

# 📁 Project Setup (Local System - VS Code)

## 🔹 Step 1: Extract ZIP File

* Extract the project ZIP (`cattle_app.zip`)
* Open the folder in **VS Code**

---

## 🔹 Step 2: Place Model File

* Copy the downloaded `model.pth`
* Paste it inside the project directory:

```
project_folder/
│── model.pth   ✅ (important)
│── app.py
│── run.py
│── requirements.txt
```

---

# 🧩 Supporting Files Explanation

## 🔸 1. `app.py`

* Main Streamlit application
* Handles:

  * UI (upload image)
  * Loading model
  * Making predictions
  * Displaying results

---

## 🔸 2. `run.py`

* Helper script (optional)
* Can be used to:

  * Initialize model
  * Run prediction logic separately

---

## 🔸 3. `requirements.txt`

Contains required libraries:

```
torch
torchvision
streamlit
numpy
pillow
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 🌐 Streamlit Integration

## 🔹 What is Streamlit?

Streamlit is a Python framework used to build interactive web apps easily.

---

## 🔹 Key Features Used

* File uploader
* Image display
* Prediction output
* Top-3 results display

---

# ▶️ Running the Application (VS Code)

## 🔹 Step 1: Open Terminal

In VS Code:

```
Terminal → New Terminal
```

---

## 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔹 Step 3: Run the App

```bash
streamlit run app.py
```

---

## 🔹 Step 4: Open in Browser

* Automatically opens:

```
http://localhost:8501
```

---

# 📸 Using the Application

## 🔹 Step 1: Upload Image

* Click **Upload Image**
* Select cattle/buffalo image

---

## 🔹 Step 2: Model Prediction

* Image is processed
* Model extracts features using EfficientNet
* Predictions are generated

---

## 🔹 Step 3: Output Display

### ✅ Shows:

* Predicted breed (Top-1)
* Top-3 predictions with probabilities

Example:

```
1. Murrah – 72%
2. Jaffarabadi – 18%
3. Surti – 10%
```

---

# 🔍 How Prediction Works

1. Image is resized and normalized
2. Passed into trained EfficientNet model
3. Model outputs probabilities
4. Top-3 predictions selected using:

```python
torch.topk()
```

---

# 📊 Evaluation

* Model performance is evaluated using:

  * Confusion Matrix

---

# ⚠️ Important Notes

* Ensure `model.pth` is in the correct folder
* Image should be clear and properly visible
* Use GPU for faster training (Colab)

---

# 🚀 Future Improvements

* Add more breeds
* Improve accuracy with larger dataset
* Deploy online (AWS / Heroku)
* Add real-time camera detection

---


# 📌 Conclusion

This project demonstrates how deep learning and transfer learning can be applied to real-world agricultural problems. Using EfficientNet and Streamlit, we created a complete pipeline from model training to deployment and user interaction.

---
