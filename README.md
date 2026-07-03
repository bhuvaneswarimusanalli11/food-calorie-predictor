# 🍽️ Food Calorie Predictor

An end-to-end deep learning application that classifies food images and estimates their calorie content using a fine-tuned InceptionV3 model. The project is deployed as an interactive web application with Streamlit, allowing users to upload food images and instantly receive predictions.

---

## 📌 Overview

Food Calorie Predictor is a deep learning project that takes an image of a food item as input, identifies the food category, and estimates its calorie content.

The model is trained on **40 food categories** from the **UECFOOD100** dataset using **Transfer Learning** with **InceptionV3**. Along with the predicted food label, the application also displays:

- Estimated calorie value
- Confidence score
- Top-5 predicted classes
- Inference time

This project demonstrates the complete deep learning workflow, including data preprocessing, model training, evaluation, and deployment through a user-friendly Streamlit interface.

---

## 🧠 Model Architecture

| Attribute | Details |
|-----------|---------|
| Base Model | InceptionV3 (Transfer Learning) |
| Dataset | UECFOOD100 |
| Input Size | 299 × 299 × 3 |
| Number of Classes | 40 |
| Validation Accuracy | ~73.7% |
| Framework | TensorFlow / Keras |

The model is fine-tuned from the pretrained InceptionV3 network (trained on ImageNet) using a curated subset of the UECFOOD100 dataset. It can recognize common food items such as pizza, sushi, ramen, fried rice, curry, tempura, and many more.

---

## 📂 Repository Structure

```text
food-calorie-predictor/
│
├── InceptionV3_Only_Food_Classifier.ipynb   # Model training and evaluation notebook
├── app.py                                   # Streamlit web application
├── requirements.txt                         # Project dependencies
├── class_names.json                         # Food class names
├── calorie_dict.json                        # Calorie mapping for each food class
├── food_model.keras                         # Trained deep learning model
└── README.md
```

> **Note:** Make sure `food_model.keras`, `class_names.json`, and `calorie_dict.json` are present in the project directory before running the application.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/bhuvaneswarimusanalli11/food-calorie-predictor.git
cd food-calorie-predictor
```

### 2. Create a virtual environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will start locally and open in your default browser at:

```
http://localhost:8501
```

---

## 🚀 Usage

1. Launch the Streamlit application.
2. Upload a food image (`.jpg`, `.jpeg`, `.png`, or `.webp`).
3. The model analyzes the image and displays:
   - Predicted food category
   - Confidence score
   - Estimated calorie value
   - Top-5 predictions with confidence percentages
   - Inference time

---

## 📊 Results

The fine-tuned InceptionV3 model achieves a validation accuracy of approximately **73.7%** across **40 food categories**.

The repository also includes the training notebook, where you can explore the complete training process, model evaluation, and performance visualizations.

---

## 💻 Tech Stack

- **Programming Language:** Python 3.x
- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** Streamlit
- **Image Processing:** Pillow (PIL), NumPy

---

## 🛠️ Limitations & Future Improvements

### Current Limitations

- Calorie values are approximate and based on predefined mappings.
- The model currently supports only **40 food categories**.
- Portion size is not considered during calorie estimation.

### Future Enhancements

- Integrate verified nutritional databases for more accurate calorie estimation.
- Expand the dataset to support a larger variety of food items.
- Add portion-size estimation for improved calorie prediction.
- Deploy the application on cloud platforms such as **Streamlit Cloud** or **Hugging Face Spaces** for public access.

---

## 👩‍💻 Author

**Bhuvaneswari Musanalli**

Engineering Student | Deep Learning Enthusiast

---

## 📄 License

This project was developed for academic and learning purposes as part of a Deep Learning course review.
