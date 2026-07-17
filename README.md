# 🍱 Food Vision AI

A Streamlit web app that identifies food from a photo, estimates calories, and includes a BMI calculator — powered by a fine-tuned **InceptionV3** image classifier trained on the UECFOOD100 dataset.

---

## ✨ Features

- **Food recognition** — upload a JPG/PNG/WEBP photo and get the top-3 predicted dishes with confidence scores
- **Calorie calculator** — per-100g calorie lookup plus a portion-size calculator (grams → total kcal, % of a 2000 kcal daily goal)
- **BMI calculator** — height/weight input with category classification (Underweight / Normal / Overweight / Obese) and health tips
- **Calorie reference table** — searchable table of all 40 food classes with calorie levels
- **Demo mode** — if model files are missing, the app falls back to randomized demo predictions so the UI still works

---

## 🧠 Model

| | |
|---|---|
| **Architecture** | InceptionV3 (ImageNet pretrained backbone) |
| **Classes** | 40 food categories (UECFOOD100 subset) |
| **Input size** | 299 × 299 px |
| **Training strategy** | Two-phase transfer learning |
| **Final validation accuracy** | ~73.5% |

**Training phases:**
1. **Phase 1 — Feature extraction (25 epochs):** InceptionV3 backbone frozen; only a custom Dense head (GlobalAveragePooling → Dense(512, ReLU) → Dense(40, softmax)) is trained.
2. **Phase 2 — Fine-tuning (10 epochs):** Last 249 layers of the backbone unfrozen and retrained at a low learning rate (1e-5) to adapt pretrained ImageNet features to food images.

Data augmentation (rotation, zoom, width/height shift, shear, horizontal flip) was applied during training. `EarlyStopping` and `ModelCheckpoint` (on `val_accuracy`) were used across both phases.

### Food classes
Japanese-style pancake, beef curry, beef noodle, bibimbap, chicken 'n' egg on rice, chicken rice, chip butty, croissant, croquette, eels on rice, fried noodle, fried rice, gratin, grilled eggplant, hamburger, miso soup, oden, omelet, pilaf, pizza, pork cutlet on rice, potage, raisin bread, ramen noodle, rice, roll bread, sandwiches, sausage, sauteed spinach, sauteed vegetables, soba noodle, spaghetti, sushi, takoyaki, tempura bowl, tempura udon, tensin noodle, toast, udon noodle, vegetable tempura

---

## 📁 Project structure

```
FoodAI/
├── app.py                      # Streamlit application
├── food_model.h5                # Trained InceptionV3 model (Keras)
├── class_names.json             # List of the 40 food class names
├── calorie_dict.json            # Per-100g calorie lookup table
├── model_info.json              # Training metadata (accuracy, epochs, etc.)
├── training_history.csv         # Epoch-by-epoch accuracy/loss log
├── training_plot.png            # Accuracy/loss curves
└── phase_comparison_plot.png    # Phase 1 vs Phase 2 comparison
```

---

## 🚀 Getting started

### Requirements
```
streamlit
tensorflow
numpy
pandas
pillow
```

### Run locally
```bash
pip install streamlit tensorflow numpy pandas pillow
streamlit run app.py
```

Make sure `food_model.h5`, `class_names.json`, and `calorie_dict.json` are in the same directory as `app.py` — without them the app runs in **Demo Mode** with randomized predictions.

---

## 🏋️ Training your own model

The model was trained in Google Colab using `InceptionV3_Only_Food_Classifier.ipynb`, which:

1. Downloads and extracts the UECFOOD100 dataset (via Google Drive)
2. Selects 40 food classes and builds a train/val split
3. Sets up `ImageDataGenerator`-based augmentation pipelines
4. Builds an InceptionV3-based model with a custom classification head
5. Runs Phase 1 (frozen backbone) then Phase 2 (fine-tuning) training
6. Evaluates the best checkpoint and generates a classification report
7. Saves `food_model.h5`, `class_names.json`, `calorie_dict.json`, `training_history.csv`, and training plots to Google Drive

To retrain: open the notebook in Colab, mount your Drive, update the `SAVE_DIR` path, and run all cells.

---

## ⚠️ Disclaimer

Calorie values are approximate per-100g estimates for reference only, not medical or dietary advice. Consult a healthcare professional or nutritionist for personalized guidance.

---

## 📄 License

Add your preferred license here (e.g. MIT).
