# 🍱 Food Vision AI

A deep learning web app that identifies food from a photo, estimates its calorie content, and includes a BMI calculator with a searchable calorie reference table. Built with **TensorFlow/Keras (InceptionV3)** and **Streamlit**.

## Features

- **Food Recognition** — Upload a food image and get the top-3 predicted classes with confidence scores.
- **Calorie Estimation** — Automatic per-100g calorie lookup plus a portion-size calculator that shows total calories and % of a 2000 kcal daily goal.
- **BMI Calculator** — Enter height and weight to get your BMI, category (Underweight/Normal/Overweight/Obese), and a health tip.
- **Calorie Reference Table** — Searchable table of all 40 food classes with calorie levels (Low/Medium/High/Very High).
- **Demo Mode** — The app runs even without the trained model files, using sample data, so the UI can be explored right away.

## Model

- **Architecture:** InceptionV3 (ImageNet pretrained) with a custom classification head (GlobalAveragePooling → Dense(512) → BatchNorm → Dropout → Dense(256) → Dropout → Softmax).
- **Dataset:** [UECFOOD100](https://foodcam.mobi/dataset100.html), first 40 classes.
- **Training strategy:** Two-phase transfer learning.
  - **Phase 1** — Base frozen, only the dense head is trained (25 epochs, LR 1e-3).
  - **Phase 2** — Last 249 layers of InceptionV3 unfrozen and fine-tuned with a very small learning rate (1e-5, 10 epochs).
- **Input size:** 299×299 (InceptionV3 native resolution).
- **Final validation accuracy:** ~73.5%
- **Callbacks used:** EarlyStopping, ModelCheckpoint (best val_accuracy), ReduceLROnPlateau.

Training was done in Google Colab; see [`InceptionV3_Only_Food_Classifier.ipynb`](./InceptionV3_Only_Food_Classifier.ipynb) for the full pipeline, including dataset download/extraction, train/val split, data augmentation, model building, training, and evaluation (classification report + accuracy/loss plots).

## Project Structure

```
FoodAI/
├── app.py                     # Streamlit application
├── food_model.h5              # Trained InceptionV3 model
├── class_names.json           # List of 40 food class labels
├── calorie_dict.json          # Calorie (kcal/100g) lookup per class
├── model_info.json            # Model metadata (architecture, epochs, accuracy)
├── training_history.csv       # Epoch-by-epoch accuracy/loss
├── training_plot.png          # Accuracy/loss curves
└── phase_comparison_plot.png  # Phase 1 vs Phase 2 comparison

InceptionV3_Only_Food_Classifier.ipynb   # Training notebook (Colab)
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/food-vision-ai.git
cd food-vision-ai
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. If `food_model.h5`, `class_names.json`, and `calorie_dict.json` are present in the project folder, it runs with real predictions; otherwise it falls back to **Demo Mode** with sample data.

### requirements.txt

```
streamlit
tensorflow
numpy
pandas
pillow
```

## Retraining the Model

1. Open `InceptionV3_Only_Food_Classifier.ipynb` in Google Colab.
2. Mount Google Drive and set the dataset/output paths in the config cell.
3. Run all cells — the notebook downloads UECFOOD100, builds the train/val split, trains in two phases, evaluates, and saves all output files (`food_model.h5`, `class_names.json`, `calorie_dict.json`, plots, and history CSV).
4. Copy the generated files into the app folder to update the deployed model.

## Notes

- Calorie values are approximate, sourced from general nutrition references and mapped per food class.
- The classifier currently supports 40 food classes from the UECFOOD100 dataset (mostly Japanese and Western dishes).

## License

Add your preferred license here (e.g. MIT).
