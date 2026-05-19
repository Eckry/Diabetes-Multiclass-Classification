# Diabetes Multiclass Classification & Model Comparison

This repository features a complete end-to-end Data Science workflow for the multiclass classification of diabetes patients using clinical data. The primary objective is to evaluate and compare the performance of a **Deep Learning approach (Multi-Layer Perceptron using Keras)** against optimized **Classic Machine Learning models (SVM and KNN)** via hyperparameter tuning.

## 🚀 Key Features & Implementation Best Practices

To ensure statistical validity and model robustness, several industry-standard practices were strictly followed:
* **Data Leakage Prevention:** The dataset partitioning (`train_test_split`) was executed prior to any feature engineering steps. Data transformers (`StandardScaler`, `OneHotEncoder`, and `LabelEncoder`) were fitted *exclusively* on the training set and subsequently used to transform both the training and testing sets.
* **Class Imbalance Handling:** To address the intrinsic class imbalance within the clinical dataset labels, class weights were computed using `compute_class_weight` to penalize the loss function in the neural network. Similarly, the `class_weight='balanced'` parameter was enabled for the SVM model.
* **Hyperparameter Optimization via Cross-Validation:** Utilized `GridSearchCV` with 5-fold cross-validation to search the optimal parameter space, optimizing for the `f1_weighted` metric.
* **Deep Learning Regularization Strategies:** Incorporated `BatchNormalization` layers to accelerate convergence stability and an `EarlyStopping` callback with `restore_best_weights` to prevent overfitting.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.11.9
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Tuning:** Scikit-Learn (`GridSearchCV`, `StandardScaler`, `OneHotEncoder`, `SVC`, `KNeighborsClassifier`)
* **Deep Learning:** TensorFlow / Keras (`Sequential`, `Dense`, `BatchNormalization`, `Callbacks`)
* **Data Visualization:** Matplotlib, Scikit-Learn Metrics (`ConfusionMatrixDisplay`)

---

## 📊 Project Workflow

### 1. Data Preprocessing & Cleaning
* Removed duplicate records based on patient constraints and dropped irrelevant identifiers (`ID`, `No_Pation`).
* Encoded categorical features (`Gender`) using *One-Hot Encoding*.
* Applied standard scaling to numerical attributes to optimize convergence for the neural network and distance-based algorithms (KNN/SVM).

### 2. Deep Learning Model Architecture (MLP)
A sequential dense neural network was designed with the following architecture:
1.  Dense Layer (512 units, ReLU activation) + BatchNormalization
2.  Dense Layer (512 units, ReLU activation) + BatchNormalization
3.  Dense Layer (512 units, ReLU activation) + BatchNormalization
4.  Output Layer (3 units, Softmax activation) for multiclass target mapping.
* **Compilation:** `Adam` optimizer, `sparse_categorical_crossentropy` loss function.

### 3. Classic Machine Learning Baseline
* **Support Vector Machine (SVC):** Fine-tuned parameters exploring `kernel` types (Linear vs. RBF) and regularization coefficients `C`.
* **K-Nearest Neighbors (KNN):** Adjusted the number of neighbors `n_neighbors`, distance weighting (`uniform` vs. `distance`), and distance metrics (`euclidean` vs. `manhattan`).

---

## 📈 Results & Evaluation

Models were thoroughly evaluated using confusion matrices and detailed classification reports, focusing heavily on **Recall** and **F1-Score** due to the high-stakes nature of clinical medical diagnostics.

### Model Performance Comparison
*(Note: Update this table with the actual decimal values printed by your script's classification report)*

| Model | Recall (Weighted) | F1-Score (Weighted) | Performance Insights |
| :--- | :---: | :---: | :--- |
| **Neural Network (MLP)** | *0.94* | *0.94* | Outstanding generalization capability driven by EarlyStopping. |
| **Optimized SVM** | *0.94* | *0.95* | Robust decision boundaries achieved via fine-tuned C-parameters. |
| **Optimized KNN** | *0.92* | *0.92* | Highly sensitive to the local feature density of clinical metrics. |

### Visualizations
*(Save your confusion matrix plots as .png files, upload them to an `images` folder in this repo, and uncomment the lines below to display them)*

---

## 💻 How to Run This Project

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/diabetes-multiclass-classification.git](https://github.com/YourUsername/diabetes-multiclass-classification.git)
   ```
2. Install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib
    ```bash
3. Ensure the dataset.csv file is placed in the project root directory.
4. Run the Python script or open the Jupyter Notebook to reproduce the training and evaluation pipelines.
