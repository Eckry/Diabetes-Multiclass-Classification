# %%
import pandas as pd

# %%
df = pd.read_csv('./dataset.csv')

# %%
df = df.drop_duplicates(subset='ID')
df = df.drop(columns=['ID', 'No_Pation'])
df['CLASS'] = df['CLASS'].str.strip()
df.head()

# %%
from sklearn.model_selection import train_test_split

# %%
y = df['CLASS']
X = df.drop(columns='CLASS')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# %%
le = LabelEncoder()

y_train_encoded =  le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

num_cols = df.select_dtypes(include='number').columns
str_cols = ['Gender']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_gender_cols = pd.DataFrame(OH_encoder.fit_transform(X_train[str_cols]), index=X_train.index)
X_test_gender_cols = pd.DataFrame(OH_encoder.transform(X_test[str_cols]), index=X_test.index)

X_train_gender_cols.columns = X_train_gender_cols.columns.astype(str)
X_test_gender_cols.columns = X_test_gender_cols.columns.astype(str)

X_train = X_train.drop(columns=str_cols)
X_test = X_test.drop(columns=str_cols)

X_train = pd.concat([X_train, X_train_gender_cols], axis=1)
X_test = pd.concat([X_test, X_test_gender_cols], axis=1)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

X_train_scaled.columns = X_train.columns
X_test_scaled.columns = X_test.columns
print(X_train_scaled.head())

# %%
import numpy as np
from tensorflow.keras import Sequential, layers, callbacks
from sklearn.utils.class_weight import compute_class_weight

# %%
classes = np.unique(y_train_encoded)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)

class_weights = dict(zip(classes, weights))

input_shape = X_train.shape[1]
model = Sequential([
    layers.Dense(512, activation='relu', input_shape=[input_shape]),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callback = callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

model.fit(X_train_scaled, 
          y_train_encoded, 
          epochs=100, 
          verbose=False, 
          validation_data=(X_test_scaled, y_test_encoded),
          class_weight=class_weights,
          callbacks=[callback])
y_preds = model.predict(X_test_scaled)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, classification_report

# %%
y_preds_final = np.argmax(y_preds, axis=1)
m = confusion_matrix(y_test_encoded, y_preds_final)
lol = ConfusionMatrixDisplay(m)
print(classification_report(y_test_encoded, y_preds_final))
print(recall_score(y_test_encoded, y_preds_final, average='weighted'))

lol.plot()
xd = pd.DataFrame(model.history.history)
xd.plot()

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# %%
param_grid_svc = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    estimator=SVC(class_weight='balanced'),
    param_grid=param_grid_svc,
    cv=5,
    scoring='f1_weighted',
)

grid_svm.fit(X_train_scaled, y_train_encoded)

print(grid_svm.best_params_)
best_svm = grid_svm.best_estimator_

y_preds_svm = best_svm.predict(X_test_scaled)
m = confusion_matrix(y_test_encoded, y_preds_svm)
print(classification_report(y_test_encoded, y_preds_svm))
m_dis = ConfusionMatrixDisplay(m)
print(recall_score(y_test_encoded, y_preds_svm, average='weighted'))
m_dis.plot()


# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_knn,
    cv=5,
    scoring='f1_weighted',
)

grid_knn.fit(X_train_scaled, y_train_encoded)

print(grid_knn.best_params_)

y_preds_knn = grid_knn.predict(X_test_scaled)
m = confusion_matrix(y_preds_knn, y_test_encoded)
m_dis = ConfusionMatrixDisplay(m)
print(classification_report(y_test_encoded, y_preds_knn))
print(recall_score(y_test_encoded, y_preds_knn, average='weighted'))
m_dis.plot()


