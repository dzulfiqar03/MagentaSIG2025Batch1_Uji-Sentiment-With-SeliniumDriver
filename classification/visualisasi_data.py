import pandas as pd
import nltk
# Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit

import matplotlib.pyplot as plt

import numpy as np

def visualisasi(url_path):
# Contoh data dengan 3 label: 2 = Positif, 1 = Netral, 0 = Negatif
    texts = [
        "saya senang belajar",       # Positif
        "ini adalah bencana",        # Negatif
        "saya sangat senang",        # Positif
        "saya tidak suka ini",       # Negatif
        "saya merasa biasa saja",    # Netral
        "ini adalah hari yang baik", # Positif
        "saya tidak punya pendapat", # Netral
        "ini cukup membosankan"      # Negatif
    ]
    labels = [2, 0, 2, 0, 1, 2, 1, 0]  # Label sesuai dengan teks

    # Langkah praproses: vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Bagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # Model SVM dengan kernel linear
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    print("Akurasi:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, target_names=["Negatif", "Netral", "Positif"]))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Negatif", "Netral", "Positif"], zero_division=0))


    path = url_path

    df = pd.read_csv(path)
    print(df)

    df['Ulasan_preprocessed'] = df['Ulasan_preprocessed'].fillna('')
    texts = df['Ulasan_preprocessed']

    print(df['label_sentiment'].value_counts())

    print(df['Ulasan_preprocessed'])

    # Menentukan texts dan labels dari DataFrame
    texts = df['Ulasan_preprocessed']
    labels = df['label_sentiment']

    # Langkah praproses: Vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # Melatih model SVM dengan kernel linear
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Melakukan prediksi dan evaluasi
    y_pred = model.predict(X_test)

    # Mencetak akurasi dan laporan klasifikasi
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # Menentukan texts dan labels dari DataFrame
    texts = df['Ulasan_preprocessed'].fillna("")  # Replace NaN with empty string
    labels = df['label_sentiment']

    # Langkah praproses: Vektorisasi dengan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts.astype(str))  # Convert to string type explicitly

    # Membagi data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # Fungsi untuk melatih model SVM dengan kernel tertentu dan menampilkan hasil
    def evaluate_svm_kernel(kernel_name):
        # Melatih model dengan kernel tertentu
        model = SVC(kernel=kernel_name, C=1.0)
        model.fit(X_train, y_train)

        # Melakukan prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)

        # Laporan klasifikasi dengan 3 angka di belakang koma
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Netral", "Positif", "Negatif"],
            zero_division=0,
            output_dict=True
        )

        # Membulatkan hasil hingga 3 angka di belakang koma
        report_df = pd.DataFrame(report_dict).transpose().round(3)

        # Menampilkan hasil
        print(f"\nKernel: {kernel_name}")
        print(f"Akurasi: {accuracy:.3f}")
        print(report_df)

        # Opsional: Simpan hasil ke file CSV
        report_df.to_csv(f"classification/classification_report_{kernel_name}.csv", index=True)

        

    # Fungsi untuk melatih model SVM dengan kernel tertentu dan menampilkan hasil
    def evaluate_svm_kernel(kernel_name):
        # ... (your existing code for model training and evaluation) ...
        # Melatih model dengan kernel tertentu
        model = SVC(kernel=kernel_name, C=1.0)
        model.fit(X_train, y_train)

        # Plot Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Netral", "Positif", "Negatif"],
                    yticklabels=["Netral", "Positif", "Negatif"])
        plt.title(f"Confusion Matrix - {kernel_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            SVC(kernel=kernel_name, C=1.0), X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8,6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curve - {kernel_name}')
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    
    # ... (Your existing code)

    # Fungsi untuk menampilkan learning curve (tidak perlu diubah)
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        # ... (Kode fungsi learning_curve yang ada)
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

    # Evaluasi untuk setiap kernel dengan learning curve dan grafik akurasi/loss
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        print(f"\nKernel: {kernel}")
        model = SVC(kernel=kernel, C=1.0)

        # Menampilkan Learning Curve
        title = f"Learning Curves (SVM, {kernel} kernel)"
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(model, title, X, labels, cv=cv, n_jobs=4)

        # Evaluasi model dengan fungsi evaluate_svm_kernel
        evaluate_svm_kernel(kernel)

        # Tambahkan kode untuk menampilkan grafik accuracy dan loss
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, labels, cv=cv, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 5))

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
        plt.plot(train_sizes, test_scores_mean, label="Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Training set size")
        plt.title(f"Accuracy vs Training set size ({kernel} kernel)")
        plt.legend(loc="best")
        plt.show()

        # Note: For Loss, you'd need to modify the learning curve to compute loss
        # or get the loss from the model directly. Currently, it only shows Accuracy.


    # Plot Training and Validation Accuracy and Loss
    def plot_accuracy_and_loss(estimator, X_train, y_train, cv, kernel_name):
        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, y_train, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)

        # Mean and std for accuracy
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Training and Validation Accuracy ({kernel_name})")
        plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Accuracy", color="blue")
        plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Accuracy", color="green")
        plt.fill_between(train_sizes,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="blue")
        plt.fill_between(train_sizes,
                        test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="green")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")

        # Loss (simulated via 1 - scores as proxy for loss)
        train_loss = 1 - train_scores_mean
        test_loss = 1 - test_scores_mean

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.title(f"Training and Validation Loss ({kernel_name})")
        plt.plot(train_sizes, train_loss, 'o-', label="Training Loss", color="blue")
        plt.plot(train_sizes, test_loss, 'o-', label="Validation Loss", color="green")
        plt.xlabel("Training Size")
        plt.ylabel("Loss")
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()

    # Evaluasi untuk setiap kernel
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        print(f"\nKernel: {kernel}")
        model = SVC(kernel=kernel, C=1.0)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        # Plot accuracy and loss
        plot_accuracy_and_loss(model, X_train, y_train, cv, kernel)
        

    # Fungsi untuk menghitung metrik secara spesifik
    def evaluate_kernel(kernel_name):
        model = SVC(kernel=kernel_name, C=1.0, probability=True)
        scoring = {
            'accuracy': 'accuracy',
            'loss': make_scorer(lambda y, y_pred: 1 - accuracy_score(y, y_pred))
        }
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=True)

        # Hasil rata-rata
        train_acc = cv_results['train_accuracy'].mean()
        val_acc = cv_results['test_accuracy'].mean()
        train_loss = cv_results['train_loss'].mean()
        val_loss = cv_results['test_loss'].mean()

        return train_acc, val_acc, train_loss, val_loss

    # Evaluasi untuk setiap kernel
    results = []
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        train_acc, val_acc, train_loss, val_loss = evaluate_kernel(kernel)
        results.append({
            'Kernel': kernel,
            'Training Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Training Loss': train_loss,
            'Validation Loss': val_loss
        })

    # Konversi hasil ke DataFrame dan tampilkan
    results_df = pd.DataFrame(results)
    print(results_df)

    # Opsional: Simpan hasil ke file CSV
    results_df.to_csv("classification/svm_kernel_evaluation.csv", index=False)

    # Pembagian data (gunakan stratifikasi jika diperlukan)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    print("Logistic Regression")
    print("Akurasi:", accuracy_score(y_test, y_pred_log_reg))
    print(classification_report(y_test, y_pred_log_reg, labels=[0, 1], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 2. Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("\nRandom Forest Classifier")
    print("Akurasi:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, labels=[0, 1], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 3. Naive Bayes Classifier (MultinomialNB)
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    print("\nNaive Bayes Classifier")
    print("Akurasi:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb, labels=[0, 1], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    # 4. K-Nearest Neighbors (KNN)
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    print("\nK-Nearest Neighbors (KNN)")
    print("Akurasi:", accuracy_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn, labels=[0, 1], target_names=["Netral", "Positif", "Negatif"], zero_division=0))

    df_test = df.iloc[y_test.index].copy()  # Menggunakan index yang sama dengan data uji

    # Menambah kolom untuk setiap hasil prediksi model
    df_test['predicted_label_svm'] = y_pred
    df_test['predicted_label_log_reg'] = y_pred_log_reg
    df_test['predicted_label_rf'] = y_pred_rf
    df_test['predicted_label_nb'] = y_pred_nb
    df_test['predicted_label_knn'] = y_pred_knn

    # Menampilkan DataFrame dengan kolom baru
    print(df_test.head())

    # Menyimpan DataFrame ke file CSV jika diperlukan
    df_test.to_csv('classification/lama_hasil_prediksi_berbagai_model2.csv', index=False)