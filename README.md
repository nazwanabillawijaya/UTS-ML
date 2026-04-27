# Klasifikasi Buah Orange dan Grapefruit Menggunakan Metode Machine Learning

## Deskripsi

Proyek ini bertujuan untuk membangun model klasifikasi yang dapat membedakan buah **orange** dan **grapefruit** berdasarkan fitur numerik yang tersedia pada dataset. Model yang digunakan dalam penelitian ini meliputi Decision Tree, Naive Bayes, dan Support Vector Machine (SVM).

Dataset yang digunakan dapat diakses melalui:
https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit

---

## Tahapan Pembuatan Model

### 1. Pengumpulan Data

Dataset diperoleh dari Kaggle yang berisi atribut numerik dari buah serta label kategori (orange atau grapefruit).

---

### 2. Pemahaman Data (Exploratory Data Analysis)

Pada tahap ini dilakukan analisis awal terhadap dataset untuk memahami karakteristik data, meliputi:

* Pemeriksaan struktur data
* Identifikasi nilai kosong (missing values)
* Distribusi jumlah data pada masing-masing kelas

Contoh kode:

```python
df.info()
df['name'].value_counts()
```

---

### 3. Pra-pemrosesan Data

#### a. Transformasi Label

Label kategori diubah menjadi nilai numerik agar dapat diproses oleh model.

```python
from sklearn.preprocessing import LabelEncoder
df['label'] = LabelEncoder().fit_transform(df['name'])
```

#### b. Pemisahan Fitur dan Target

```python
X = df.drop(['name', 'label'], axis=1)
y = df['label']
```

#### c. Pembagian Data

Data dibagi menjadi data latih dan data uji dengan perbandingan 80:20.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### d. Normalisasi Data

Normalisasi dilakukan untuk meningkatkan performa model tertentu, terutama SVM.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 4. Pembangunan Model

Tiga algoritma digunakan dalam penelitian ini, yaitu:

* Decision Tree
* Naive Bayes
* Support Vector Machine (SVM)

---

### 5. Pelatihan Model

Setiap model dilatih menggunakan data latih.

```python
model.fit(X_train, y_train)
```

---

### 6. Evaluasi Model

Evaluasi dilakukan menggunakan data uji dengan beberapa metrik, antara lain:

* Accuracy
* Precision
* Recall
* F1-score

Contoh kode:

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

### 7. Perbandingan Model

Hasil dari ketiga model dibandingkan berdasarkan nilai evaluasi untuk menentukan model dengan performa terbaik.

---

### 8. Penarikan Kesimpulan

Kesimpulan diambil berdasarkan hasil evaluasi dan perbandingan performa model.

---

## Hasil Evaluasi

### Decision Tree

![Decision Tree](decision-tree.png)

### Naive Bayes

![Naive Bayes](naive-bayes.png)

### Support Vector Machine (SVM)

![SVM](SVM.png)

---

## Perbandingan Akurasi

| Model                  | Accuracy |
| ---------------------- | -------- |
| Decision Tree          | 0.942    |
| Naive Bayes            | 0.920    |
| Support Vector Machine | 0.937    |

---

## Analisis

Berdasarkan hasil evaluasi, model Decision Tree menunjukkan performa terbaik dengan tingkat akurasi sebesar 94.2%. Hal ini menunjukkan bahwa model tersebut mampu menangkap pola data dengan baik.

Model Support Vector Machine juga memberikan hasil yang cukup baik dengan akurasi sebesar 93.7%, sedangkan Naive Bayes menghasilkan akurasi sebesar 92%. Performa Naive Bayes yang lebih rendah kemungkinan disebabkan oleh asumsi independensi antar fitur yang tidak sepenuhnya sesuai dengan karakteristik dataset.

---

## Kesimpulan

Dari hasil pengujian yang dilakukan, dapat disimpulkan bahwa model Decision Tree merupakan metode terbaik untuk kasus klasifikasi pada dataset ini karena memiliki nilai akurasi tertinggi dibandingkan model lainnya.

---

## Struktur Proyek

```
uts-ml/
│── citrus.csv
│── main.py
│── decision-tree.jpg
│── naive-bayes.jpg
│── svm.jpg
│── README.md
```

---

## Cara Menjalankan Program

1. Install dependensi:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Jalankan program:

```
python main.py
```

---

## Penulis

Nama: (NAZWA NABILLA WIJAYA-1237050116
)
