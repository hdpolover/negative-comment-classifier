# Negative Comment Classifier

A machine learning application that classifies texts/comments based on their level of negativity 

Team members:
- 171111022	-- Suhendra
- 171111036	-- Nurami Nasrullah
- 171111069	-- Qori Hidayatulloh

## Langkah Pembuatan Program:

*  Data Collection

Aplikasi ini menggunakan pre-collected toxic comment dataset dari [Kaggle](https://www.kaggle.com/nichaoku/toxic-comment-merge-train-and-test-with-label).

* Data Preparation

Melakukan analisa awal dengan melihat isi baris dan kolom dari data untuk menentukan atribut mana yang cocok digunakan sebagai parameter. Kemudian membagi data ke dalam atribut dan label. Setelah itu membagi data ke dalam training dan test set.

* Choosing a Model

Menentukan algoritma apa yang akan digunakan untuk mengklasifikasi dataset. Aplikasi ini menggunakan algoritma SVM, Karena algoritma ini bisa mengklasifikasikan data dengan baik dan cepat.

* Training the Model

Melakukan training atau memberi pengetahuan terhadap algoritma sebelum bisa melakukan prediksi.

* Evaluating the Model

Menggunakan data sample modifikasi dari training set untuk melakukan evaluasi terhadap efektifitas dan efesiensi dari algoritma yang dipilih. Seperti tingkat akurasi dan lama pemrosesan. 

* Making Predictions

Membuat prediksi nyata dengan menggunakan sampel dari test set atau data yang tidak termasuk dalam training set. Input dari web kemudian dapat digunakan untuk melakukan prediksi data baru.

## Library Yang Dibutuhkan:

* [Pandas](https://pandas.pydata.org/) untuk mengolah dataset menjadi table dataframe yang bisa dibaca.
* [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)   untuk mengubah dataset menjadi vector agar bisa digunakan sebagai input untuk proses prediksi.
* [Train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) untuk membagi dataset kedalam Training Set dan Test Set serta meminimalisir overfitting atau underfitting dengan mengacak dataset.
* [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) untuk mengklasifikasi dan memprediksi data. LinearSVC dipilih karena proses eksekusinya lebih cepat dan mampu memisahkan data dengan baik. 
* [Accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) untuk menghitung akurasi dari hasil prediksi. 

## Prerequisites:

Untuk menjalankan program ini di direktori lokal, Anda harus menginstal beberapa _tool_ yang dibutuhkan seperti akan dijelaskan dibawah. Namun jika Anda sudah melakukan semuanya, Anda dapat langsung mendownload atau _clone_ repositori ini.

```bash
git clone https://github.com/hdpolover/negative-comment-classifier.git
```

#### Installing Python :

Python dapat diunduh di https://www.python.org/downloads/. Ikuti langkah instalasinya sampai selesai.

#### Installing PIP

PIP dapat diunduh di https://pip.pypa.io/en/stable/installing/ atau dengan menginstallnya melalui console.

```bash
python install pip 
```
#### Installing Flask

Install Virtual Environment pada direktori yang ingin anda gunakan.

```bash
python -3 –m venv venv 
```
Kemudian unduh Flask di http://flask.pocoo.org/ atau melalui PIP.

```bash
pip install flask
```



Setelah semua proses berhasil, jalankan program di direktori dimana anda mengunduh atau _clone_ repositori ini.

```python
python flask_app.py
```

Lalu akses http://127.0.0.1:5000 di browser Anda. 

