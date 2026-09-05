# Text Vectorizer: Bag of Words, TF-IDF, dan Word Embedding (Word2Vec)

Implementasi dan verifikasi tiga teknik dasar untuk mengubah teks menjadi representasi numerik (vektor) yang bisa diproses model machine learning.

## Teknik yang Dicakup

| Teknik | Library | Ide Utama |
|---|---|---|
| Bag of Words (BoW) | `sklearn.CountVectorizer` | Hitung frekuensi kemunculan kata, abaikan urutan |
| TF-IDF | `sklearn.TfidfVectorizer` | Bobot kata disesuaikan seberapa umum kata itu di seluruh korpus |
| Word Embedding | `gensim.Word2Vec` | Petakan kata ke vektor berdimensi rendah berdasarkan konteks |

## Hasil Terverifikasi

Dokumen contoh: `"saya suka machine learning"` dan `"machine learning adalah masa depan"`.

Kata yang muncul di **kedua** dokumen (`machine`, `learning`) mendapat bobot TF-IDF lebih rendah (~0.35–0.41) dibanding kata yang cuma muncul di satu dokumen (~0.50–0.58) — inilah yang membedakan TF-IDF dari BoW biasa:

![Perbandingan bobot TF-IDF](images/tfidf-weight-comparison.png)

*Grafik di atas dihasilkan langsung dari eksekusi notebook, bukan ilustrasi manual.*

## Audit & Catatan Jujur

- **Word2Vec dengan korpus 2 kalimat pendek tidak menghasilkan embedding yang bermakna.** Similarity score antar kata semuanya di bawah 0.1 — nilai ini murni ilustrasi cara pakai API, bukan bukti bahwa model "memahami makna". Word2Vec butuh korpus jauh lebih besar (biasanya jutaan kata) untuk hasil yang bisa diandalkan.

## Cara Menjalankan

```bash
pip install -r requirements.txt
jupyter notebook text_vectorizer_bow_tfidf_word2vec.ipynb
```

## Struktur Repo

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── text_vectorizer_bow_tfidf_word2vec.ipynb
└── images/
    └── tfidf-weight-comparison.png
```

## Proyek Terkait

Penerapan TF-IDF untuk sistem pencarian FAQ (termasuk audit Euclidean vs Cosine similarity) dibahas di repo terpisah: [`faq-search-engine-tfidf-sastrawi-arielshakaramiro`](https://github.com/arielshakaramiro/faq-search-engine-tfidf-sastrawi-arielshakaramiro).

---

*Bagian dari catatan belajar AI Engineering saya — sesi materi NLP, rubythalib.ai AI Engineer Bootcamp.*
