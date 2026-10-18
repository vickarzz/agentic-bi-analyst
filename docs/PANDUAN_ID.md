# Panduan Lengkap (Bahasa Indonesia): Jalankan, Deploy, dan Latih Agent

Panduan ini untuk pemilik proyek. Dokumentasi teknis berbahasa Inggris ada di `README.md` dan `docs/APPENDIX.md`.

> Semua data di proyek ini **sintetis**. "Wadi Bank" adalah bank fiktif. Jangan pernah memasukkan data atau kode milik kantor ke repo publik.

---

## 0. Isi folder

| Folder / file | Isi |
|---|---|
| `app/` | Aplikasi Streamlit: halaman chat dan halaman **Business Review** (multi-agent) |
| `bi_agent/` | Otak agent: semantic layer, compiler SQL, guardrails, planner, LangGraph |
| `bi_agent/orchestrator/` | Tim multi-agent: supervisor, 5 agent domain, impact analyst, discovery, synthesizer, verifier |
| `semantic/` | **"Pengetahuan" agent** dalam bentuk YAML: metrik, sinonim, hipotesis dampak, kalender bisnis |
| `scripts/` | Generator data, setup sekali jalan, publish ke GitHub, dan **alat training** |
| `evals/` | Pertanyaan uji (golden questions) dan laporan akurasi |
| `bi_agent/ds/` | **Data Science Lab**: causal inference, model churn, forecasting, A/B test, segmentasi |
| `tests/` | 94 test otomatis |
| `docs/presentation/` | Deck PowerPoint dan PDF (43 slide) |
| `docs/examples/` | Contoh briefing hasil multi-agent |
| `docs/APPENDIX.md` | Penjelasan pembuatan langkah demi langkah (Inggris) |

---

## 1. Jalankan di MacBook (satu perintah)

Prasyarat: **Python 3.12** dari python.org (cukup sekali install).

```bash
cd ~/Projects/agentic-bi-analyst        # sesuaikan dengan lokasi folder Anda
bash scripts/setup_mac.sh --run
```

Skrip ini otomatis:
1. membuat virtual environment `.venv`
2. menginstall semua library (hanya kali ini butuh internet)
3. membuat data dummy bank
4. menjalankan 94 test
5. menjalankan evaluasi 28 pertanyaan
6. menjalankan Data Science Lab dan mengecek hasilnya terhadap efek yang ditanam (17 pemeriksaan)
7. menjalankan review multi-agent (8 agent)
8. membuka aplikasi di **http://localhost:8501**

Setelah itu semuanya **offline**. Untuk membuka lagi di lain hari:

```bash
cd ~/Projects/agentic-bi-analyst
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Perintah lain yang berguna:

| Perintah | Fungsi |
|---|---|
| `python -m bi_agent.cli` | Chat dengan agent di Terminal |
| `python -m bi_agent.orchestrator` | Review seluruh bisnis, hasilnya di `reports/business_review.md` |
| `python scripts/train_from_feedback.py` | Laporan "apa yang perlu diajarkan" ke agent |
| `make test` / `make evals` | Cek semua masih benar |

---

## 2. Deploy semuanya sekaligus (GitHub + link demo publik)

### Persiapan (sekali saja)
1. **Akun GitHub**: login sebagai `vickarzz`.
2. **Buat repo kosong**: di github.com klik **+ → New repository**, isi nama `agentic-bi-analyst`, pilih **Public**, dan **jangan** centang "Add a README".
3. **Token**: buka GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate. Centang **repo** dan **workflow**, lalu salin tokennya.

### Publish ke GitHub (satu perintah)
```bash
bash scripts/publish_github.sh vickarzz
```
- Saat pertama kali, skrip akan menanyakan nama dan email.
- Saat diminta **Password**, tempel **token** (bukan password GitHub). Layar memang tidak menampilkan apa-apa saat Anda mengetik.
- Skrip ini otomatis melakukan commit dan push ke `https://github.com/vickarzz/agentic-bi-analyst`.
- File data, `.venv`, dan log feedback **tidak** ikut ter-upload (sudah diatur di `.gitignore`).

Cek tab **Actions** di GitHub. Dalam sekitar 3 menit, CI akan berwarna hijau ✅ (test, evals, dan review multi-agent berjalan otomatis).

### Link demo publik (Streamlit Community Cloud, gratis)
1. Buka **share.streamlit.io**, lalu login dengan GitHub.
2. Klik **Create app**, pilih **Deploy a public app from GitHub**, lalu isi:
   - Repository: `vickarzz/agentic-bi-analyst`
   - Branch: `main`
   - Main file: `app/streamlit_app.py`
   - App URL: `agentic-bi-analyst`
3. Buka **Advanced settings**, pilih **Python 3.12**, lalu klik **Deploy**.
4. Tunggu 3–5 menit. Link Anda: `https://agentic-bi-analyst.streamlit.app`

Halaman **Business Review** otomatis ikut ter-deploy (ada di sidebar).

> Jangan masukkan API key di demo publik. Mode offline sudah cukup dan tanpa biaya.

### Update di kemudian hari
Edit file, lalu jalankan perintah yang sama:
```bash
bash scripts/publish_github.sh vickarzz
```
Streamlit akan otomatis memperbarui link demo dalam beberapa menit.

### Pasang link-nya
- **GitHub:** di halaman repo, klik ⚙️ di sebelah **About**, lalu isi Website dengan link Streamlit. Topics: `agentic-ai`, `langgraph`, `semantic-layer`, `business-intelligence`, `data-governance`, `multi-agent`
- **README:** tambahkan baris `**Live demo:** https://agentic-bi-analyst.streamlit.app` di bagian atas
- **LinkedIn:** masukkan ke bagian Featured dan Projects, lalu posting dengan video demo 30 detik (draf post ada di `docs/APPENDIX.md` bagian Q)

---

## 2b. Data Science Lab (untuk role data scientist)

Lima analisis yang biasa ditanyakan di interview data scientist. Semuanya memakai data yang sama dan dicek terhadap efek yang **sengaja ditanam** di data sintetis, sehingga kita tahu metodenya benar.

| Modul | Pertanyaan bisnis | Metode | Hasil |
|---|---|---|---|
| Causal inference | Apakah kampanye cashback Western berhasil? | Difference-in-differences, randomisation inference, event study, synthetic control | +17,0% (efek yang ditanam +18%) |
| | Apakah fee cut membuat nasabah pindah ke digital? | Interrupted time series + placebo | Metode naif +11,2%, metode benar ≈ 0 |
| Model churn | Siapa yang akan keluar 90 hari ke depan, dan apakah layak ditelepon? | Fitur point-in-time, validasi out-of-time, kalibrasi, audit fairness, hitungan ekonomi | ROC-AUC 0,92; 10% teratas menangkap 73% churner |
| Forecasting | Berapa remitansi kuartal depan? | Regresi kalender Hijriah + 20 backtest | Error 3,0% vs 11,8% metode "minggu yang sama tahun lalu" |
| A/B test | Apakah fitur one-tap layak dirilis? | SRM, CUPED, power, efek per usia, guardrail, A/A | +5,2% (tanpa CUPED tidak meyakinkan) |
| Segmentasi | Ada tipe pengirim remitansi apa saja? | K-means + stabilitas bootstrap | 2 segmen stabil (ARI 0,95) |

Cara menjalankan:
```bash
python -m bi_agent.ds                    # semua modul → reports/ds/lab_report.md + churn_model_card.md
python -m bi_agent.ds validate --strict  # 17 pemeriksaan terhadap efek yang ditanam
```
Di aplikasi, klik **Data Science Lab** di sidebar. Pertama kali dibuka perlu sekitar 20–30 detik untuk melatih model, setelah itu tersimpan di cache. Hasilnya juga ikut masuk ke Business Review sebagai **agent ke-8 (data scientist)**.

Penjelasan lengkap dan bahan interview ada di `docs/APPENDIX.md` bagian V.

### Cara update proyek di Mac (kalau sudah punya versi sebelumnya)
```bash
cd ~/Downloads && unzip -o agentic-bi-analyst-ds.zip          # ekstrak versi baru
rsync -a ~/Downloads/agentic-bi-analyst/ ~/Projects/agentic-bi-analyst/
cd ~/Projects/agentic-bi-analyst
bash scripts/setup_mac.sh --run                                # install library baru + data versi baru otomatis
bash scripts/publish_github.sh vickarzz                        # kirim ke GitHub, demo ikut ter-update
```
`rsync` menimpa file kode tapi tetap menyimpan `.git` dan `.venv` Anda. Data lama otomatis dibangun ulang karena versinya berbeda.

## 3. Cara "melatih" agent

### Konsep penting
Agent di proyek ini **tidak dilatih seperti model machine learning** (tidak ada bobot yang diubah). Di bank, itu justru disengaja: angka harus bisa diaudit dan konsisten. Agent menjadi lebih pintar dengan cara **menambah pengetahuan yang di-govern**, yaitu file YAML yang bisa direview, di-commit, dan diuji.

Ada 7 cara melatih agent, dari yang paling sering dipakai:

| # | Yang diajarkan | Contoh | Caranya |
|---|---|---|---|
| 1 | **Sinonim**: kata baru untuk metrik atau dimensi | "kredit macet" berarti NPF ratio | `python scripts/teach.py synonym npf_ratio "kredit macet" --check` |
| 2 | **Alias nilai**: kata untuk nilai filter | "kartu debit" berarti produk Mada Debit Card | `python scripts/teach.py alias "kartu debit" product "Mada Debit Card"` |
| 3 | **Golden question**: mengunci perilaku yang benar | "kredit macet per wilayah" | `python scripts/teach.py golden "kredit macet per wilayah"` |
| 4 | **Metrik baru** | Card spend, ROA, fee income per nasabah | Tambahkan blok di `semantic/semantic_layer.yaml` (contoh di APPENDIX §D) |
| 5 | **Hipotesis dampak lintas bisnis** | "Adopsi digital vs fee income" | Tambahkan blok di `semantic/impact_links.yaml` (contoh di APPENDIX §T4) |
| 6 | **Kalender bisnis** | Promo, perubahan harga, Hari Nasional | Tambahkan event di `semantic/business_calendar.yaml` |
| 7 | **Ambang alert dan domain** | NPF alert 4% (bukan 5%) | Ubah `alerts` di `bi_agent/orchestrator/domains.py` |

Setiap perubahan lewat `teach.py` **otomatis divalidasi**. Kalau membuat semantic layer rusak atau ambigu (misalnya kata yang sama dipakai untuk dua metrik), perubahan ditolak dan dibatalkan.

### Siklus training mingguan (15 menit)

```
 pakai aplikasi + beri 👍/👎  →  train_from_feedback.py  →  teach.py  →  make test + make evals  →  publish
```

1. **Kumpulkan feedback.** Pakai aplikasi di laptop. Setiap pertanyaan otomatis tercatat di `data/feedback/log.jsonl`. Klik 👍 atau 👎 di bawah jawaban.
2. **Buat laporan training:**
   ```bash
   python scripts/train_from_feedback.py
   ```
   Laporannya (`reports/training_report.md`) berisi:
   - pertanyaan yang **tidak dimengerti**, beserta kata-kata yang belum dikenal dan perintah `teach.py` yang siap dijalankan
   - jawaban 👎 yang perlu dicek (metrik, dimensi, atau periodenya salah?)
   - jawaban 👍 yang layak dijadikan golden question
   - pertanyaan yang diblokir, untuk dicek apakah ada yang sebenarnya sah
3. **Ajarkan** kata-kata yang paling sering muncul:
   ```bash
   python scripts/teach.py synonym npf_ratio "kredit macet" --check
   python scripts/teach.py synonym region "wilayah"
   python scripts/teach.py synonym transaction_count "transaksi"
   python scripts/teach.py golden "kredit macet per wilayah"
   ```
4. **Pastikan tidak ada yang rusak:** jalankan `make test` lalu `make evals`. Skor harus tetap ≥ 90%.
5. **Publish:** `bash scripts/publish_github.sh vickarzz`

Contoh nyata dari uji coba: sebelum training, pertanyaan *"kredit macet per wilayah"* tidak dimengerti. Setelah 2 sinonim diajarkan, agent menjawab NPF ratio per region, dan pertanyaan itu kini menjadi golden question sehingga tidak akan rusak lagi di masa depan.

### Kalau memakai Claude (mode LLM)
- Semua sinonim, metrik, dan alias baru **otomatis** masuk ke katalog yang dibaca Claude. Tidak ada prompt yang perlu diubah.
- Bandingkan kualitas kedua planner dengan golden question yang sama:
  ```bash
  python evals/run_evals.py --planner rules
  python evals/run_evals.py --planner llm      # butuh ANTHROPIC_API_KEY
  ```
- Claude biasanya memahami variasi kalimat yang lebih luas. Sinonim tetap berguna sebagai "kamus resmi" bank.

### Catatan untuk feedback di cloud
Di Streamlit Cloud gratis, file log akan **hilang saat aplikasi restart**. Jadi lakukan siklus training di laptop. Kalau nanti ingin mengumpulkan feedback dari pengguna publik, langkah lanjutannya adalah menyimpan log ke database atau Google Sheets.

---

## 4. Checklist akhir

- [ ] `bash scripts/setup_mac.sh --run` berhasil, aplikasi terbuka
- [ ] Screenshot: grafik tren, tab SQL, penolakan data pribadi, dan halaman Business Review
- [ ] `bash scripts/publish_github.sh vickarzz`, lalu CI hijau
- [ ] Deploy di share.streamlit.io, link demo jalan
- [ ] Link dipasang di About repo, README, dan LinkedIn Featured
- [ ] Post LinkedIn dengan video demo
- [ ] Jadwalkan 15 menit per minggu untuk siklus training
