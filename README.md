# 🧠 Yapay Zeka Destekli Hayvan Sınıflandırıcı

Bu proje, **Erzurum Teknik Üniversitesi - Bulut Bilişim ve Yapay Zeka Teknolojileri** dersi kapsamında geliştirilmiştir.  
Amaç; farklı hayvan türlerine ait görselleri sınıflandırabilen, PyTorch tabanlı bir CNN modeli geliştirip kullanıcı dostu bir Streamlit arayüzü ile sunmaktır.

Kullanıcı, bir görsel yükleyerek ilgili hayvan sınıfını tahmin edebilir. Model, eğitim verileriyle optimize edilmiş ve etkileşimli bir web arayüzüyle entegre edilmiştir.

---

## 🎯 Proje Hedefleri

- 🔍 Görüntü verilerini işleyerek etkili bir **CNN modeli** eğitmek  
- 💾 Eğitilen modeli `.pth` formatında kaydedip gerektiğinde yüklenebilir hale getirmek  
- 🖥️ Kullanıcı dostu bir **Streamlit arayüzü** geliştirmek  
- 📷 Kullanıcının yüklediği görseli sınıflandırmak ve sonucu net şekilde göstermek  

---

## 🧰 Kullanılan Teknolojiler

| Alan             | Kütüphane / Araç                      |
|------------------|----------------------------------------|
| Derin Öğrenme     | `torch`, `torchvision` (PyTorch)       |
| Görüntü İşleme    | `Pillow (PIL)`, `transforms`           |
| Web Arayüz        | `streamlit`                            |
| Yardımcılar       | `tqdm`, `os`, `random`, `matplotlib`   |
| Veri Seti         | [Animals-10 Dataset (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10) |

---

## 🧠 Model Mimarisi

`SimpleCNN` sınıfı şu katmanlardan oluşur:

```python
Conv2d(3, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d  
Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool2d  
Flatten  
Linear(64 * 32 * 32, 128) → ReLU  
Linear(128, 10)
```

Model, `train_model.ipynb` dosyasında eğitilmiş ve `simple_cnn_animals.pth` olarak kaydedilmiştir.

---

## 📦 Proje Yapısı

```
├── app.py                    # Streamlit arayüzü
├── model.ipynb               # Model tanımı
├── train_model.ipynb         # Eğitim süreci
├── data_preprocess.ipynb     # Görsellerin hazırlanması
├── rename_folders.ipynb      # Klasör isimlerini dönüştürme
├── translate.py              # Label çevirme scripti
├── requirements.txt          # Bağımlılıklar
├── simple_cnn_animals.pth    # Eğitilmiş model ağırlığı
├── raw-img/                  # Veri kümesi (görseller)
├── 📸 Ekran görüntüleri (PNG)
```

---

## 🐾 Sınıflandırılan Hayvanlar

Model şu 10 hayvan sınıfını tanımaktadır:

- 🐶 Dog  
- 🐱 Cat  
- 🐴 Horse  
- 🐘 Elephant  
- 🐄 Cow  
- 🐥 Chicken  
- 🐏 Sheep  
- 🐛 Butterfly  
- 🕷️ Spider  
- 🐿️ Squirrel  

---

## 📷 Uygulama Arayüzü

### Yükleme Ekranı:

> Uygulama, kullanıcıdan bir hayvan resmi alır ve tahmin edilen sınıfı ekranda gösterir.

```
🐾 Hayvan Sınıflandırıcı AI  
📷 Resim Yükle → 🧠 Tahmin: **DOG**
```

![Yükleme Ekranı](Ekran_goruntusu_2025-05-14_235821.png)

---

### Tahmin Sonucu:

![Tahmin Ekranı](Ekran_goruntusu_2025-05-15_000005.png)
---

### Alt Bölüm Örneği:

![Alt Bölüm](Ekran_goruntusu_2025-05-15_000034.png)

---

## 🧪 Eğitim Bilgileri

- Veri kümesi oranı: `%80 Train / %20 Validation`  
- Epoch: `10`  
- Loss fonksiyonu: `CrossEntropyLoss`  
- Optimizer: `Adam` (`lr=0.001`)  
- Batch size: `32`  

---

## 🚀 Uygulamanın Çalıştırılması

### 1. Gerekli Kütüphaneleri Kur:

```bash
pip install torch torchvision streamlit pillow tqdm
```

### 2. Uygulamayı Başlat:

```bash
streamlit run app.py
```

Tarayıcıda uygulama otomatik açılır. Görsel yükleyerek tahmin sonucunu görebilirsiniz.

---

## 👩‍💻 Hazırlayan

- **Adı Soyadı:** Şebnem Karaman  
- **Üniversite:** Erzurum Teknik Üniversitesi  
- **Ders:** Bulut Bilişim ve Yapay Zeka Teknolojileri  
- **Yıl:** 2025  

---

## 📌 Ek Notlar

- Bu proje tamamen **eğitsel** amaçla hazırlanmıştır.  
- Streamlit sayesinde uygulama **lokal tarayıcıda** kolayca çalıştırılabilir.  
- Tüm dosyalar bu repoda yer almaktadır.  
- Görseller `.png` formatında olup doğrudan README dosyasına entegre edilmiştir.

---


