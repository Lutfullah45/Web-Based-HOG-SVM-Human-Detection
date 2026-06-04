# Web-Based HOG + SVM Human Detection 🚶‍♂️🔍

Bu proje, bilgisayarlı görü (computer vision) literatüründe standart bir temel (baseline) kabul edilen **Yönelimli Gradyanların Histogramı (HOG)** ve **Doğrusal Destek Vektör Makineleri (Linear SVM)** algoritmaları kullanılarak geliştirilmiş, uçtan uca (end-to-end) web tabanlı bir insan tespit sistemidir.

Proje, modelin sadece komut satırında çalışmasını aşarak, **FastAPI** ve **HTML5 Canvas** mimarisiyle son kullanıcının doğrudan etkileşime girebileceği modern bir istemci-sunucu (client-server) altyapısına taşınmıştır.

## ✨ Öne Çıkan Mühendislik Özellikleri

* **Zorlu Negatif Avı (Hard-Negative Mining):** Modelin ağaç gövdeleri veya dikey direkler gibi karmaşık arka planları insan sanması (False Positive) problemini aşmak için, modelin hatalarından kendi kendine öğrendiği iteratif bir eğitim stratejisi uygulanmıştır.
* **Çoklu Kutu Çakışması Engelleme (NMS):** Kayan pencere (sliding window) taramasında aynı kişi üzerinde oluşan çoklu aday kutular, En Büyük Olmayanı Bastırma (Non-Maximum Suppression) optimizasyonu ile filtrelenerek tekillik sağlanmıştır.
* **Asenkron REST API:** Python tabanlı yüksek performanslı FastAPI framework'ü kullanılarak, istemciden gelen anlık görüntüler milisaniyeler içerisinde analiz edilip sonuçlandırılır.
* **Görsel HOG Analizi:** Sistem sadece tespit koordinatlarını değil, akademik analiz imkanı sunmak adına görüntünün çıkartılan gradyan/HOG haritasını da Base64 formatında eşzamanlı olarak arayüze aktarır.

## 📂 Klasör Yapısı

Depo (Repository) aşağıdaki modüler yapıya sahiptir:

* `app/`: FastAPI sunucu kodlarını ve API uç noktalarını (`api_sunucu.py`) barındırır.
* `scripts/`: Veri seti hazırlama (64x128 kırpma/padding), model eğitimi ve Zorlu Negatif avcısı Python betiklerini içerir.
* `Webui/`: Sistemin önyüzünü oluşturan HTML ve JavaScript dosyalarını barındırır (`index.html`). Tespit işlemleri doğrudan Canvas API ile tarayıcıda çizilir.
* `tools/`: Çıkarım (inference) ve test süreçleri için geliştirilmiş çeşitli yardımcı araçlar.
* `baseline_work/`: Eğitilmiş Linear SVM model ağırlıkları (`.joblib`) ve önceden işlenmiş öznitelik dosyaları.

## 🛠️ Kurulum ve Çalıştırma

Projeyi kendi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Depoyu Klonlayın:**
    git clone https://github.com/Lutfullah45/Web-Based-HOG-SVM-Human-Detection.git
    cd Web-Based-HOG-SVM-Human-Detection

**2. Gerekli Kütüphaneleri Yükleyin:**
Python yüklü olduğundan emin olduktan sonra terminalde şu komutu çalıştırın:
    pip install fastapi uvicorn opencv-python scikit-learn numpy imageio

**3. API Sunucusunu Ayağa Kaldırın:**
FastAPI sunucusunu başlatmak için ana dizinde şu komutu çalıştırın:
    uvicorn app.api_sunucu:app --reload

Sunucu başarıyla başladığında terminalde "Uvicorn running on http://127.0.0.1:8000" mesajını göreceksiniz.

**4. Arayüzü Açın:**
Dosya gezgininden `Webui` klasörünün içindeki `index.html` dosyasına çift tıklayarak tarayıcınızda açın. Sisteme bir fotoğraf yükleyip tespit sürecini doğrudan test edebilirsiniz!

## 💻 Kullanılan Teknolojiler
* **Makine Öğrenmesi & Görüntü İşleme:** OpenCV, Scikit-Learn, Numpy
* **Arka Yüz (Backend):** Python, FastAPI, Uvicorn
* **Ön Yüz (Frontend):** HTML5, Vanilla JavaScript, Canvas API

---
**Geliştirici:** Lütfullah Dinçer  
*Bilecik Şeyh Edebali Üniversitesi - Bilgisayar Mühendisliği Bitirme Tasarım Projesi*
