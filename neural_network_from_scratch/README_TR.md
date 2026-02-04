# Sıfırdan Yapay Sinir Ağı (Neural Network)

## 📚 Ders Özeti

Bu derste, PyTorch, TensorFlow veya Keras gibi derin öğrenme framework'leri kullanmadan, sadece Python ve NumPy ile sıfırdan bir **Yapay Sinir Ağı** nasıl oluşturulacağını öğreniyoruz. Uygulama, ileri yayılım, geri yayılım ve MNIST ile CIFAR10 veri setlerinde eğitim içerir.

### Öğrenme Hedefleri
- Yapay sinir ağı mimarisinin temellerini anlamak
- İleri ve geri yayılım algoritmalarını öğrenmek
- Aktivasyon fonksiyonlarını uygulamak (ReLU, Softmax)
- Gradyan inişi optimizasyonunda uzmanlaşmak
- Ağı gerçek dünya görüntü sınıflandırma görevlerine uygulamak

---

## 🎯 Ön Koşullar

### Gerekli Kütüphaneler
```bash
pip install numpy torchvision jupyter
```
**Not:** `torchvision` sadece CIFAR10 veri setini indirmek için kullanılır, sinir ağı uygulaması için değil.

### Bilgi Gereksinimleri
- Sağlam Python programlama becerileri
- Güçlü NumPy işlemleri anlayışı
- Lineer cebir temelleri (matris çarpımı, iç çarpım)
- Kalkülüs temelleri (türevler, zincir kuralı)
- Makine öğrenimi kavramlarına temel anlayış

---

## 🔍 Kod Yapısı

### Ana Bileşenler

#### 1. **LinearLayer Sınıfı**
Tamamen bağlı (dense) katman uygular:
- **Başlatma**: Ağırlıklar için He başlatma kullanır (`W ~ N(0, sqrt(2/n_in))`)
- **`forward(input_data)`**: `çıkış = giriş × W + b` hesaplar
- **`backward(output_gradient, learning_rate)`**: Gradyanları hesaplar ve ağırlıkları/bias'ları günceller

**Matematiksel İşlemler:**
```
İleri:  Z = X × W + b
Geri:   ∂W = X^T × ∂Z
        ∂X = ∂Z × W^T
```

#### 2. **ReLULayer Sınıfı**
Doğrultulmuş Doğrusal Birim aktivasyonu:
- **`forward(input_data)`**: `max(0, input_data)` döndürür
- **`backward(output_gradient, learning_rate)`**: Gradyanı sadece pozitif girişler için geçirir

**Formül:**
```
ReLU(x) = max(0, x)
∂ReLU/∂x = 1 eğer x > 0, yoksa 0
```

#### 3. **SoftmaxLayer Sınıfı**
Logit'leri olasılık dağılımına dönüştürür:
- **`forward(input_data)`**: Softmax normalizasyonu uygular
- **`backward(output_gradient, learning_rate)`**: Gradyanı geçirir (cross-entropy ile birleşik)

**Formül:**
```
Softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

#### 4. **CrossEntropyLoss Sınıfı**
Sınıflandırma kaybını hesaplar:
- **`forward(predictions, targets)`**: Negatif log-likelihood hesaplar
- **`backward()`**: Gradyan hesaplar: `(tahminler - one_hot_hedefler) / batch_boyutu`

**Formül:**
```
Kayıp = -1/m × Σ log(p_doğru_sınıf)
```

#### 5. **NeuralNetwork Sınıfı**
Ana ağ mimarisi:
- **Yapı**: Giriş → Linear → ReLU → Linear → ReLU → Softmax
- **`forward(x)`**: Tüm katmanlardan ileri geçiş yapar
- **`backward(loss_grad, learning_rate)`**: Gradyanları geri yayılım yapar ve ağırlıkları günceller

---

## 🚀 Kullanım Örneği

### MNIST Üzerinde Eğitim

```python
from neural_network import NeuralNetwork
import numpy as np

# MNIST verisini yükle (28x28 gri tonlamalı görüntüler, 10 sınıf)
X_train = X_train.reshape(-1, 784) / 255.0  # Düzleştir ve normalize et
Y_train = Y_train.astype(int)

# Ağı başlat
model = NeuralNetwork(
    input_size=784,   # 28x28 piksel
    hidden_size=128,  # Gizli katman nöronları
    output_size=10    # 10 rakam sınıfı
)

# Eğitim döngüsü
for epoch in range(epochs):
    # İleri geçiş
    tahminler = model.forward(X_batch)
    
    # Kayıp hesapla
    loss = model.loss_function.forward(tahminler, Y_batch)
    
    # Geri geçiş
    loss_gradient = model.loss_function.backward()
    model.backward(loss_gradient, learning_rate)
```

### CIFAR10 Üzerinde Eğitim

Benzer yaklaşım ancak:
- Giriş boyutu: 3072 (32x32x3 RGB görüntüler)
- Daha karmaşık mimari önerilir
- Daha iyi performans için veri artırma

---

## 💡 Temel Kavramlar

### İleri Yayılım (Forward Propagation)
Veri katmanlardan sırayla akar:
```
Giriş → Linear1 → ReLU → Linear2 → ReLU → Softmax → Çıkış
```

### Geri Yayılım (Backpropagation)
Gradyanlar zincir kuralı kullanarak geriye akar:
```
Kayıp ← Softmax ← ReLU ← Linear2 ← ReLU ← Linear1 ← Giriş
```

### Ağırlık Başlatma
- **He Başlatma**: ReLU ağlarında kaybolan/patlayan gradyanları önler
- Formül: `W ~ N(0, sqrt(2 / n_giriş))`

### Gradyan İnişi
- Ağırlıkları kaybı azaltan yönde günceller
- Formül: `W_yeni = W_eski - öğrenme_oranı × ∂Kayıp/∂W`

---

## 📊 Performans İpuçları

1. **Normalizasyon**: Giriş verisini [0, 1] aralığına ölçekle
2. **Batch İşleme**: Birden fazla örneği eş zamanlı işle
3. **Öğrenme Oranı**: 0.001-0.01 ile başla, yakınsamaya göre ayarla
4. **Gizli Katman Boyutu**: Kapasite ve aşırı öğrenme arasında denge
5. **Epoch'lar**: Aşırı öğrenmeyi önlemek için doğrulama kaybını izle

---

## 🎓 Önemli Çıkarımlar

1. **Katman Soyutlama**: Her katman kendi ileri/geri işlemlerini yönetir
2. **Otomatik Türev Alma**: Geri yayılım gradyanları otomatik hesaplar
3. **Modüler Tasarım**: Yeni katmanlar veya aktivasyon fonksiyonları eklemek kolay
4. **Saf NumPy**: Düşük seviye işlemleri anlamak güçlü temeller oluşturur
5. **Ölçeklenebilirlik**: Aynı prensipler daha derin, daha karmaşık ağlara uygulanır

---

## 📝 Ek Dosyalar

- **`neural_network_mnist.ipynb`**: Eksiksiz MNIST eğitim notebook'u
- **`neural_network_cifar10.ipynb`**: Görselleştirmelerle CIFAR10 eğitimi
- **`readme.md`**: Orijinal Türkçe dokümantasyon

---

## 🔗 Kaynaklar

Bu uygulamanın detaylı video açıklaması için:
[Sıfırdan Yapay Sinir Ağı Geliştirdim](https://youtu.be/witsTpml9YM?si=ZXiH-ehFT5fVM1uH)

---

## ⚡ Sonraki Adımlar

- Farklı mimarilerle deney yap (daha fazla katman, farklı boyutlar)
- Diğer aktivasyon fonksiyonlarını uygula (Sigmoid, Tanh, Leaky ReLU)
- Düzenlileştirme teknikleri ekle (L2, Dropout)
- Farklı optimize ediciler dene (Momentum, Adam)
- Diğer veri setlerine uygula
