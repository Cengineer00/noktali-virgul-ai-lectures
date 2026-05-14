# Sıfırdan GPT (Generative Pre-trained Transformer)

## 📚 Ders Özeti

Bu derste, PyTorch kullanarak sıfırdan bir **GPT tarzı dil modeli** nasıl oluşturulacağını öğreniyoruz. Uygulama, çok başlı öz-dikkat, konumsal gömülümler ve otoregresif metin üretimi ile eksiksiz Transformer mimarisini içerir.

**⚠️ Not:** Bu proje, Andrej Karpathy'nin [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) çalışmasından esinlenmiş ve Türkçe anlatım ile uyarlanmıştır.

### Öğrenme Hedefleri
- Transformer mimarisini ve öz-dikkat mekanizmasını anlamak
- Çok başlı dikkati sıfırdan uygulamayı öğrenmek
- Konumsal gömülümleri ve önemini kavramak
- Otoregresif dil modellemesini uygulamak
- Karakter seviyesinde dil modeli eğitmek
- Eğitilmiş modeller kullanarak tutarlı metin üretmek

---

## 🎯 Ön Koşullar

### Gerekli Kütüphaneler
```bash
pip install torch numpy
```

### Bilgi Gereksinimleri
- Güçlü Python ve PyTorch temelleri
- Yapay sinir ağları ve geri yayılım anlayışı
- Dikkat mekanizmalarına aşinalık
- Dizi modelleme kavramları bilgisi
- Doğal dil işleme hakkında temel anlayış

---

## 🔍 Kod Yapısı

### Hiperparametreler

```python
batch_size = 64        # Paralel işlenen diziler
block_size = 256       # Maksimum bağlam uzunluğu
max_iters = 5000       # Eğitim iterasyonları
learning_rate = 3e-4   # Adam optimize edici öğrenme oranı
n_embd = 256           # Gömülüm boyutu
n_head = 6             # Dikkat baş sayısı
n_layer = 6            # Transformer blok sayısı
dropout = 0.2          # Düzenlileştirme için dropout oranı
```

### Ana Bileşenler

#### 1. **Head Sınıfı**
Tekil öz-dikkat başı:
- **Key, Query, Value projeksiyonları**: Girişin doğrusal dönüşümleri
- **Dikkat skorları**: `Q × K^T / sqrt(d_k)` olarak hesaplanır
- **Nedensel maskeleme**: Gelecek token'lara dikkat etmeyi engeller
- **Ağırlıklı toplama**: Dikkat ağırlıklarını kullanarak değerleri birleştirir

**Matematiksel Formül:**
```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

#### 2. **MultiHeadAttention Sınıfı**
Paralel dikkat başları:
- **Çoklu bakış açıları**: Her baş farklı dikkat örüntüleri öğrenir
- **Birleştirme**: Tüm başlardan çıktıları birleştirir
- **Projeksiyon**: Baş çıktılarını karıştırmak için doğrusal katman
- **Dropout**: Aşırı öğrenmeyi önlemek için düzenlileştirme

**Mimari:**
```
Giriş → [Baş1, Baş2, ..., BaşN] → Birleştir → Projeksiyon → Çıkış
```

#### 3. **FeedForward Sınıfı**
Konum bazlı ileri beslemeli ağ:
- **Genişletme**: 4x gömülüm boyutuna projekte eder
- **ReLU aktivasyonu**: Doğrusal olmayan fonksiyon
- **Sıkıştırma**: Gömülüm boyutuna geri projekte eder
- **Dropout**: Düzenlileştirme

**Yapı:**
```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2
```

#### 4. **Block Sınıfı**
Transformer bloğu (n_layer kez tekrarlanır):
- **Öz-dikkat**: Token'lar arası iletişim
- **İleri besleme**: Bireysel token'lar üzerinde hesaplama
- **Katman normalizasyonu**: Eğitimi stabilize eder
- **Artık bağlantılar**: Derin ağları mümkün kılar

**Mimari:**
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

#### 5. **GPTLanguageModel Sınıfı**
Eksiksiz dil modeli:
- **Token gömülümleri**: Karakterleri vektörlere dönüştürür
- **Konum gömülümleri**: Token pozisyonlarını kodlar
- **Transformer blokları**: n_layer blok yığını
- **Dil modelleme başlığı**: Kelime hazinesi boyutuna projekte eder
- **Üretim**: Otoregresif metin üretimi

---

## 🚀 Kullanım Örneği

### Modeli Eğitme

```python
import torch
from gpt_model import GPTLanguageModel

# Metin verisini yükle ve hazırla
with open('nutuk.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Karakter seviyesinde kelime hazinesi oluştur
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Karakter kodlama/çözme
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Modeli başlat
model = GPTLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Eğitim döngüsü
for iter in range(max_iters):
    # Batch örnekle
    xb, yb = get_batch('train')
    
    # İleri geçiş
    logits, loss = model(xb, yb)
    
    # Geri geçiş
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Metin Üretme

```python
# Eğitilmiş modelden üret
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
```

---

## 💡 Temel Kavramlar

### Öz-Dikkat Mekanizması
- **Amaç**: Her token'in tüm önceki token'lara dikkat etmesini sağlar
- **Nedensel maskeleme**: Otoregresif özelliği garanti eder (gelecek bilgisi yok)
- **Ölçekli iç çarpım**: Büyük boyutlarda gradyan sorunlarını önler

### Konumsal Gömülümler
- **Neden gerekli**: Transformer'ların doğuştan konum kavramı yok
- **Öğrenilen gömülümler**: Model optimal konum temsillerini öğrenir
- **Toplama**: İşlemeden önce token gömülümleriyle birleştirilir

### Otoregresif Üretim
- **Süreç**: Bir seferde bir token üret, önceki token'lara koşullandır
- **Örnekleme**: Sonraki token'ı örneklemek için softmax olasılıklarını kullan
- **Bağlam penceresi**: block_size token ile sınırlı

### Katman Normalizasyonu
- **Stabilizasyon**: Özellikler boyunca aktivasyonları normalize eder
- **Pre-norm mimarisi**: Dikkat ve ileri beslemeden önce uygulanır
- **Faydalar**: Daha hızlı yakınsama, daha iyi gradyan akışı

### Artık Bağlantılar
- **Atlamalı bağlantılar**: Her alt katmanın çıkışına girişini ekler
- **Gradyan akışı**: Çok derin ağların eğitimini mümkün kılar
- **Formül**: `çıkış = giriş + AltKatman(giriş)`

---

## 📊 Eğitim İpuçları

1. **Veri Hazırlığı**:
   - Daha iyi sonuçlar için büyük metin korpusu kullan
   - Karakter seviyesi vs. token seviyesi dengesi
   - Uygun eğitim/doğrulama ayrımı

2. **Hiperparametre Ayarlama**:
   - Daha hızlı iterasyon için daha küçük modellerle başla
   - Daha fazla kapasite için `n_layer` ve `n_embd` artır
   - Kayıp eğrilerine göre `learning_rate` ayarla

3. **Düzenlileştirme**:
   - Dropout aşırı öğrenmeyi önler
   - Doğrulama kaybını izle
   - Doğrulama kaybı artarsa erken durdur

4. **Üretim Kalitesi**:
   - Çeşitlilik için sıcaklık örneklemesi
   - Tutarlılık için top-k veya nucleus örneklemesi
   - Daha iyi tutarlılık için daha uzun bağlam pencereleri

---

## 🎓 Önemli Çıkarımlar

1. **Transformer Mimarisi**: Öz-dikkat + ileri besleme + normalizasyon + artıklar
2. **Ölçeklenebilirlik**: Aynı mimari küçükten GPT-3 boyutuna kadar ölçeklenir
3. **Otoregresif Modelleme**: Önceki bağlam verildiğinde sonraki token'ı tahmin et
4. **Dikkat Örüntüleri**: Çok başlı dikkat çeşitli ilişkiler öğrenir
5. **Karakter Seviyesi**: Tokenizasyondan daha basit, ancak daha uzun diziler

---

## 📝 Ek Dosyalar

- **`bigram.py`**: Basit bigram taban modeli
- **`gpt.py`**: Değerlendirmeli eğitim scripti
- **`gpt_dev.ipynb`**: Deneylerle geliştirme notebook'u
- **`infinite_gpt.py`**: Sürekli metin üretim scripti
- **`nutuk.txt`**: Eğitim için Türkçe metin korpusu
- **`readme.md`**: Orijinal Türkçe dokümantasyon

---

## 🔗 Kaynaklar

Bu uygulamanın detaylı video açıklaması için:
[Sıfırdan GPT Geliştirmek - Kodlaması ve Anlatımı](https://youtu.be/PKKKr-YMWho?si=Z2q3QoKNAdgKkV0f)

**Orijinal İlham:**
- Andrej Karpathy'nin [Neural Networks: Zero to Hero](https://github.com/karpathy/ng-video-lecture)
- "Attention is All You Need" makalesi (Vaswani et al., 2017)

---

## ⚡ Sonraki Adımlar

- Farklı mimarilerle deney yap (daha fazla katman, daha büyük gömülümler)
- Byte-pair encoding (BPE) tokenizasyonu uygula
- Sıcaklık ve top-k örnekleme stratejileri ekle
- Daha büyük veri setlerinde eğit (kitaplar, Wikipedia, vb.)
- Belirli görevler için ince ayar yap (çeviri, özetleme)
- GPT-2 veya GPT-3 mimari iyileştirmelerini uygula
