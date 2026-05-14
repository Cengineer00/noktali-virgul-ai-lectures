# Sıfırdan Karar Ağacı (Decision Tree)

## 📚 Ders Özeti

Bu derste, sadece Python ve NumPy kullanarak sıfırdan bir **Karar Ağacı Sınıflandırıcı** nasıl oluşturulacağını öğreniyoruz. Uygulama, hem kategorik hem de sayısal özellikleri destekler ve optimal bölme seçimi için Gini safsızlık metriğini kullanır.

### Öğrenme Hedefleri
- Karar ağacı algoritmalarının temellerini anlamak
- Düğüm bölme için Gini safsızlığını hesaplamayı öğrenmek
- Uygun durdurma kriterleriyle özyinelemeli ağaç oluşturmayı uygulamak
- Hem kategorik hem de sayısal özellik türlerini işlemek
- Özel çizim fonksiyonlarıyla karar ağaçlarını görselleştirmek

---

## 🎯 Ön Koşullar

### Gerekli Kütüphaneler
```bash
pip install numpy matplotlib
```

### Bilgi Gereksinimleri
- Temel Python programlama
- NumPy dizileri hakkında bilgi
- Sınıflandırma kavramlarına aşinalık
- Özyineleme (recursion) hakkında temel bilgi

---

## 🔍 Kod Yapısı

### Ana Bileşenler

#### 1. **Node Sınıfı**
Karar ağacındaki her düğümü temsil eder:
- Veri, alt düğümler ve bölme bilgilerini saklar
- Hem yaprak düğümleri (tahminler) hem de karar düğümlerini (bölmeler) işler
- Sürekli özellikler için sayısal eşik değerlerini destekler

#### 2. **DecisionTreeClassifier Sınıfı**
Ana sınıflandırıcı uygulaması:
- **`fit(X, Y)`**: Ağacı özyinelemeli olarak oluşturarak modeli eğitir
- **`find_best_split(node)`**: Bölme için optimal özellik ve eşik değerini bulur
- **`calculate_gini_impurity(Y)`**: Düğüm değerlendirmesi için Gini safsızlığını hesaplar
- **`split_on_categorical_feature()`**: Kategorik özellik bölmelerini işler
- **`split_on_numerical_feature()`**: Eşik seçimiyle sayısal özellik bölmelerini işler
- **`predict(X)`**: Tahmin yapmak için ağacı dolaşır
- **`plot_tree()`**: Karar ağacı yapısını görselleştirir

### Temel Algoritmalar

**Gini Safsızlık Hesaplama:**
```
Gini = 1 - Σ(p_i²)
```
Burada p_i, düğümdeki i sınıfının olasılığıdır.

**Bölme Stratejisi:**
- Kategorik özellikler için: Birden fazla dala bölme (her benzersiz değer için bir dal)
- Sayısal özellikler için: Orta nokta değerlendirmesi kullanarak optimal eşik bulma
- Minimum ağırlıklı Gini safsızlığına sahip bölmeyi seçme

**Durdurma Kriterleri:**
- Saf düğüm (Gini safsızlığı = 0)
- Bölünecek özellik kalmadı
- Geçerli bölme bulunamadı

---

## 🚀 Kullanım Örneği

`main.py` dosyası tıbbi teşhis senaryosunu gösterir:

```python
from tree import DecisionTreeClassifier
import numpy as np

# Tıbbi semptom verisi (Öksürük, Koku, Sıcaklık, Teşhis)
data = np.array([
    ["Öksürüyor", "Koku Alabiliyor", 39, 0],
    ["Öksürüyor", "Koku Alamıyor", 37.8, 1],
    # ... daha fazla örnek
])

X, Y = data[:, :-1], data[:, -1]

# Karışık özellik türleriyle modeli eğit
model = DecisionTreeClassifier(
    feature_types=["categorical", "categorical", "numerical"]
)
model.fit(X, Y)

# Tahmin yap
tahminler = model.predict(test_ornekleri)

# Ağacı görselleştir
model.plot_tree(
    feature_names=["Öksürük Durumu", "Koku Alma", "Vücut Sıcaklığı"],
    class_names={0: "Negatif", 1: "Pozitif"},
    save_path="decision_tree.png"
)
```

---

## 📊 Görselleştirme Özellikleri

`plot_tree()` metodu profesyonel ağaç diyagramları oluşturur:
- **Mavi kutular**: Bölme koşullarını gösteren karar düğümleri
- **Yeşil/Kırmızı kutular**: Tahminleri gösteren yaprak düğümleri
- **Kenar etiketleri**: Bölme değerleri veya koşulları
- **Düğüm bilgileri**: Örnek sayıları, sınıf dağılımı, Gini safsızlığı

---

## 🎓 Önemli Çıkarımlar

1. **Özyinelemeli Yapı**: Karar ağaçları optimal bölmeler bularak özyinelemeli olarak oluşturulur
2. **Safsızlık Metrikleri**: Gini safsızlığı bölme sürecine rehberlik eder
3. **Karışık Veri Türleri**: Uygulama hem kategorik hem de sayısal özellikleri işler
4. **Aşırı Öğrenme Önleme**: Durdurma kriterleri aşırı ağaç derinliğini önler
5. **Yorumlanabilirlik**: Ağaç görselleştirmesi modelin kararlarını şeffaf hale getirir

---

## 📝 Ek Dosyalar

- **`test_iris.py`**: Sınıflandırıcıyı Iris veri setinde test eder
- **`test_iris_sklearn.py`**: Uygulamayı scikit-learn ile karşılaştırır

---

## 🔗 Kaynaklar

Bu uygulamanın detaylı video açıklaması için [Noktali Virgul YouTube kanalını](https://www.youtube.com/@noktalıvirgul) ziyaret edin.
