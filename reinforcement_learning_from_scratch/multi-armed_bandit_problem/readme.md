# Sıfırdan Multi-Armed Bandit Problemi

Bu proje, pekiştirmeli öğrenmenin (Reinforcement Learning) temellerinden biri olan **Multi-Armed Bandit** problemini ve çözüm yöntemlerini Python ile sıfırdan inşa ettiğimiz çalışmadır.

Ajanın "keşfetme" (exploration) ve "sömürme" (exploitation) ikilemini **Epsilon-Greedy (ε-greedy)** yaklaşımıyla nasıl çözdüğünü adım adım kodluyor ve Q-değerlerinin güncellenmesini görselleştiriyoruz.

## 🚀 Nasıl Çalıştırılır?

Bu projede temel olarak farklı amaçlara hizmet eden dosyalar bulunmaktadır:

### 1. Animasyonlu Eğitim Çıktısı (Bandit Scene)
Epsilon-Greedy, Greedy veya Random gibi stratejilerin tek tek nasıl çalıştığını ve ajanın nasıl karar verdiğini adım adım görmek için:
```bash
python bandit_scene.py
```
*(Eğer MP4 olarak kaydetmek isterseniz: `python bandit_scene.py --save out.mp4`)*

### 2. Deneyler ve Grafikler (Bandit Dashboard)
Çoklu deneme (multi-run) sonuçlarının grafiklerini statik olarak çizdirmek ve hemen ardından genel performans gelişiminin canlı bir animasyonunu görmek için:
```bash
python bandit.py
```
*(Eğer MP4 olarak kaydetmek isterseniz: `python bandit.py --save out.mp4`)*

### 3. İnteraktif Eğitim (Notebook)
Kodun adım adım nasıl çalıştığını, matematiğini ve çıktılarını notebook ortamında incelemek isterseniz, `youtube_notebook.ipynb` dosyasını kullanabilirsiniz.

---
**Not:** Algoritmanın temel matamatiği ve `Agent`, `BanditEnv` gibi yapılar `bandit_core.py` içerisinde barındırılmaktadır. Görselleştirme araçları bu çekirdek (core) kodlardan beslenmektedir.

## 🎥 Video Anlatımı
Bu projenin detaylı YouTube anlatımı için (Çok Yakında):
[Sıfırdan Multi-Armed Bandit]()
