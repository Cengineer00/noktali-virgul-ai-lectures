# GPT (Generative Pre-trained Transformer) from Scratch

## 📚 Course Overview

This lesson demonstrates how to build a **GPT-style language model** from scratch using PyTorch. The implementation includes the complete Transformer architecture with multi-head self-attention, positional embeddings, and autoregressive text generation.

**⚠️ Note:** This project is inspired by Andrej Karpathy's [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) and adapted with Turkish narration and explanations.

### Learning Objectives
- Understand the Transformer architecture and self-attention mechanism
- Learn how to implement multi-head attention from scratch
- Master positional embeddings and their importance
- Implement autoregressive language modeling
- Train a character-level language model
- Generate coherent text using trained models

---

## 🎯 Prerequisites

### Required Libraries
```bash
pip install torch numpy
```

### Knowledge Requirements
- Strong Python and PyTorch fundamentals
- Understanding of neural networks and backpropagation
- Familiarity with attention mechanisms
- Knowledge of sequence modeling concepts
- Basic understanding of natural language processing

---

## 🔍 Code Structure

### Hyperparameters

```python
batch_size = 64        # Parallel sequences processed
block_size = 256       # Maximum context length
max_iters = 5000       # Training iterations
learning_rate = 3e-4   # Adam optimizer learning rate
n_embd = 256           # Embedding dimension
n_head = 6             # Number of attention heads
n_layer = 6            # Number of transformer blocks
dropout = 0.2          # Dropout rate for regularization
```

### Core Components

#### 1. **Head Class**
Single self-attention head:
- **Key, Query, Value projections**: Linear transformations of input
- **Attention scores**: Computed as `Q × K^T / sqrt(d_k)`
- **Causal masking**: Prevents attending to future tokens
- **Weighted aggregation**: Combines values using attention weights

**Mathematical Formula:**
```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

#### 2. **MultiHeadAttention Class**
Parallel attention heads:
- **Multiple perspectives**: Each head learns different attention patterns
- **Concatenation**: Combines outputs from all heads
- **Projection**: Linear layer to mix head outputs
- **Dropout**: Regularization to prevent overfitting

**Architecture:**
```
Input → [Head1, Head2, ..., HeadN] → Concat → Projection → Output
```

#### 3. **FeedForward Class**
Position-wise feed-forward network:
- **Expansion**: Projects to 4x embedding dimension
- **ReLU activation**: Non-linearity
- **Compression**: Projects back to embedding dimension
- **Dropout**: Regularization

**Structure:**
```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2
```

#### 4. **Block Class**
Transformer block (repeated n_layer times):
- **Self-attention**: Communication between tokens
- **Feed-forward**: Computation on individual tokens
- **Layer normalization**: Stabilizes training
- **Residual connections**: Enables deep networks

**Architecture:**
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

#### 5. **GPTLanguageModel Class**
Complete language model:
- **Token embeddings**: Converts characters to vectors
- **Position embeddings**: Encodes token positions
- **Transformer blocks**: Stack of n_layer blocks
- **Language modeling head**: Projects to vocabulary size
- **Generation**: Autoregressive text generation

---

## 🚀 Usage Example

### Training the Model

```python
import torch
from gpt_model import GPTLanguageModel

# Load and prepare text data
with open('nutuk.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character encoding/decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Initialize model
model = GPTLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # Sample batch
    xb, yb = get_batch('train')
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Generating Text

```python
# Generate from trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
```

---

## 💡 Key Concepts

### Self-Attention Mechanism
- **Purpose**: Allows each token to attend to all previous tokens
- **Causal masking**: Ensures autoregressive property (no future information)
- **Scaled dot-product**: Prevents gradient issues with large dimensions

### Positional Embeddings
- **Why needed**: Transformers have no inherent notion of position
- **Learned embeddings**: Model learns optimal position representations
- **Addition**: Combined with token embeddings before processing

### Autoregressive Generation
- **Process**: Generate one token at a time, conditioning on previous tokens
- **Sampling**: Use softmax probabilities to sample next token
- **Context window**: Limited to block_size tokens

### Layer Normalization
- **Stabilization**: Normalizes activations across features
- **Pre-norm architecture**: Applied before attention and feed-forward
- **Benefits**: Faster convergence, better gradient flow

### Residual Connections
- **Skip connections**: Add input to output of each sub-layer
- **Gradient flow**: Enables training of very deep networks
- **Formula**: `output = input + SubLayer(input)`

---

## 📊 Training Tips

1. **Data Preparation**:
   - Use large text corpus for better results
   - Character-level vs. token-level trade-offs
   - Proper train/validation split

2. **Hyperparameter Tuning**:
   - Start with smaller models for faster iteration
   - Increase `n_layer` and `n_embd` for more capacity
   - Adjust `learning_rate` based on loss curves

3. **Regularization**:
   - Dropout prevents overfitting
   - Monitor validation loss
   - Early stopping if validation loss increases

4. **Generation Quality**:
   - Temperature sampling for diversity
   - Top-k or nucleus sampling for coherence
   - Longer context windows for better coherence

---

## 🎓 Key Takeaways

1. **Transformer Architecture**: Self-attention + feed-forward + normalization + residuals
2. **Scalability**: Same architecture scales from small to GPT-3 size models
3. **Autoregressive Modeling**: Predict next token given previous context
4. **Attention Patterns**: Multi-head attention learns diverse relationships
5. **Character-Level**: Simpler than tokenization, but longer sequences

---

## 📝 Additional Files

- **`bigram.py`**: Simple bigram baseline model
- **`gpt.py`**: Training script with evaluation
- **`gpt_dev.ipynb`**: Development notebook with experiments
- **`infinite_gpt.py`**: Continuous text generation script
- **`nutuk.txt`**: Turkish text corpus for training
- **`readme.md`**: Original Turkish documentation

---

## 🔗 Resources

For a detailed video explanation of this implementation:
[Sıfırdan GPT Geliştirmek - Kodlaması ve Anlatımı](https://youtu.be/PKKKr-YMWho?si=Z2q3QoKNAdgKkV0f)

**Original Inspiration:**
- Andrej Karpathy's [Neural Networks: Zero to Hero](https://github.com/karpathy/ng-video-lecture)
- "Attention is All You Need" paper (Vaswani et al., 2017)

---

## ⚡ Next Steps

- Experiment with different architectures (more layers, larger embeddings)
- Implement byte-pair encoding (BPE) tokenization
- Add temperature and top-k sampling strategies
- Train on larger datasets (books, Wikipedia, etc.)
- Fine-tune for specific tasks (translation, summarization)
- Implement GPT-2 or GPT-3 architectural improvements
