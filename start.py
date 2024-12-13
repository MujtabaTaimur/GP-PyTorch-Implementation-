import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# --- Simple Tokenizer (for demonstration purposes) ---
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
    
    def encode(self, text):
        """Convert text to list of token ids."""
        return [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
    
    def decode(self, token_ids):
        """Convert token ids back to text."""
        return ' '.join([self.inv_vocab.get(token_id, '<unk>') for token_id in token_ids])

# Sample vocab (simplified)
vocab = {'hello': 1, 'how': 2, 'are': 3, 'you': 4, '<unk>': 0}
tokenizer = SimpleTokenizer(vocab)

# --- Scaled Dot-Product Attention ---
def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention."""
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))  # QK^T
    d_k = query.size()[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # Softmax across last dimension
    output = torch.matmul(attention_weights, value)  # Weighted sum of values
    return output, attention_weights

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.output_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the input into multiple heads."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split the query, key, and value into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Compute attention
        output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        # Concatenate the attention heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        return self.output_dense(output)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Transformer Layer ---
class GPTLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(GPTLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output = self.multihead_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed Forward Network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

# --- GPT-like Model ---
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000):
        super(GPT, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([GPTLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.output_layer(x)

# --- Example Usage ---
if __name__ == "__main__":
    # Define model parameters
    vocab_size = len(vocab)  # Vocabulary size
    d_model = 128  # Dimensionality of the embedding and hidden layers
    num_heads = 8  # Number of attention heads
    num_layers = 6  # Number of transformer layers
    d_ff = 512  # Feed-forward layer size

    # Instantiate the GPT model
    model = GPT(vocab_size, d_model, num_heads, num_layers, d_ff)

    # Tokenize input text
    text = "hello how are you"
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    output = model(input_tensor)

    # Decode the predicted tokens (just the first token for simplicity)
    predicted_tokens = torch.argmax(output, dim=-1)
    decoded_output = tokenizer.decode(predicted_tokens[0].tolist())

    print(f"Input: {text}")
    print(f"Predicted: {decoded_output}")
