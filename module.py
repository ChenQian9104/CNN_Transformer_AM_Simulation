import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
import math



class PositionalEmbedding(nn.Module): 

    def __init__(self, max_seq_len, embed_model_dim): 
    
        super().__init__() 
        
        self.embed_dim = embed_model_dim 
        
        pe = torch.zeros(max_seq_len, self.embed_dim) 
        
        for pos in range(max_seq_len): 
        
            for i in range(int(self.embed_dim/2)): 
                denominator = np.power(10000, 2*i/self.embed_dim) 
                
                pe[pos, 2*i] = np.sin(pos/denominator) 
                pe[pos, 2*i+1] = np.cos(pos/denominator) 
                
        pe = pe.unsqueeze(0) 
        
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        #x = x * math.sqrt(self.embed_dim) 
        
        seq_len = x.size(1) 
        
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False) 
        return x
            
            
class MultiHeadAttention(nn.Module): 
    
    def __init__(self, embed_dim = 2048, n_heads = 8, drop=0.1): 
        
        super().__init__()
        
        assert embed_dim % n_heads == 0, 'the dimension of input embedding vector must be divisble by the number of attention head'
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads 
        self.scale = (embed_dim // n_heads) ** -0.5 
        
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=True)
        
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x): 
        
        """
        Args: 
            x: the input vector in [batch_size, num_frames, seq_length, dim] or
                                   [batch_size, num_frames, dim]
        """
        if x.dim() == 4: 
            B, T, N, D = x.shape 
            qkv = self.qkv(x).reshape(B, T, 3, N, self.head_num, D // self.head_num).permute(2, 0, 1, 4, 3, 5)
            # 4 x 16 x 197 x 768 -> 4 x 16 x 197 x (768*3) -> 4 x 16 x 3 x 197 x 12 x 64 -> 3 x 4 x 16 x 12 x 197 x 64
        elif x.dim() == 3:
            B, T, D = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.head_num, D // self.head_num).permute(2, 0, 3, 1, 4)
            # 4 x 17 x 768 -> 4 x 17 x (768*3) -> 4 x 17 x 3 x 12 x 64 -> 3 x 4 x 12 x 17 x 64 

        q, k, v = qkv[0], qkv[1], qkv[2]   # 4 x 16 x 12 x 197 x 64 or 4 x 12 x 17 x 64
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale   #  4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        scores = F.softmax(attn, dim = -1)   # 4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        if x.dim == 4:
            x = torch.matmul(scores, v).permute(0, 1, 3, 2, 4).reshape(B, T, N, -1)
            # 4 x 16 x 12 x 197 x 64 -> 4 x 16 x 197 x 12 x 64 -> 4 x 16 x 197 x 768
            
        elif x.dim == 3:
            x = torch.matmul(scores, v).permute(0, 2, 1, 3).reshape(B, T, -1)
            # 4 x 12 x 17 x 64 -> 4 x 17 x 12 x 64
        
        x = self.proj(x)
        #x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    
    def __init__(self,embed_dim = 2048, n_heads = 8, expansion_factor = 4, drop=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads
        self.expansion_factor = expansion_factor
        
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        
        
        
        
        self.attention = MultiHeadAttention(embed_dim, n_heads, drop)
        
        #self.mlp = nn.Sequential(
        #    nn.Linear(embed_dim, embed_dim * expansion_factor), 
        #    nn.GELU(), 
        #    nn.Dropout(0.1),
        #    nn.Linear(embed_dim * expansion_factor, embed_dim),
        #    nn.GELU(),
        #    nn.Dropout(0.1))
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, embed_dim * expansion_factor),
                          nn.ReLU(),
                          nn.Linear(embed_dim * expansion_factor, embed_dim))
    
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, x): 
        """
        Args: 
            x: input vector [4 x 16 x 197 x 762]
        """
        
        #x = self.attention(self.norm1(x)) + x 
        attention_out = self.attention(x) + x 
        norm1_out = self.dropout1(self.norm1(attention_out))
        
        feed_forward_out = self.feed_forward(norm1_out) + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_forward_out))
        return norm2_out
        #x = self.mlp(self.norm2(x)) + x
        
        return x

class Transformer(nn.Module): 
    
    def __init__(self, embed_dim=2048, n_heads=8, expansion_factor=4, L=4, drop=0.1): 
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, n_heads, expansion_factor, drop) for _ in range(L)])
        
    def forward(self, x):
        """
        Args: 
            x: input video frames 
            4 x 16 x 197 x 192
        """
     
        for layer in self.layers: 
            x = layer(x)
            
        return x

class AMTransformer(nn.Module): 

    def __init__(self, embed_dim=(2048+512), n_heads=10, expansion_factor=4, L=4, drop=0.1, seq_length=300): 

        super().__init__()
        self.pos_embedding = PositionalEmbedding(seq_length, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1, embed_dim))
        self.transformer = Transformer(embed_dim, n_heads, expansion_factor, L, drop)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1),
            nn.ReLU()
        )

    def forward(self, x): 

        """
        Arg:
            x: input layer-wise embeeding [batch_size, seq_length, embed_dim]
        """

        B, N, D = x.shape 
        #print(x.shape)

        cls_token = self.cls_token.repeat(B, 1, 1)
        
        #x = torch.cat((cls_token, x), dim=1)   # B x N x D ==> B x (N+1) X D

        #x += self.pos_embedding[:, :(N+1), :]
        
        x = self.pos_embedding(x)

        x = self.transformer(x)   # B x N+1 x D

        x = self.mlp(x[:,0,:])

        return x


class TransformerFlashAttention(nn.Module): 

    def __init__(self, embed_dim=2304, n_heads=9, L=6, expansion_factor=4): 
        super().__init__() 
        
        self.embed_dim = embed_dim 
        self.n_heads = n_heads 
        self.expansion_factor = expansion_factor 
        
        self.encode_layer = nn.TransformerEncoderLayer(self.embed_dim, self.n_heads) 
        
        self.encoder = nn.TransformerEncoder(self.encode_layer, L) 
        
        self.cls_token = nn.Parameter(torch.randn(1,1, self.embed_dim)) 
        
        self.mlp = nn.Sequential(
                            nn.Linear(self.embed_dim, self.embed_dim * self.expansion_factor),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim * self.expansion_factor, 1))
                            
    def forward(self, x): 
        batch_size = x.size(0) 
        
        cls_token = self.cls_token.repeat(batch_size, 1, 1) 
        
        #x = torch.cat((cls_token, x), dim = 1) 
        
        encoder_out = self.encoder(x) 
        
        out = self.mlp(encoder_out[:, 0, :])
        
        return out
                            
        
class BaselineLSTM(nn.Module): 

    def __init__(self, embed_dim=2304, hidden_dim=256, expansion_factor=4): 
        super().__init__() 
        
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim 
        self.expansion_factor = expansion_factor 
        
        #self.rnn = nn.LSTM(input_size=self.embed_dim, num_layers = 2, hidden_size=self.hidden_dim,batch_first=True)
        self.rnn = nn.RNN(input_size=self.embed_dim, num_layers = 2, hidden_size=self.hidden_dim,batch_first=True)
        
        self.mlp = nn.Sequential(
                            nn.Linear(self.hidden_dim, self.hidden_dim* self.expansion_factor),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim * self.expansion_factor, 1))
                            
    def forward(self, x): 
    
        x, _ = self.rnn(x) 
        rnn_out = torch.mean(x, 1)
        
        out = self.mlp(rnn_out) 
        
        return out            
                            
        

