import hyperparams as hp
from module2 import *
from utils import get_positional_table, get_sinusoid_encoding_table

class SpatialAttention(nn.Module):
    def __init__(self, num_hidden):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(num_hidden, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attn_weights = self.softmax(self.conv(x))
        return x * attn_weights

class TemporalAttention(nn.Module):
    def __init__(self, num_hidden):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(num_hidden, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn_weights = self.softmax(self.attn(x))
        return x * attn_weights

class Encoder(nn.Module):
    def __init__(self, embedding_size, num_hidden):
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(t.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0), freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)
        self.spatial_attn = SpatialAttention(num_hidden)
        self.temporal_attn = TemporalAttention(num_hidden)
        self.layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)

    def forward(self, x, pos):
        pos = self.pos_emb(pos)
        x = self.pos_dropout(pos * self.alpha + x)
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        attns = []
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x)
            x = ffn(x)
            attns.append(attn)
        return x, attns

class MelDecoder(nn.Module):
    def __init__(self, num_hidden):
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0), freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.selfattn_layers = clones(Attention(num_hidden), 3)
        self.dotattn_layers = clones(Attention(num_hidden), 3)
        self.ffns = clones(FFN(num_hidden), 3)
        self.mel_linear = Linear(num_hidden, hp.num_mels * hp.outputs_per_step)
        self.postconvnet = PostConvNet(num_hidden)

    def forward(self, memory, decoder_input, c_mask, pos):
        pos = self.pos_emb(pos)
        decoder_input = self.pos_dropout(pos * self.alpha + self.decoder_prenet(decoder_input))
        attn_dot_list = []
        attn_dec_list = []
        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input)
            decoder_input, attn_dot = dotattn(memory, decoder_input)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)
        mel_out = self.mel_linear(decoder_input)
        out = self.postconvnet(mel_out.transpose(1, 2)).transpose(1, 2)
        return mel_out, out, attn_dot_list, attn_dec_list

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)
        self.decoder = MelDecoder(hp.hidden_size)

    def forward(self, vid, mel_input, pos_vid, pos_mel):
        memory, attns_enc = self.encoder(vid, pos=pos_vid)
        mel_output, postnet_output, attn_probs, attns_dec = self.decoder(memory, mel_input, pos=pos_mel)
        return mel_output, postnet_output, attn_probs, attns_enc, attns_dec
