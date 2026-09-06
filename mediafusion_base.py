import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.models.video as models
from torchvision.models.swin_transformer import SwinTransformerBlock

from impl.head_cls import GroupedConvHead


class VectorGatedShift(nn.Module):
    """
    Applies a gated temporal shift to 1D feature vectors (B, T, C).
    This is inspired by the GSF module but adapted for pre-extracted features
    without spatial dimensions.
    """
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim = dim

        self.n_div = n_div
        self.fold_dim = dim // n_div

        # A small network to learn the gates dynamically from input
        # Using Conv1d to capture a small local temporal context (kernel_size=3)
        self.gate_network = nn.Sequential(
            nn.Conv1d(self.fold_dim * 2, self.fold_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.fold_dim, 2, kernel_size=1), # 2 output channels for forward/backward gates
            nn.Tanh()
        )
        
        # A small network to learn the fusion weights (how to combine shifted and residual)
        self.fusion_weight_network = nn.Sequential(
            nn.Linear(self.fold_dim * 2, self.fold_dim // 2),
            nn.ReLU(),
            nn.Linear(self.fold_dim // 2, 2), # 2 weights: 1 for shifted, 1 for residual
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Input shape: (B, T, C) where T is n_segment
        b, t, c = x.shape
        
        # Part that remains unchanged
        unaffected_part = x[:, :, self.fold_dim * 2:]

        # Part that will be processed
        gated_part = x[:, :, :self.fold_dim * 2]

        # Learn gates from the data
        # (B, T, C) -> (B, C, T) for Conv1D
        gates = self.gate_network(gated_part.permute(0, 2, 1)).permute(0, 2, 1) # Shape: (B, T, 2)
        gate_fwd, gate_bwd = gates.chunk(2, dim=-1) # Shapes: (B, T, 1) and (B, T, 1)

        # Split into two groups for forward and backward shift
        group1, group2 = gated_part.chunk(2, dim=-1) # Shape: (B, T, fold_dim)

        # Apply gates
        gated_group1 = gate_fwd * group1
        gated_group2 = gate_bwd * group2
        
        # The residual (un-gated) part
        residual_group1 = group1 - gated_group1
        residual_group2 = group2 - gated_group2

        # Perform temporal shift
        shifted_group1 = torch.roll(gated_group1, shifts=-1, dims=1)
        shifted_group1[:, -1, :] = 0 # Zero-pad the last frame
        shifted_group2 = torch.roll(gated_group2, shifts=1, dims=1)
        shifted_group2[:, 0, :] = 0 # Zero-pad the first frame

        # Learn fusion weights
        # Weights for group 1
        weights_input1 = torch.cat([shifted_group1, residual_group1], dim=-1)
        fusion_weights1 = self.fusion_weight_network(weights_input1)
        w_shifted1, w_residual1 = fusion_weights1.chunk(2, dim=-1)
        
        # Weights for group 2
        weights_input2 = torch.cat([shifted_group2, residual_group2], dim=-1)
        fusion_weights2 = self.fusion_weight_network(weights_input2)
        w_shifted2, w_residual2 = fusion_weights2.chunk(2, dim=-1)

        # Fuse the shifted and residual parts
        fused_group1 = shifted_group1 * w_shifted1 + residual_group1 * w_residual1
        fused_group2 = shifted_group2 * w_shifted2 + residual_group2 * w_residual2
        
        # Concatenate all parts back together
        out = torch.cat([fused_group1, fused_group2, unaffected_part], dim=-1)
        return out

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x có shape (batch, seq_len, dim)
        pe_tensor = torch.as_tensor(self.pe)
        pe_slice = pe_tensor[:, :x.size(1), :].to(x.device)
        x = x + pe_slice
        return self.dropout(x)

class DilatedConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Chồng các lớp Conv với dilation tăng theo cấp số nhân
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4)
        self.norm = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x_in -> conv1 -> conv2 -> conv3 -> x_out
        # Có thể thêm residual connection trong khối này
        x_in = x
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        x = self.relu(self.norm(self.conv3(x)))
        return x + x_in

class BaseModel(nn.Module):
    """
    MediaFusion base model class
    """
    def __init__(
        self,
        chunk_size = 8,
        n_output = 24,
        baidu = True,
        audio = False,
        model_cfg = None
    ):

        super().__init__()

        self.chunk_size = chunk_size
        self.n_output = n_output
        self.baidu = baidu
        self.audio = audio
        self.model_cfg = model_cfg

        #Baidu backbone
        if self.baidu:
            self.Bfeat_dim = [2048, 2048, 384, 2048, 2048, 768]
            self.baidu_LL = nn.ModuleList([Bfeat_module(self.Bfeat_dim[i], model_cfg['dim'], drop = model_cfg['dropout']) for i in range(len(self.Bfeat_dim))])
            # self.encTB = nn.Parameter(torch.rand(self.chunk_size, model_cfg['dim']))
            self.gated_shift_modules = nn.ModuleList([
                VectorGatedShift(dim=model_cfg['dim'], n_div=8)
                for _ in range(len(self.Bfeat_dim))
            ])
            
            self.video_pos_encoder = PositionalEncoding(d_model=model_cfg['dim'], max_len= self.n_output)
            self.encFB = nn.Parameter(torch.rand(len(self.Bfeat_dim), model_cfg['dim']))

        #Audio backbone
        if self.audio:
            self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.vggish_features = self.vggish.features
            self.vggish_embeddings = self.vggish.embeddings
            self.vggish_embeddings[0] = nn.Linear(24576, 4096)
            self.vggish_embeddings[4] = nn.Linear(4096, model_cfg['dim'])
            self.encTA = nn.Parameter(torch.rand(self.chunk_size * 100 // 96, model_cfg['dim']))
            self.encA = nn.Parameter(torch.rand(model_cfg['dim']))
            del self.vggish

        #Features augmentation (for both audio + baidu)
        if model_cfg['feature_augmentation']:
            if self.baidu:
                self.temporal_dropB = temporal_dropM(p = model_cfg['temporal_drop_p'])
            if self.audio:
                self.temporal_dropA = temporal_dropM(p = model_cfg['temporal_drop_p'], dim = 128)
            self.random_switch = random_switchM(p = model_cfg['random_switch_p'])

        # MediaFusion model
    
        self.encoder_layers = nn.ModuleList()
        num_layers = model_cfg.get('TE_layers', 8)
        dim = model_cfg.get('dim', 512)
        num_heads = model_cfg.get('nhead', 8)
        
        for _ in range(num_layers):
            self.encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout= 0.1,
                    batch_first=True
                )
            )
        
        
        #Transformer decoder
        if model_cfg['queries'] is not None:
            print("=> loading queries '{}'".format(model_cfg['queries']))
            self.queries = nn.Parameter(torch.from_numpy(np.load(model_cfg['queries'])).float(), requires_grad = True)
        else:
            self.queries = nn.Parameter(torch.rand((self.n_output, model_cfg['dim'])))
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_cfg['dim'], nhead=8, dim_feedforward=model_cfg['dim'] * 4, batch_first=True)
        self.Tdecoder = nn.TransformerDecoder(decoder_layer, model_cfg['TD_layers'])

        self.residual_transform = nn.Sequential(
            DilatedConvBlock(model_cfg['dim']),
            nn.AdaptiveAvgPool1d(self.n_output)
        )
        self.fusion_layer = nn.Linear(model_cfg['dim'] * 2, model_cfg['dim'])
        #Prediction heads
        self.clas_head = GroupedConvHead(input_dim = model_cfg['dim'])

        if model_cfg['uncertainty']:
            self.displ_head = uncertainty_head(model_cfg['dim'], model_cfg['num_classes'] + 1, drop = model_cfg['dropout'])
        else:
            self.displ_head = GroupedConvHead()


        #Mixup queue system
        if model_cfg['mixup']:
            #Initialize queues (for labels and inputs)
            self.register_buffer('labelQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1))
            self.labelQ[:, :, :, 0] = 1
            self.register_buffer('labelDQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.n_output, model_cfg['num_classes'] + 1) + 1000)

            self.register_buffer('featBQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size, 9344))
            if self.audio:
                self.register_buffer('featAQ', torch.zeros(model_cfg['num_classes'] + 1, model_cfg['mixup_nqueue'], self.chunk_size * 100, 128))
            else:
                self.featAQ = None

            self.do_mixup = mixupYolo(alpha = model_cfg['mixup_alpha'], beta = model_cfg['mixup_beta'])

    def forward(self, featsB = None, featsA = None, labels = None, labelsD = None, inference = False):
        
        #Float type data
        if labels != None:
            labels = labels.float()
        if labelsD != None:
            labelsD = labelsD.float()
        if self.baidu:
            featsB = featsB.float()
            b = len(featsB)

        if self.audio and featsA is not None:
            featsA = featsA.float()
            #log mel spectrogram
            # featsA = 10.0 * (torch.log10(torch.maximum(torch.tensor(1e-10), featsA)) - torch.log10(torch.tensor(7430.77))) #7430.77 is the maximum value of the log mel spectrogram
            # featsA = torch.maximum(featsA, featsA.max() - 80)
            featsA = rearrange(featsA, 'b f h -> b f h')

        #Mixup (not inference)
        if self.model_cfg['mixup'] & (not inference):
            y = labels.clone()
            yD = labelsD.clone()
            if self.baidu:
                xB = featsB.clone()
            if self.audio and featsA is not None:
                xA = featsA.clone()

            featsB, featsA, labels, labelsD = self.do_mixup(featsB, self.featBQ, featsA, self.featAQ, labels, self.labelQ, labelsD, self.labelDQ)

            #Update mixup queue
            batch_action = (y[:, :, :] == 1).sum(1).nonzero() #(batch, action) pairs of clip with action
                
            for i in range(self.model_cfg['num_classes']+1):
                aux = batch_action[batch_action[:, 1] == i] #idxs containing action i

                if len(aux) >= self.model_cfg['mixup_nqueue']:
                    idx = aux[:self.model_cfg['mixup_nqueue'], 0] #keep first ones
                    self.labelQ[i, :] = y[idx].clone().detach()
                    self.labelDQ[i, :] = yD[idx].clone().detach()
                    if self.baidu:
                        self.featBQ[i, :] = xB[idx].clone().detach()
                    if self.audio and featsA is not None:
                        self.featAQ[i, :] = xA[idx].clone().detach()

                elif len(aux) > 0:
                    idx1 = torch.randint(0, self.model_cfg['mixup_nqueue'], (len(aux),), device = 'cuda:0')
                    idx = aux[:, 0]
                    self.labelQ[i, idx1] = y[idx].clone().detach()
                    self.labelDQ[i, idx1] = yD[idx].clone().detach()
                    if self.baidu:
                        self.featBQ[i, idx1] = xB[idx].clone().detach()
                    if self.audio and featsA is not None:
                        self.featAQ[i, idx1] = xA[idx].clone().detach()

        #DATA AUGMENTATIONS + PREPROCESSING BEFORE TE (INCLUDING POSITIONAL ENCODING)

        #Baidu features
        feature_sequence_list = []

        # Baidu features
        if self.baidu:
            if self.model_cfg.get('feature_augmentation', False) and not inference:
                featsB = self.temporal_dropB(featsB)
                featsB = self.random_switch(featsB)
            
            # PFFN
            # featsB có shape (b, cs, total_feat_dim)
            featsB_list = [self.baidu_LL[i](featsB[:, :, int(torch.tensor(self.Bfeat_dim[:i]).sum()):int(torch.tensor(self.Bfeat_dim[:i+1]).sum())]) for i in range(len(self.Bfeat_dim))]
            shifted_feats_list = [self.gated_shift_modules[i](featsB_list[i]) for i in range(len(featsB_list))]

            # Stack các feature lại
            x_grid  = torch.stack(shifted_feats_list, dim=2)  # Shape: (b, cs, num_features, d) -> (8, 50, 5, 512)
            x_grid = x_grid + self.video_pos_encoder.pe[:, :self.chunk_size, :].unsqueeze(2)
        print("Đây là x_grid sau khi xử lý baidu:", x_grid.shape)

        #Audio features
        if self.audio and featsA is not None:

            #Feature Augmentation
            if self.model_cfg['feature_augmentation'] & (not inference):
                featsA = self.temporal_dropA(featsA)
                featsA = self.random_switch(featsA)
            print("Đây là featsA:", featsA.shape)
            #VGGish backbone
            fA = featsA.shape[1] // 96 #number of 0.96 segments
            featsA = rearrange(featsA[:, :fA * 96, :], 'b (f ns) h -> (b f) 1 ns h', f = fA) #batch*segments x d x 96 x 128
            featsA = self.vggish_features(featsA)
            print("Đây là featsA sau khi qua backbone:", featsA.shape)
            featsA = featsA.flatten(1)
            print("Đây là featsA sau khi flatten:", featsA.shape)
            featsA = self.vggish_embeddings(featsA)
            print("Đây là featsA sau khi qua embedding:", featsA.shape)
            #Positional Encoding
            featsA = rearrange(featsA, '(b f) d -> b f d', f = fA) + self.encTA.expand(b, -1, -1) #batch x segments x d
            featsA += self.encA.expand(b, fA, -1) #batch x segments x d

            # Combine with x_grid
            if self.baidu:
                # Interpolate audio features to match video chunk size
                featsA_aligned = F.adaptive_avg_pool1d(featsA.permute(0, 2, 1), self.chunk_size).permute(0, 2, 1)
                print("Đây là featsA_aligned:", featsA_aligned.shape)
                # Expand to (b, chunk_size, 1, d) and concatenate
                x_grid = torch.cat([x_grid, featsA_aligned.unsqueeze(2)], dim=2)

        num_features = x_grid.shape[2]
        print(x_grid.shape, "âssssssssss")
        for i, layer in enumerate(self.encoder_layers):
            # Mỗi lớp TransformerEncoderLayer sẽ xử lý một chuỗi (B_new, L, D)
            # Do đó chúng ta cần reshape trước mỗi lần attention

            # 1. Temporal Attention
            # Reshape để trục thời gian (T) là trục sequence
            # (B, T, M, D) -> (B*M, T, D)
            x_for_temporal = rearrange(x_grid, 'b t m d -> (b m) t d')
            x_after_temporal = layer(x_for_temporal) # Áp dụng lớp Transformer
            x_grid = rearrange(x_after_temporal, '(b m) t d -> b t m d', m=num_features)
            
            # 2. Feature-wise Attention (trong cùng một lớp, đây là một biến thể)
            # Chúng ta có thể dùng cùng một lớp TransformerEncoderLayer hoặc một lớp khác
            # Reshape để trục feature (M) là trục sequence
            # (B, T, M, D) -> (B*T, M, D)
            x_for_feature = rearrange(x_grid, 'b t m d -> (b t) m d')
            x_after_feature = layer(x_for_feature) # Áp dụng lại cùng lớp Transformer
            x_grid = rearrange(x_after_feature, '(b t) m d -> b t m d', t=self.chunk_size)
            
        x_encoded_flat = rearrange(x_grid, 'b t m d -> b (t m) d')
        # --- Transformer decoder ---
        # Logic decoder giữ nguyên, nhưng nó sẽ hoạt động với tensor có số chiều và độ dài khác
        queries = self.queries.expand((b, -1, -1)).to(x_encoded_flat.device)
        decoder_output = self.Tdecoder(queries, x_encoded_flat)

        refine_feats = x_encoded_flat.permute(0, 2, 1)
        refine_feats = self.residual_transform(refine_feats)
        refine_residual_feats = refine_feats.permute(0, 2, 1)
        
        decoder_output = torch.cat([decoder_output,refine_residual_feats], dim = -1) #Concatenate decoder output with residual features
        decoder_output = self.fusion_layer(decoder_output)
    
        #Classification head
        y1 = self.clas_head(decoder_output)
        y2 = self.displ_head(decoder_output)

        output = dict()
        output['preds'] = y1
        output['predsD'] = y2
        output['labels'] = labels
        output['labelsD'] = labelsD

        return output
    
class uncertainty_head(nn.Module):
    """
    Uncertainty-aware prediction head
    """
    def __init__(self, input_dim, output_dim, drop = 0.2):
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),    
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):

        x = self.shared_head(x)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        x = torch.stack((mean, logvar), dim = 3)
        return x

class Bfeat_module(nn.Module):
    def __init__(self, input_dim, output_dim, drop = 0.2):
        super().__init__()
        self.head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.head(x)

class mixupYolo(torch.nn.Module):
    """
    Mixup module class
    """
    def __init__(self, alpha = 0.3, beta = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.betaD = torch.distributions.beta.Beta(alpha, beta)
        self.n_queues = 2
            
    def forward(self, featB, featBQ, featA, featAQ, labels, labelsQ, labelsD, labelsDQ):
        #len of batch
        b = len(labels)
        classes = labels.shape[-1]

        #same lambda for all the batch
        lamb = self.betaD.sample()

        #Index of action and nqueue to do mixup
        idxa = torch.randint(0, classes, (b,))
        idxnq = torch.randint(0, self.n_queues, (b,))

        #Mixture
        if featB != None:
            featB = featB * lamb + (1-lamb) * featBQ[idxa, idxnq]
        if featA != None:
            featA = featA * lamb + (1-lamb) * featAQ[idxa, idxnq]
        if labels != None:
            labels = labels * lamb + (1-lamb) * labelsQ[idxa, idxnq]
        if labelsD != None:
            labelsD = ((labelsD == 1000) & (labelsDQ[idxa, idxnq] == 1000)) * 1000 + ((labelsD == 1000) & (labelsDQ[idxa, idxnq] != 1000)) * labelsDQ[idxa, idxnq] + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] == 1000)) * labelsD + ((labelsD != 1000) & (labelsDQ[idxa, idxnq] != 1000)) * (labelsD * lamb + (1-lamb) * labelsDQ[idxa, idxnq])

        return featB, featA, labels, labelsD
        
# Augmentation modules
class temporal_dropM(nn.Module):
    def __init__(self, p = 0.0, dim = 9344):
        super().__init__()
        self.p = p
        self.embedding = nn.Parameter(torch.rand(dim))
        
    def forward(self, x: torch.Tensor):
        x_aux = x.clone()
        mask = torch.rand(x_aux.shape[1]) < self.p
        x_aux[:, mask] = self.embedding
        return x_aux
    
class random_switchM(nn.Module):
    def __init__(self, p = 0.0):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor):
        x_aux = x.clone()
        idxs = torch.arange(x_aux.shape[1]-1)[torch.rand(x_aux.shape[1]-1) < self.p]
        x_aux[:, idxs, :], x_aux[:, idxs+1, :] = x_aux[:, idxs+1, :], x_aux[:, idxs, :]
        return x_aux
