import torch
import torch.nn as nn
from einops import rearrange

from impl.head_cls import FastGELU

class GatedFusionUnit(nn.Module):
    """
    Học cách kết hợp đặc trưng video và audio một cách động
    thay vì chỉ ghép nối cứng nhắc.
    """
    def __init__(self, model_dim: int):
        super().__init__()
        # Mạng cổng sẽ nhận cả hai đặc trưng và quyết định trọng số
        self.gate_linear = nn.Linear(model_dim * 2, model_dim)

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        # Cả hai đầu vào phải có shape (batch, sequence_length, model_dim)
        if video_feats.shape[1] != audio_feats.shape[1]:
            raise ValueError(f"Sequence lengths must match for Gated Fusion, but got {video_feats.shape[1]} and {audio_feats.shape[1]}")
        
        # <<< SỬA LỖI: Đồng bộ hóa batch size >>>
        target_batch_size = video_feats.shape[0]
        if audio_feats.shape[0] != target_batch_size:
            if audio_feats.shape[0] == 1:
                # Repeat audio features để khớp với batch size của video
                audio_feats = audio_feats.repeat(target_batch_size, 1, 1)
            else:
                # Trường hợp phức tạp hơn - có thể cần xử lý khác
                raise ValueError(f"Cannot align batch sizes: video={target_batch_size}, audio={audio_feats.shape[0]}")
            
        combined = torch.cat((video_feats, audio_feats), dim=-1)
        
        # Tính toán cổng sigmoid, giá trị từ 0 đến 1
        gate = torch.sigmoid(self.gate_linear(combined))
        
        # Trộn thông tin: gate=1 -> giữ video, gate=0 -> giữ audio
        fused_feats = gate * video_feats + (1 - gate) * audio_feats
        return fused_feats

class DeeperGatedFusion(nn.Module):
    def __init__(self, model_dim: int, bottleneck_factor=2):
        super().__init__()
        bottleneck_dim = model_dim // bottleneck_factor
        self.gating_network = nn.Sequential(
            nn.Linear(model_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, model_dim)
        )
    
    def forward(self, video_feats, audio_feats):
        # <<< SỬA LỖI: Đồng bộ hóa batch size >>>
        target_batch_size = video_feats.shape[0]
        if audio_feats.shape[0] != target_batch_size:
            if audio_feats.shape[0] == 1:
                # Repeat audio features để khớp với batch size của video
                audio_feats = audio_feats.repeat(target_batch_size, 1, 1)
            else:
                # Trường hợp phức tạp hơn - có thể cần xử lý khác
                raise ValueError(f"Cannot align batch sizes: video={target_batch_size}, audio={audio_feats.shape[0]}")
        
        gate = torch.sigmoid(self.gating_network(torch.cat((video_feats, audio_feats), dim=-1)))
        if gate > 0.9:
            print("gate at current step: ", gate, "audio feature đang không có ý nghĩa")
        fused_feats = gate * video_feats + (1 - gate) * audio_feats
        return fused_feats

# Trong file chứa module fusion của bạn, ví dụ: impl/modules/gating.py

import torch
import torch.nn as nn
from einops import rearrange

class ConvGatedFusion(nn.Module):
    """
    Cơ chế fusion có cổng sử dụng tích chập để nhận biết ngữ cảnh.
    PHIÊN BẢN ĐÃ SỬA LỖI ĐỒNG BỘ HÓA ĐỘ DÀI CHUỖI.
    """
    def __init__(self, model_dim: int, kernel_size: int = 3):
        super().__init__()
        # Conv1d hoạt động trên chiều (Batch, Channels, Length)
        # Channels ở đây là model_dim * 2
        self.gating_conv = nn.Conv1d(
            in_channels=model_dim * 2,
            out_channels=model_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 # Giữ nguyên độ dài chuỗi
        )
    
    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        # Lấy độ dài chuỗi video làm tham chiếu
        if audio_feats is None:
            return video_feats
        
        target_batch_size = video_feats.shape[0]
        target_len = video_feats.shape[1]
        
        # <<< SỬA LỖI: Đồng bộ hóa batch size >>>
        # Nếu batch size không khớp, repeat audio features
        if audio_feats.shape[0] != target_batch_size:
            if audio_feats.shape[0] == 1:
                # Repeat audio features để khớp với batch size của video
                audio_feats = audio_feats.repeat(target_batch_size, 1, 1)
            else:
                # Trường hợp phức tạp hơn - có thể cần xử lý khác
                raise ValueError(f"Cannot align batch sizes: video={target_batch_size}, audio={audio_feats.shape[0]}")
        
        # <<< SỬA LỖI: Tự động điều chỉnh độ dài chuỗi audio để khớp với video >>>
        # Chỉ thực hiện nếu độ dài không khớp
        if audio_feats.shape[1] != target_len:
            # Chuyển (Batch, Length, Dim) -> (Batch, Dim, Length) để pooling
            audio_feats_T = rearrange(audio_feats, 'b l d -> b d l')
            
            # Sử dụng Adaptive Pooling để thay đổi độ dài chuỗi
            audio_resized_T = nn.functional.adaptive_avg_pool1d(audio_feats_T, target_len)
            
            # Chuyển về lại shape ban đầu (Batch, Length, Dim)
            audio_feats = rearrange(audio_resized_T, 'b d l -> b l d')
        
        # Bây giờ, video_feats và audio_feats chắc chắn có cùng shape[0] và shape[1]
        # Lệnh cat này bây giờ sẽ hoạt động một cách an toàn
        combined = torch.cat((video_feats, audio_feats), dim=-1) # Shape: (B, L, 2*D)
        
        # Chuyển vị để phù hợp với Conv1d: (B, L, 2*D) -> (B, 2*D, L)
        combined_T = rearrange(combined, 'b l d -> b d l')
        
        # Áp dụng tích chập để tính toán gate dựa trên ngữ cảnh
        gate_logits_T = self.gating_conv(combined_T) # Shape: (B, D, L)
        
        # Chuyển vị lại: (B, D, L) -> (B, L, D)
        gate_logits = rearrange(gate_logits_T, 'b d l -> b l d')
        
        gate = torch.sigmoid(gate_logits)
        
        # Trộn thông tin: gate=1 -> giữ video, gate=0 -> giữ audio
        fused_feats = gate * video_feats + (1 - gate) * audio_feats
        return fused_feats


class AttentionFusion(nn.Module):
    def __init__(self, model_dim: int, num_heads=8):
        super().__init__()
        # Cho phép video "hỏi" audio
        self.video_asks_audio = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Linear(model_dim, model_dim) # Một lớp FFN đơn giản

    def forward(self, video_feats, audio_feats):
        # <<< SỬA LỖI: Đồng bộ hóa batch size >>>
        target_batch_size = video_feats.shape[0]
        if audio_feats.shape[0] != target_batch_size:
            if audio_feats.shape[0] == 1:
                # Repeat audio features để khớp với batch size của video
                audio_feats = audio_feats.repeat(target_batch_size, 1, 1)
            else:
                # Trường hợp phức tạp hơn - có thể cần xử lý khác
                raise ValueError(f"Cannot align batch sizes: video={target_batch_size}, audio={audio_feats.shape[0]}")
        
        # Video là Query, Audio là Key và Value
        attended_audio, _ = self.video_asks_audio(
            query=video_feats, 
            key=audio_feats, 
            value=audio_feats
        )
        
        # Kết quả là sự kết hợp của video gốc và thông tin nó "nghe" được từ audio
        # Đây là một dạng gating/điều chỉnh rất phức tạp
        fused_feats = self.norm1(video_feats + attended_audio)
        fused_feats = self.ffn(fused_feats) # Qua một lớp biến đổi cuối cùng
        
        return fused_feats