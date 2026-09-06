import torch
import torch.nn as nn
import torch.nn.functional as F


class FastGELU(nn.Module):
    """
    An approximation of the GELU activation function, used in models like BERT.
    This is the "new" GELU, which is slightly different from the original one.
    It's faster and mathematically equivalent to `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

class LighterMLPHead(nn.Module):
    """
    Một phiên bản "nhẹ" hơn của pred_head tiêu chuẩn với lớp ẩn nhỏ hơn.
    """
    def __init__(self, input_dim, num_classes, hidden_dim_factor=4, sigmoid=True, drop=0.2):
        """
        Args:
            input_dim (int): Chiều của feature đầu vào (d).
            num_classes (int): Số lượng lớp đầu ra (C).
            hidden_dim_factor (int): Hệ số để giảm chiều lớp ẩn. 
                                     hidden_dim = input_dim // hidden_dim_factor.
            sigmoid (bool): Có áp dụng sigmoid ở cuối không.
            drop (float): Tỷ lệ dropout.
        """
        super().__init__()
        hidden_dim = input_dim // hidden_dim_factor
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes)
        )
        if sigmoid:
            self.head.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        # Input x: (B, K, d)
        # Output: (B, K, C)
        return self.head(x)


class GroupedConvHead(nn.Module):
    """
    Sử dụng Grouped Convolutions để tạo ra một head nằm giữa Standard và GroupWise.
    """
    def __init__(self, 
                 input_dim= 512, 
                 num_classes=18, 
                 hidden_channels= 256, 
                 num_groups=2, 
                 sigmoid=True):
        """
        Args:
            input_dim (int): Chiều của feature đầu vào (d).
            num_classes (int): Số lượng lớp đầu ra (C).
            num_groups (int): Số lượng nhóm để chia sẻ tham số.
                               Phải là ước số chung của input_dim và num_classes.
            sigmoid (bool): Có áp dụng sigmoid ở cuối không.
        """
        super().__init__()
        # Để sử dụng grouped convolutions, số kênh vào và ra phải chia hết cho số nhóm
        # if input_dim % num_groups != 0 or hidden_channels % num_groups != 0:
            # raise ValueError("input_dim và hidden_channels phải chia hết cho num_groups.")

        # --- Layer 1: Trích xuất đặc trưng ngữ cảnh với kernel=3 ---
        # Sử dụng grouped convolution để hiệu quả
        self.conv_k3 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_channels,
            kernel_size=3,
            groups=num_groups,
            padding=1, # padding=1 với kernel=3 để giữ nguyên độ dài chuỗi
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.activation = nn.ReLU()
        
        # --- Layer 2: Tổng hợp và phân loại với kernel=1 ---
        # Phải dùng groups=1 để trộn thông tin từ tất cả các kênh đã học
        self.conv_k1 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=num_classes, # Chiếu về 18 lớp
            kernel_size=1,
            groups=1 # Quan trọng: groups=1 để tổng hợp
        )
        
        self.final_activation = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor đầu vào có shape (B, K, d)
        """
        # Conv1d yêu cầu đầu vào có shape (B, d, K)
        x_permuted = x.permute(0, 2, 1)
        
        # 1. Đi qua lớp đầu tiên (k=3)
        features = self.conv_k3(x_permuted)
        features = self.batch_norm(features)
        features = self.activation(features)
        
        # 2. Đi qua lớp thứ hai (k=1)
        # Output shape: (B, C, K)
        output_conv = self.conv_k1(features)
        
        # 3. Chuyển về shape mong muốn và trả về
        output = output_conv.permute(0, 2, 1)
        
        return self.final_activation(output)


import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedConvHeadV2(nn.Module):
    """
    Sử dụng Cross-Attention để tích hợp class_embeddings vào prediction head.
    """
    def __init__(self, 
                 input_dim: int = 512, 
                 num_classes: int = 18, 
                 hidden_channels: int = 256, 
                 num_groups: int = 2, 
                 num_attn_heads: int = 8,
                 dropout_p: float = 0.1,
                 sigmoid: bool = True):
        """
        Args:
            input_dim (int): Chiều của feature đầu vào (d).
            num_classes (int): Số lượng lớp đầu ra (C).
            hidden_channels (int): Chiều của các feature trung gian và attention.
            num_groups (int): Số nhóm cho lớp Conv1D ban đầu.
            num_attn_heads (int): Số lượng head trong Multi-head Attention.
            dropout_p (float): Tỷ lệ dropout.
            sigmoid (bool): Có áp dụng sigmoid ở cuối không.
        """
        super().__init__()
        
        # --- 1. Lớp trích xuất đặc trưng cục bộ (Local Feature Extractor) ---
        self.local_feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_channels,
                kernel_size=3,
                groups=num_groups,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # --- 2. Lớp chuẩn bị Ngữ cảnh (Context Preparation) ---
        # Chiếu class_embeddings từ input_dim về hidden_channels để khớp chiều
        self.context_projection = nn.Linear(input_dim, hidden_channels)

        # --- 3. Lớp Cross-Attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_attn_heads,
            dropout=dropout_p,
            batch_first=True  # Rất quan trọng để nhận input (B, K, d)
        )
        self.attn_dropout = nn.Dropout(dropout_p)
        self.attn_layernorm = nn.LayerNorm(hidden_channels)
        
        # --- 4. Lớp phân loại cuối (Final Classifier) ---
        # Feed-Forward Network sau attention, một pattern phổ biến
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels * 2, num_classes)
        )
        
        self.final_activation = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor, class_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor đầu vào có shape (B, K, d), với d=input_dim.
            class_embeddings (torch.Tensor): Tensor embedding có shape (C, d), với d=input_dim.
        """
        batch_size = x.size(0)
        
        # --- 1. Trích xuất feature cục bộ ---
        # Input shape cho Conv1d: (B, d, K)
        x_permuted = x.permute(0, 2, 1)
        # local_features shape: (B, hidden_channels, K)
        local_features = self.local_feature_extractor(x_permuted)
        # Chuyển về shape (B, K, hidden_channels) để làm Query
        query = local_features.permute(0, 2, 1)
        
        # --- 2. Chuẩn bị Key và Value từ class_embeddings ---
        # projected_context shape: (C, hidden_channels)
        projected_context = self.context_projection(class_embeddings)
        # Mở rộng cho cả batch để làm Key và Value
        # key_value shape: (B, C, hidden_channels)
        key_value = projected_context.unsqueeze(0).expand(batch_size, -1, -1)
        
        # --- 3. Áp dụng Cross-Attention ---
        # Query: (B, K, hidden_channels)
        # Key & Value: (B, C, hidden_channels)
        # attn_output shape: (B, K, hidden_channels)
        attn_output, attn_weights = self.attention(
            query=query, 
            key=key_value, 
            value=key_value
        )
        
        # Áp dụng Dropout và Residual Connection + LayerNorm (quan trọng cho sự ổn định)
        query = query + self.attn_dropout(attn_output)
        query = self.attn_layernorm(query)

        # --- 4. Đưa qua lớp phân loại cuối ---
        # `query` bây giờ là các feature đã được làm giàu ngữ nghĩa
        logits = self.ffn(query) # Shape: (B, K, num_classes)
        
        return self.final_activation(logits)


class MoEHead(nn.Module):
    """
    Một prediction head dựa trên kiến trúc Mixture of Experts.
    """
    def __init__(self, input_dim, num_classes, num_experts, top_k=2, sigmoid=True, drop=0.2):
        """
        Args:
            input_dim (int): Chiều của feature đầu vào (d).
            num_classes (int): Số lượng lớp đầu ra (C).
            num_experts (int): Số lượng "chuyên gia" để lựa chọn.
            top_k (int): Số lượng chuyên gia hàng đầu được kích hoạt cho mỗi token.
            sigmoid (bool): Có áp dụng sigmoid ở cuối không.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Gating network: một lớp Linear đơn giản để quyết định trọng số cho các expert
        self.gate = nn.Linear(input_dim, num_experts)

        # Experts: một danh sách các mạng MLP nhỏ (giống pred_head)
        # Mỗi expert là một chuyên gia xử lý
        self.experts = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(input_dim, num_classes)
            ) for _ in range(num_experts)]
        )
        
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor):
        # Input x: (B, K, d) - B: batch, K: queries, d: dim
        batch_size, num_queries, dim = x.shape

        # 1. Gating: Tính toán trọng số cho các expert
        # gate_logits.shape: (B, K, num_experts)
        gate_logits = self.gate(x)
        
        # 2. Chọn top_k expert và tính trọng số softmax
        # find top_k values and their indices
        # top_k_weights.shape, top_k_indices.shape: (B, K, top_k)
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1, largest=True)
        
        # Áp dụng softmax cho các trọng số đã chọn
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 3. Tính toán đầu ra từ các expert
        # Ta cần tính toán một cách hiệu quả thay vì dùng vòng lặp for
        
        # Làm phẳng batch và query để xử lý
        # flat_x.shape: (B*K, d)
        flat_x = x.reshape(-1, dim)
        
        # flat_indices.shape: (B*K, top_k)
        flat_indices = top_k_indices.reshape(-1, self.top_k)
        
        # Tạo ma trận kết quả cuối cùng
        # final_output.shape: (B*K, C)
        final_output = torch.zeros(batch_size * num_queries, self.num_classes, device=x.device)
        
        # Lặp qua các expert để tính toán
        # Đây là cách làm đơn giản, có thể tối ưu hóa thêm cho tốc độ
        for i, expert in enumerate(self.experts):
            # Tìm tất cả các token được gán cho expert này
            # expert_mask.shape: (B*K, top_k)
            expert_mask = (flat_indices == i)
            
            # Lấy chỉ số của các token đó
            # row_indices.shape: (num_tokens_for_this_expert,)
            row_indices, col_indices = torch.where(expert_mask)
            
            if row_indices.numel() > 0:
                # Lấy các token đầu vào tương ứng
                expert_input = flat_x[row_indices]
                
                # Tính đầu ra từ expert
                expert_output = expert(expert_input) # shape: (num_tokens, C)
                
                # Lấy trọng số tương ứng
                # weights_for_expert.shape: (num_tokens, 1)
                weights_for_expert = top_k_weights.reshape(-1, self.top_k)[row_indices, col_indices].unsqueeze(1)
                
                # Cộng dồn vào kết quả cuối cùng (tổng có trọng số)
                # Dùng index_add_ để tránh race condition trên GPU
                final_output.index_add_(0, row_indices, expert_output * weights_for_expert)

        # Reshape kết quả về dạng ban đầu
        # final_output.shape: (B, K, C)
        final_output = final_output.reshape(batch_size, num_queries, self.num_classes)
        
        return self.sigmoid(final_output)

class pred_head(nn.Module):
    """
    Standard prediction head
    """
    def __init__(self, input_dim, output_dim, sigmoid = True, drop = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, output_dim)
        )
        if sigmoid:
            self.head.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        return self.head(x) 

class uncertainty_head(nn.Module):
    """
    Uncertainty-aware prediction head
    """
    def __init__(self, input_dim, output_dim, drop = 0.2):
        """
        Initializes the uncertainty-aware prediction head

        Parameters
        ----------
        input_dim : int
            Input dimension of the head
        output_dim : int
            Output dimension of the head
        drop : float, optional
            Dropout probability, by default 0.2
        """
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(input_dim, input_dim),
            nn.GELU()
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

class FactorizedGroupWise(nn.Module):
    def __init__(self, input_dim, num_classes, rank=64, sigmoid=True, dropout=0.1):
        super().__init__()
        # Low-rank factorization: W = U @ V^T
        # Thay vì lưu matrix (num_classes × input_dim)
        # Ta lưu U (num_classes × rank) và V (input_dim × rank)
        
        self.rank = rank
        self.U = nn.Parameter(torch.Tensor(num_classes, rank))
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        
        # Thêm non-linearity nhẹ
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.sigmoid = nn.Sigmoid() if sigmoid else nn.Identity()
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: (B, K, input_dim)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Factorized computation: x @ V @ U^T
        x_proj = torch.matmul(x, self.V)  # (B, K, rank)
        output = torch.matmul(x_proj, self.U.t())  # (B, K, num_classes)
        output = output + self.bias
        
        return self.sigmoid(output)
