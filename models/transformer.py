import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from models.blocks import AttnBlock, Attention, Mlp

class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            fused_attn=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x

class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    time -> view -> space attention
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        view_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
        use_checkpoint=True,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.use_checkpoint = use_checkpoint

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )

        
        # self.view_blocks = nn.ModuleList(
        #     [
        #         AttnBlock(
        #             hidden_size,
        #             num_heads,
        #             mlp_ratio=mlp_ratio,
        #             attn_class=Attention,
        #         )
        #         for _ in range(view_depth)
        #     ]
        # )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True, add_view_attn=True):
        use_ckpt = self.training and self.use_checkpoint
    
        def _cp_call(module, *args, _use_ckpt=False, **kwargs):
            if not _use_ckpt:
                return module(*args, **kwargs)
            
            return cp.checkpoint(lambda *t: module(*t, **kwargs), *args, use_reentrant=False)
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        layers = []
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            # time_tokens = self.time_blocks[i](time_tokens)
            time_tokens = _cp_call(self.time_blocks[i], time_tokens, _use_ckpt=use_ckpt)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C

            
            view_tokens = tokens.permute(1, 2, 0, 3).contiguous().view(N*T, B, -1)    # (N T) B C

            tokens = view_tokens.view(N, T, B, -1).permute(2, 0, 1, 3)
                # print('view attention good')
                 

            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                # virtual_tokens = self.space_virtual2point_blocks[j](
                #     virtual_tokens, point_tokens, mask=mask
                # )

                # virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                # point_tokens = self.space_point2virtual_blocks[j](
                #     point_tokens, virtual_tokens, mask=mask
                # )

                # virtual <- point (cross-attn)
                virtual_tokens = _cp_call(
                    self.space_virtual2point_blocks[j],
                    virtual_tokens, point_tokens,
                    _use_ckpt=use_ckpt,
                    mask=mask
                )

                # virtual self-attn
                virtual_tokens = _cp_call(
                    self.space_virtual_blocks[j],
                    virtual_tokens,
                    _use_ckpt=use_ckpt
                )

                # point <- virtual (cross-attn)
                point_tokens = _cp_call(
                    self.space_point2virtual_blocks[j],
                    point_tokens, virtual_tokens,
                    _use_ckpt=use_ckpt,
                    mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow