import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from dgl.nn.pytorch.conv import GATConv  # 引入GATConv以学习边权重
import dgl
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x,**kwargs):
        return self.norm(x)

class VAET(nn.Module):
    def __init__(self,input_dim, hidden_dim, latent_dim,head):
        super(VAET, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mu = nn.Linear(input_dim,hidden_dim)
        self.logvar = nn.Linear(input_dim,hidden_dim)
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 由于输入为[0,1]范围内的像素值，使用Sigmoid激活函数
        )
        self.loss = nn.MSELoss()
        self.head = head

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        eps = torch.exp(0.5 * eps)
        return mu + eps * logvar

    def forward(self, x):
        # 编码器
        b,n,c = x.shape
        x = x.unsqueeze(1).expand(b, self.head, n, c).permute(0, 1, 3, 2)
        x = F.adaptive_avg_pool2d(x, (c, 4096))
        enc_output = self.encoder(x)
        mu, logvar = enc_output[:,:,:, :self.latent_dim], enc_output[:,:,:, self.latent_dim:]
        # 重新参数化
        z = self.reparameterize(mu, logvar)
        KL_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # KL divergence
        loss = KL_element
        # return loss,z
        return loss, z
class LinformerAttention_wind1(nn.Module):
    def __init__(self, dim=48, heads=4, dim_head=16, dropout=0., k=None, window_size=800,Iscrosstrans = None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.Iscrosstrans =Iscrosstrans
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.w = 30
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_head = nn.Linear(dim, inner_dim, bias=False)

        # reshape the output of self.to_qkv
        self.reshape = Rearrange('b n (head dim_head) -> b head n dim_head', head=self.heads)
        self.E = nn.Parameter(torch.randn(heads, dim_head, self.w))
        self.F = nn.Parameter(torch.randn(heads, dim_head, self.w))

        self.fc_mu = nn.Linear(4096, dim)
        self.fc_logvar = nn.Linear(4096, dim)
        self.fc11 = nn.Linear(dim, dim)
        self.fc12 = nn.Linear(dim, dim)
        self.fc13 = nn.Linear(dim, dim)

        self.fc21 = nn.Linear(dim, dim)
        self.fc22 = nn.Linear(dim, dim)
        self.fc23 = nn.Linear(dim, dim)

        self.lin1 = nn.Linear(self.w, dim_head)
        self.lin2 = nn.Linear(self.w, dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, *inputs):
        mask = None
        if len(inputs) == 4:
            x1, x2, z1, z2= inputs
            e1 = self.fc11(z1)
            f1 = self.fc12(z1)
            g1 = self.fc13(z1)

            e2 = self.fc11(z2)
            f2 = self.fc12(z2)
            g2 = self.fc13(z2)
        if len(inputs) == 2:
            x1, x2 = inputs
        b, n, c = x1.shape
        # compute the number of windows
        num_windows = (n + self.window_size - 1) // self.window_size

        # prepare output tensor
        out1 = torch.empty(b, n, self.heads * self.dim_head, device=x1.device)
        out2 = torch.empty(b, n, self.heads * self.dim_head, device=x2.device)
        # iterate over windows
        for i in range(num_windows):
            # get start and end indices of the current window
            start_idx = i * self.window_size
            end_idx = min(start_idx + self.window_size, n)

            # extract the current window
            x1_window = x1[:, start_idx:end_idx, :]
            x2_window = x2[:, start_idx:end_idx, :]

            # apply self.to_qkv and reshape its output
            qkv1 = self.reshape(self.to_qkv(x1_window)).chunk(3, dim=-1)
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

            qkv2 = self.reshape(self.to_qkv(x2_window)).chunk(3, dim=-1)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

            if len(inputs) == 4:
                q1 = torch.einsum('b h n d,b h d d  -> b h n d', q1, e2)
                k1 = torch.einsum('b h n d,b h d d  -> b h n d', k1, f2)
                v1 = torch.einsum('b h n d,b h d d  -> b h n d', v1, g2)

                q2 = torch.einsum('b h n d,b h d d  -> b h n d', q2, e1)
                k2 = torch.einsum('b h n d,b h d d  -> b h n d', k2, f1)
                v2 = torch.einsum('b h n d,b h d d  -> b h n d', v2, g1)

            # compute attention scores
            dots1 = torch.einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
            dots2 = torch.einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale

            if mask is not None:
                # pad the mask to match the shape of the attention scores
                mask_window = mask[:, start_idx:end_idx]
                mask_window = F.pad(mask_window.flatten(1), (1, 0), value=True)
                assert mask_window.shape[-1] == dots1.shape[-1], 'mask has incorrect dimensions'
                mask_window = rearrange(mask_window, 'b i -> b () i ()') * rearrange(mask_window, 'b j -> b () () j')
                dots1.masked_fill_(~mask_window, float('-inf'))
                dots2.masked_fill_(~mask_window, float('-inf'))

            # apply softmax and compute weighted sum
            attn1 = dots1.softmax(dim=-1)
            attn2 = dots2.softmax(dim=-1)

            out_window1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
            out_window2 = torch.einsum('b h i j, b h j d -> b h i d', attn2, v2)

            # reshape and store the output of the current window
            out_window1 = rearrange(out_window1, 'b head n dim_head -> b n (head dim_head)')
            out_window2 = rearrange(out_window2, 'b head n dim_head -> b n (head dim_head)')
            out1[:, start_idx:end_idx, :] = out_window1
            out2[:, start_idx:end_idx, :] = out_window2

        # apply final linear transformation and return the output
        out1 = self.to_out(out1)
        out2 = self.to_out(out2)
        return out1+out2
class Transformer_l(nn.Module):  ### heads maybe 3, dim_head may be 16
    def __init__(self, dim=48, depth=1, heads=4, dim_head=32, mlp_dim=48, window_size=800, sp_sz=64 * 64,
                 num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinformerAttention_wind1(dim, heads=heads, dim_head=dim, window_size=window_size,dropout=dropout),
                Residual(PreNorm(dim)),
                FeedForward(dim, mlp_dim, dropout=dropout),
                Residual(PreNorm(dim))]))

    def forward(self, *inputs):
        mask = None
        if len(inputs) == 4:
            x1, x2, e1, e2 = inputs
            for attn, R1,ff,R2 in self.layers:
                w1 = attn(x1, x2,e1,e2)
                w1= R1(w1)
                w1= ff(w1)
                w1= R2(w1)
        if len(inputs) == 2:
            x1, x2 = inputs
            for attn, R1, ff, R2 in self.layers:
                w1 = attn(x1, x2)
                w1 = R1(w1)
                w1 = ff(w1)
                w1 = R2(w1)
        return w1


class DynamicGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=1):
        super(DynamicGNNLayer, self).__init__()
        self.gnn_fusion = GATConv(in_feats, out_feats // num_heads, num_heads=num_heads)

    def forward(self, graph, features):
        return self.gnn_fusion(graph, features)


# 自适应图构建
def construct_adaptive_graph(hsi_features, msi_features,number):
    k = number
    B, C, H, W = hsi_features.shape
    num_nodes = B * H * W
    hsi_features = rearrange(hsi_features, 'B C H W -> B (H W) C')
    msi_features = rearrange(msi_features, 'B C H W -> B (H W) C')

    similarity = torch.einsum('bik,bjk->bij', hsi_features, msi_features)
    _, indices = similarity.topk(k, dim=-1)

    src = torch.arange(num_nodes).repeat(k).to(hsi_features.device)
    dst = indices.view(-1)

    graph = dgl.graph((src, dst)).to(hsi_features.device)
    return hsi_features,msi_features,dgl.add_self_loop(graph)


class SGA(nn.Module):
    def __init__(self, inchannel):
        super(SGA, self).__init__()
        self.upscale = nn.Upsample(scale_factor=4, mode='bicubic')
        self.embed = nn.Sequential( nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0),
                                     nn.PReLU(),
                                    nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0),
                                     nn.PReLU(),
                                       )
        self.embed2 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0),
                                     nn.PReLU(),
                                    nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1, padding=0),
                                     nn.PReLU(),
                                    )

        self.upsampler = nn.Sequential(
                                        nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1),
                                       nn.PReLU(),
                                       nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1),
                                       nn.PReLU()
                                       )

        # 调整GATConv以适应拼接后的特征维度(64 + 64 = 128)
        self.gnn_fusion = GATConv(in_feats=inchannel*2, out_feats=inchannel, num_heads=1)



    def forward(self, msi_image, hsi_image):
        hsi_image = self.upscale(hsi_image)
        iden = hsi_image
        B, C, H, W = hsi_image.shape
        # 提取特征
        hsi_features = self.embed(hsi_image)
        msi_features = self.embed2(msi_image)
        RE_hsi_features,RE_msi_features,graph = construct_adaptive_graph(hsi_features,msi_features,5)
        combined_features = torch.cat([RE_hsi_features, RE_msi_features], dim=2)
        # 应用GAT融合特征
        combined_features = rearrange(combined_features, 'B (H W) C ->  (B H W) C',H=H,W=W)
        combined_features = self.gnn_fusion(graph,combined_features).squeeze()
        combined_features = rearrange(combined_features, '(B H W) c-> B c H W', H=H,W=W)
        # 重塑并上采样
        out = self.upsampler(combined_features)+iden
        return out




class CSGAV(nn.Module):
    def __init__(self, hsichannel=145, msichannel=4, img_size=64, patch_size=1, embed_dim=None, num_heads=4):
        super(CSGAV, self).__init__()
        self.conv0 = nn.Conv2d(msichannel, hsichannel, kernel_size=1, stride=1, padding=0)
        self.up = SGA(hsichannel)
        self.convupout = nn.Conv2d(hsichannel, hsichannel, kernel_size=3, stride=1, padding=1)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embedding =nn.Sequential(nn.Conv2d(hsichannel, embed_dim, kernel_size=3, padding=1),
            nn.PReLU()  # 可以加上激活函数，如 ReLU
        )
        self.vaet = VAET(4096, embed_dim, embed_dim, num_heads)
        self.Transformer_l = Transformer_l(dim=embed_dim, heads=num_heads, window_size=800)
        self.concat = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
                                    # 将输出通道数调整为输入通道数的两倍
                                    nn.ReLU(),
                                    nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1))  # 可以加上激活函数，如 ReLU)
        self.last = nn.Sequential(nn.Conv2d(embed_dim , embed_dim , kernel_size=3, padding=1),
                                  # 将输出通道数调整为输入通道数的两倍
                                  nn.ReLU(),
                                  nn.Conv2d(embed_dim , hsichannel, kernel_size=3, padding=1))  # 可以加上激活函数，如 ReLU)

    def forward(self, hsi, msi):
        B,c,H,W = msi.shape
        msi_up= self.conv0(msi)
        x1indenty = self.up(msi_up,hsi)
        concatenated_tensor = self.convupout(x1indenty)
        x1 = self.patch_embedding(msi_up)
        x2 = self.patch_embedding(concatenated_tensor)
        x1 = x1.permute(0, 2, 3, 1).flatten(1, 2)
        x2 = x2.permute(0, 2, 3, 1).flatten(1, 2)
        loss1,z1 = self.vaet(x1)
        loss2,z2 = self.vaet(x2)
        x3 = torch.cat((x1,x2),dim=2)
        x3 = rearrange(x3, 'B (H W) c -> B c H W', H=H)
        x3 = self.concat(x3)
        w= self.Transformer_l(x1, x2,z1,z2)
        w = rearrange(w, 'B (H W) c -> B c H W', H=H)
        x = self.last(x3*w+x3)+concatenated_tensor
        return x  #for comparsion


# 测试模型
if __name__ == '__main__':
    hsi_channel = 145 #hsi
    msi_channel = 4 #msi
    model = CSGAV(hsi_channel, msi_channel, embed_dim=96).cuda()
    x1 = torch.ones(1, 145, 16, 16).cuda() #Example input 1
    x2 = torch.ones(1, 4, 64, 64).cuda() #Example input 2
    output = model(x1, x2)
    print("Output shape:", output.shape)  # 输出形状应该与输入形状相同
    #测试模型