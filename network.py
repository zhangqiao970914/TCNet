import torch
import time
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from os.path import join
from t2t_vit import T2t_vit_t_14
from transformer_block import Block, get_sinusoid_encoding
from timm.models.layers import trunc_normal_
from token_performer import Token_performer

np.set_printoptions(suppress=True, threshold=1e5)
import argparse


def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


"""
weights_init:
    Weights initialization.
"""


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


""""
VGG16:
    VGG16 backbone.
"""


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers)
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats)
        return feats


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))  # [B, 14*14, 384]
        fea_1_16 = self.encoderlayer(fea_1_16)
        return fea_1_16


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio, img_size // ratio), kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea



class MCM(nn.Module):
    def __init__(self, in_dim):
        super(MCM, self).__init__()
        # Co-attention
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

        self.query = nn.Linear(512, 512)
        self.key = nn.Linear(512, 512)
        self.value = nn.Linear(512, 512)

        self.conv512_64 = nn.Conv2d(512, 64, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, xc, xt):
        # xc[10,512,16,16]  xt[10,256,384]
        B, C, H, W = xc.size()
        c_value = self.value_conv(xc)
        c_query = self.query_conv(xc).view(B, -1, W * H).permute(0, 2, 1)  # [B,HW,C]
        c_key = self.key_conv(xc).view(B, -1, W * H).permute(1, 0, 2)  # [B,C,HW]
        c_query = c_query.contiguous().view(-1, C)  # [BHW,C]
        c_key = c_key.contiguous().view(C, -1)
        c_xw = torch.matmul(c_query, c_key)  # [BHW,BHW]
        c_xw = c_xw.view(B * H * W, B, H * W)  # [BHW, B, HW]
        c_max = torch.max(c_xw, dim=-1)[0]  # [BHW, B]
        c_avg = torch.mean(c_xw, dim=-1)  # [BHW, B]
        c_co = c_max + c_avg  # [BHW, B]
        c_co = c_co.sum(-1)  # [BWH]
        c_co = c_co.view(B, -1) * self.scale
        c_co = F.softmax(c_co, dim=-1)  # [B,HW]
        c_co = c_co.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        # 局部自注意力
        Bt, HW, Ct = xt.size() #[10,196,512]
        t_value = xt.transpose(1, 2).reshape(Bt, Ct, int(np.sqrt(HW)), int(np.sqrt(HW)))
        t_query = self.query(xt).contiguous().view(-1, Ct)
        t_key = self.key(xt).contiguous().view(-1, Ct)
        t_xw = torch.matmul(t_query, t_key.permute(1, 0))
        t_xw = t_xw.view(Bt * HW, Bt, HW)
        t_max = torch.max(t_xw, dim=-1)[0]  # [BHW, B]
        t_avg = torch.mean(t_xw, dim=-1)  # [BHW, B]
        t_co = t_max + t_avg  # [BHW, B]
        t_co = t_co.sum(-1)  # [BWH]
        t_co = t_co.view(Bt, -1) * self.scale
        t_co = F.softmax(t_co, dim=-1)
        t_co = t_co.view(Bt, int(np.sqrt(HW)), int(np.sqrt(HW))).unsqueeze(1)  # [ B,1,16,16]
        # 全局与局部
        ct_xw = torch.matmul(c_query, t_key.permute(1, 0))
        ct_xw = ct_xw.view(B * H * W, B, H * W)

        ct_max = torch.max(ct_xw, dim=-1)[0]  # [BHW, B]
        ct_avg = torch.mean(ct_xw, dim=-1)  # [BHW, B]
        ct_co = ct_max + ct_avg  # [BHW, B]
        ct_co = ct_co.sum(-1)  # [BWH]
        ct_co = ct_co.view(B, -1) * self.scale
        ct_co = F.softmax(ct_co, dim=-1)  # [B,HW]
        ct_co = ct_co.view(B, H, W).unsqueeze(1)  # [B,1,16,16]
        # 局部与全局
        tc_xw = torch.matmul(t_query, c_key)
        tc_xw = tc_xw.view(Bt * HW, Bt, HW)
        tc_max = torch.max(tc_xw, dim=-1)[0]  # [BHW, B]
        tc_avg = torch.mean(tc_xw, dim=-1)  # [BHW, B]
        tc_co = tc_max + tc_avg  # [BHW, B]
        tc_co = tc_co.sum(-1)  # [BWH]
        tc_co = tc_co.view(Bt, -1) * self.scale
        tc_co = F.softmax(tc_co, dim=-1)
        tc_co = tc_co.view(Bt, int(np.sqrt(HW)), int(np.sqrt(HW))).unsqueeze(1)  # [ B,1,16,16]
        # 相乘部分
        c_co = self.conv512_64(c_co * c_value)  # [64,16,16]
        t_co = self.conv512_64(t_co * t_value)  # [512,16,16]

        ct_co = self.conv512_64(ct_co * c_value)  # [64,16,16]

        tc_co = self.conv512_64(tc_co * t_value)  # [512,16,16]

        c_ct_cat = self.conv1(torch.cat([c_co, ct_co], 1))
        t_ct_cat = self.conv2(torch.cat([t_co, tc_co], 1))

        out_final = self.conv3(torch.cat([c_ct_cat, t_ct_cat], 1))

        return out_final




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CCM(nn.Module):
    def __init__(self):
        super(CCM, self).__init__()

        self.attention_feature0 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                                nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.PReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, z):
        # x:CNN y:Transformer z:co-attention
        z = self.sigmoid(z)

        Gx = x * z
        Gy = y * z

        G0 = self.attention_feature0(torch.cat((Gx, Gy), dim=1))
        G0 = F.adaptive_avg_pool2d(torch.sigmoid(G0), 1)
        c0_Gx = G0[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * Gx
        c0_Gy = G0[:, 1, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * Gy

        temp_y = c0_Gy.mul(self.ca(c0_Gy))
        temp_x = c0_Gx.mul(self.sa(c0_Gx))
        final = self.conv2(torch.cat([temp_y, temp_x], 1))
        return final


class GCPD(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, img_size=224):

        super(GCPD, self).__init__()
        self.img_size = img_size

        self.decoder0 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=False)
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.token_trans6 = token_Transformer(embed_dim=384, depth=8, num_heads=8, mlp_ratio=3.)
        self.token_trans5 = token_Transformer(embed_dim=384, depth=6, num_heads=6, mlp_ratio=3.)
        self.token_trans4 = token_Transformer(embed_dim=384, depth=4, num_heads=4, mlp_ratio=3.)
        self.token_trans3 = token_Transformer(embed_dim=384, depth=2, num_heads=2, mlp_ratio=3.)

        self.fc = nn.Linear(64, 384)
        self.fc_192_384 = nn.Linear(192, 384)
        self.fc_448_384 = nn.Linear(448, 384)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=False)

        self.pre_1_16 = nn.Linear(384, 1)
        self.pre_1_8 = nn.Linear(384, 1)
        self.pre_1_4 = nn.Linear(384, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, z):
        # x[B, 64, 14, 14];y[B, 64, 28, 28];z[B, 64, 56, 56] sa[B, 64, 14, 14]
        x5 = x
        y4 = self.downsample2(y)
        z3 = self.downsample4(z)
        feat_t = torch.cat([x5, y4, z3], 1)  # [B, 192, 14, 14]
        B, Ct, Ht, Wt = feat_t.shape
        feat_t = feat_t.view(B, Ct, -1).transpose(1, 2)
        feat_t = self.fc_192_384(feat_t)  # [B, 14*14, 384]
        Tt = self.token_trans6(feat_t)
        Tt = Tt + feat_t
        mask_g = self.pre_1_16(Tt)
        mask_g = mask_g.transpose(1, 2).reshape(B, 1, Ht, Wt)

        #####################第五层########################
        B, Cx, Hx, Wx = x.shape
        x_fea = x.view(B, Cx, -1).transpose(1, 2)  # [B, 14*14, Cx]
        tx = torch.cat([Tt, x_fea], dim=2)
        tx = self.fc_448_384(tx)
        # x_fea = self.fc(x_fea)  # [B, 14*14, 384]
        tx = self.token_trans5(tx)
        tx = tx + self.fc(x_fea)
        mask_x = self.pre_1_16(tx)
        mask_x = mask_x.transpose(1, 2).reshape(B, 1, Hx, Wx)

        #####################第四层######################
        B, Cy, Hy, Wy = y.shape
        y_fea = y.view(B, Cy, -1).transpose(1, 2)  # [B, 28*28, Cy]
        xy = self.decoder1(tx, y_fea)  # [B, 28*28, 64]
        ty = self.fc(xy)  # [B, 28*28, 384]
        ty = self.token_trans4(ty)
        ty = ty + self.fc(y_fea)
        mask_y = self.pre_1_8(ty)
        mask_y = mask_y.transpose(1, 2).reshape(B, 1, Hy, Wy)

        ####################第三层######################
        B, Cz, Hz, Wz = z.shape
        z_fea = z.view(B, Cz, -1).transpose(1, 2)  # [B, 56*56, Cz]
        yz = self.decoder2(ty, z_fea)  # [B, 56*56, Cz]
        tz = self.fc(yz)
        tz = self.token_trans3(tz)
        tz = tz + self.fc(z_fea)
        mask_z = self.pre_1_4(tz)
        mask_z = mask_z.transpose(1, 2).reshape(B, 1, Hz, Wz)
        return self.upsample4(mask_z), self.upsample8(mask_y), self.upsample16(mask_x), self.upsample16(mask_g)


class ICNet(nn.Module):
    def __init__(self, channel=64):
        super(ICNet, self).__init__()
        # Backbone
        self.vgg = VGG16()
        parser = argparse.ArgumentParser()
        parser.add_argument('--pretrained_model', default='/hy-tmp/TCNet-main/80.7_T2T_ViT_t_14.pth.tar',
                            type=str, help='load Pretrained model')
        args = parser.parse_args()
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)


        # 共同注意力部分
        self.fc = nn.Linear(384, 512)
        self.conv512_256 = nn.Conv2d(512, 256, 1)
        self.conv512_128 = nn.Conv2d(512, 128, 1)
        self.conv512_64 = nn.Conv2d(512, 64, 1)
        self.conv256_64 = nn.Conv2d(256, 64, 1)
        self.conv128_64 = nn.Conv2d(128, 64, 1)
        self.conv64_1 = nn.Conv2d(64, 1, 1)
        self.sa = MCM(512)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.sig = nn.Sigmoid()

        #CCM
        self.ccm5 = CCM()
        self.ccm4 = CCM()
        self.ccm3 = CCM()
        # GCPD
        self.decoder = GCPD(embed_dim=384, token_dim=64, img_size=224)

    def forward(self, image_group, is_training):
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_group)
        rgb_fea_1_16 = self.fc(rgb_fea_1_16)
        Bt, HW, Ct = rgb_fea_1_16.size()
        t_value = rgb_fea_1_16.transpose(1, 2).reshape(Bt, Ct, int(np.sqrt(HW)), int(np.sqrt(HW)))

        # Extract features from the VGG16 backbone.
        conv1_2 = self.vgg(image_group, 'conv1_1', 'conv1_2_mp')  # shape=[N, 64, 224, 224]
        conv2_2 = self.vgg(conv1_2, 'conv1_2_mp', 'conv2_2_mp')  # shape=[N, 128, 112, 112]
        conv3_3 = self.vgg(conv2_2, 'conv2_2_mp', 'conv3_3_mp')  # shape=[N, 256, 56, 56]
        conv4_3 = self.vgg(conv3_3, 'conv3_3_mp', 'conv4_3_mp')  # shape=[N, 512, 28, 28]
        conv5_3 = self.vgg(conv4_3, 'conv4_3_mp', 'conv5_3_mp')  # shape=[N, 512, 14, 14]
        sa = self.sa(conv5_3, rgb_fea_1_16)  # [64,14,14]

        x5 = self.ccm5(self.conv512_64(conv5_3), self.conv512_64(t_value), sa)
        x4 = self.ccm4(self.conv512_64(conv4_3), self.conv512_64(self.upsample2(t_value)), self.upsample2(sa))
        x3 = self.ccm3(self.conv256_64(conv3_3), self.conv512_64(self.upsample4(t_value)), self.upsample4(sa))

        S_3_pred, S_4_pred, S_5_pred, S_g_pred = self.decoder(x5, x4, x3)

        # Return predicted co-saliency maps.
        if is_training:
            return S_3_pred, S_4_pred, S_5_pred, S_g_pred
        else:
            preds = S_3_pred
            return preds
