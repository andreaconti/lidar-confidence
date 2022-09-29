import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from typing import Literal

__all__ = ["LCE"]

# % MODEL DEFINITION
# (it's useful to separate the model and the training definition to be always
# able to export the model in TorchScript)


class LCE(nn.Module):
    def __init__(self, head: Literal["mlp", "decoder"] = "mlp"):

        super().__init__()

        self.head = head
        assert self.head in ["mlp", "decoder"]

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        def encoder_block(in_size, out_size):
            block = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            block.apply(weights_init)
            return block

        def decoder_block(in_size, out_size):
            block = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.UpsamplingNearest2d(scale_factor=2),
            )
            block.apply(weights_init)
            return block

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 32, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            ),
            encoder_block(64, 128),
            encoder_block(128, 256),
            encoder_block(256, 512),
            encoder_block(512, 512),
            encoder_block(512, 512),
        )
        self.encoder.apply(weights_init)

        # head

        if self.head == "mlp":

            self.decoder = nn.Sequential(
                nn.LeakyReLU(inplace=True),
                nn.Linear(64 + 128 + 256 + 512 + 512 + 512, 1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.LeakyReLU(inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(inplace=True),
                nn.Linear(256, 128),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, 64),
                nn.LeakyReLU(inplace=True),
                nn.Linear(64, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.decoder.apply(weights_init)

        else:

            self.decoder = nn.Sequential(
                decoder_block(512, 512),
                decoder_block(512, 512),
                decoder_block(512, 256),
                decoder_block(256, 128),
                decoder_block(128, 64),
                nn.Sequential(
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(32, 1, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                ),
            )
            self.decoder.apply(weights_init)

    def _per_point_features(self, features, orig_mask):
        _, _, h_orig, w_orig = orig_mask.shape
        b, _, h, w = features.shape

        orig_mask = orig_mask[:, 0]

        idxs = torch.stack(
            torch.meshgrid(
                torch.arange(b, device=orig_mask.device),
                torch.arange(h, device=orig_mask.device),
                torch.arange(w, device=orig_mask.device),
            )
        )  # 3 x B x H x W
        idxs = idxs.permute(1, 0, 2, 3)  # B x 3 x H x W
        idxs = F.interpolate(idxs.to(torch.float32), (h_orig, w_orig))
        idxs = idxs.permute(0, 2, 3, 1)[orig_mask]  # N x 3 (batch, height, width)
        idxs = torch.round(idxs).to(torch.long)
        features = features[idxs[:, 0], :, idxs[:, 1], idxs[:, 2]]
        return features

    def forward(self, x):

        _, _, h_orig, w_orig = x.shape

        # encoding with cnn
        stem_block = self.encoder[0](x)
        encoded_2 = self.encoder[1](stem_block)
        encoded_4 = self.encoder[2](encoded_2)
        encoded_8 = self.encoder[3](encoded_4)
        encoded_16 = self.encoder[4](encoded_8)
        encoded_32 = self.encoder[5](encoded_16)

        mask = x[:, -1:] > 0  # B x 1 x H x W
        if self.head == "mlp":
            # for each original lidar point associate to it a feature
            # vector from the output of the encoder
            features = torch.cat(
                [
                    self._per_point_features(stem_block, mask),
                    self._per_point_features(encoded_2, mask),
                    self._per_point_features(encoded_4, mask),
                    self._per_point_features(encoded_8, mask),
                    self._per_point_features(encoded_16, mask),
                    self._per_point_features(encoded_32, mask),
                ],
                -1,
            )

            # now MLP is applied to each of such features
            inv_conf = self.decoder(features)  # N x 1

            # for compatibility such vector is transformed into a sparse map
            output = torch.zeros_like(mask, dtype=inv_conf.dtype)
            output[mask] = inv_conf.squeeze()

            return output  # B x 1 x H(orig) x W(orig)
        else:
            decoded_32 = self.decoder[0](encoded_32)
            decoded_16 = self.decoder[1](decoded_32 + encoded_16)
            decoded_8 = self.decoder[2](decoded_16 + encoded_8)
            decoded_4 = self.decoder[3](decoded_8 + encoded_4)
            decoded_2 = self.decoder[4](decoded_4 + encoded_2)
            inv_conf = self.decoder[5](decoded_2 + stem_block)

            output = torch.zeros_like(mask, dtype=inv_conf.dtype)
            output[mask] = inv_conf[mask]
            return output
