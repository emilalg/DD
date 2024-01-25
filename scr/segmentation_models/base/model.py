import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.mask_decoder)
        init.initialize_head(self.mask_segmentation_head)
        init.initialize_decoder(self.contour_decoder)
        init.initialize_head(self.contour_segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        mask_decoder_output = self.mask_decoder(*features)
        masks = self.mask_segmentation_head(mask_decoder_output)

        contour_decoder_output = self.contour_decoder(*features)
        contours = self.contour_segmentation_head(contour_decoder_output)

        return masks, contours

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
