import torch

from timm.models.vision_transformer import checkpoint_filter_fn, build_model_with_cfg, partial
import timm.models.vision_transformer


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError(
            'features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn,
                             interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

    def forward(self, x, return_features=False):
        if return_features:
            return self.forward_features(x)
        else:
            return super().forward(x)


def vit_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer(
        'vit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_large_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer(
        'vit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


if __name__ == "__main__":
    model = vit_base_patch16_384(pretrained=True)
    print(model)
    imgs = torch.rand(2, 3, 384, 384)
    out = model(imgs, return_features=True)
    print(out.shape)
