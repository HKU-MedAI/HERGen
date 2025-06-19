from typing import Any
import os
# import tome
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import (GPT2Config, GPT2TokenizerFast,
                          GPT2LMHeadModel, PretrainedConfig, EncoderDecoderModel)
from transformers.modeling_outputs import BaseModelOutput
from hergen.models.base_model import BaseLightningModule
from hergen.backbones.cvt import CvT
from hergen.backbones.vits import vit_base_patch16_384
from hergen.backbones.custom_resnet import get_resnet50
# from hergen.backbones.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


class Cvt2DistilGPT2Module(BaseLightningModule):
    ''' Encoder-decoder architecture '''

    def __init__(self,
                 dataset_name: str,
                 annotation_file: str,
                 dataset_dir: str,
                 exp_log_dir: str,
                 visual_model: str = "microsoft/cvt-21-384-22k",
                 freeze_visual_model: bool = False,
                 language_model: str = "distilgpt2",
                 train_data_pct: float = 1.,
                 max_length: int = 128,
                 batch_size: int = 16,
                 image_size: int = 384,
                 mean: float = 0.,
                 std: float = 1.,
                 num_workers: int = 16,
                 encoder_lr: float = 5e-5,
                 decoder_lr: float = 5e-4,
                 num_beams: int = 3,
                 gpt2_ckpt_path: str = "",
                 *args,
                 **kwargs) -> None:

        # define some rules about visual model and image size
        if "cvt" in visual_model:
            image_size = 384
        elif visual_model == "vit_base_patch16_384":
            image_size = 384
        elif visual_model == "resnet_50":
            image_size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif visual_model == "biovil_t":
            image_size = 448
            mean = 0
            std = 1
        elif visual_model == "biovil":
            image_size = 480
            mean = 0
            std = 1
        elif visual_model in ["gloria", "convirt", "random"]:
            image_size = 224
            mean = 0
            std = 1
        elif visual_model == "gloria_chexpert":
            mean, std = 0.5, 0.5
            image_size = 224
        elif visual_model in ["medclip_vit", "medclip_cnn"]:
            mean, std = 0.5862785803043838, 0.27950088968644304
            image_size = 224
        elif visual_model == "our_medclip":
            mean, std = 0, 1
            image_size = 512
        elif visual_model == "medklip":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            image_size = 224
        elif visual_model == "kad_resnet_224":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            image_size = 224
        elif visual_model == "kad_resnet_512":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            image_size = 512
        else:
            raise NotImplementedError

        super().__init__(dataset_name, annotation_file, dataset_dir, exp_log_dir, language_model, train_data_pct,
                         max_length, num_beams, batch_size, image_size, mean, std, num_workers)

        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr

        self.visual_model = visual_model

        if "cvt" in self.visual_model:
            # Load from the original repo
            self.encoder = CvT(
                warm_start=True,
                model_config='cvt-21-384x384',
                ckpt_name='CvT-21-384x384-IN-22k',
                ckpt_path=os.path.join(
                    REPO_ROOT_DIR, "pretrained/cvt/CvT-21-384x384-IN-22k.pth"),
                is_encoder=True
            )
            emb_dim = 384

            self.encoder_compact = nn.Conv1d(
                in_channels=576, out_channels=50, kernel_size=1)
        
        elif self.visual_model == "resnet_50":
            # self.encoder = get_biovil_image_encoder()
            # emb_dim = self.encoder.feature_size
            self.encoder = get_resnet50(pretrained=False)
            emb_dim = 2048

        elif self.visual_model == "vit_base_patch16_384":
            self.encoder = vit_base_patch16_384(pretrained=True)
            # use tome for token merging
            # tome.patch.timm(self.encoder, trace_source=True)
            # for each stage, remove 40 tokens
            # self.encoder.r = 40
            emb_dim = self.encoder.embed_dim
        
        elif self.visual_model == "biovil_t":
            # Load biovil
            from cxrseg.third_party.biovil.image import get_image_inference
            from cxrseg.third_party.biovil.image.utils import ImageModelType

            image_inference = get_image_inference(ImageModelType.BIOVIL_T)
            self.encoder = image_inference.model
            emb_dim = 512
        
        elif self.visual_model == "biovil":
            # Load biovil
            from cxrseg.third_party.biovil.image import get_image_inference
            from cxrseg.third_party.biovil.image.utils import ImageModelType

            image_inference = get_image_inference(ImageModelType.BIOVIL)
            self.encoder = image_inference.model
            emb_dim = 2048

        elif self.visual_model == "gloria_chexpert":
            from cxrseg.third_party.gloria.load_original_gloria import load_gloria
            model = load_gloria()
            self.encoder = model.img_encoder
            emb_dim = 1024
        
        elif self.visual_model == "medclip_cnn":
            from cxrseg.third_party.medclip import MedCLIPVisionModel, MedCLIPModel
            medclip = MedCLIPModel(vision_cls=MedCLIPVisionModel)
            medclip.from_pretrained(
                input_dir="/home/fywang/Documents/CXRSeg/pretrained/medclip/medclip-resnet")
            self.encoder = medclip.vision_model.model
            emb_dim = 2048

        elif self.visual_model == "medclip_vit":
            from cxrseg.third_party.medclip import MedCLIPVisionModelViT, MedCLIPModel
            medclip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            medclip.from_pretrained(
                input_dir="/home/fywang/Documents/CXRSeg/pretrained/medclip/medclip-vit")
            self.encoder = medclip.vision_model.model
            emb_dim = 768

        elif self.visual_model == "our_medclip":
            from cxrseg.modeling.our_medclip import SILCModule
            # Load the checkpoint
            ckpt = torch.load(
                "/disk1/fywang/CXRSEG/logs/medclip/ckpts/MedCLIP_2024_04_21_14_48_11/epoch=11-step=5040.ckpt",
                map_location=device)
            hyper_parameters = ckpt["hyper_parameters"]
            silc_module = SILCModule(**hyper_parameters).to(device)

            # only load three modules
            img_encoder_ckpt = dict()
            for k, v in ckpt["state_dict"].items():
                if "img_encoder_student" in k:
                    img_encoder_ckpt[k.replace("img_encoder_student.", "")] = v

            silc_module.img_encoder_student.load_state_dict(img_encoder_ckpt)
            self.encoder = silc_module.img_encoder_student
            emb_dim = 2048

        elif self.visual_model == "medklip":
            from cxrseg.third_party.medklip.load_pretrained_medklip import load_pretrained_medklip
            medklip = load_pretrained_medklip(
                model_path="/home/fywang/Documents/CXRSeg/pretrained/MedKLIP", device=device)
            self.encoder = medklip
            emb_dim = 256

        elif self.visual_model == "kad_resnet_224":
            from cxrseg.third_party.kad.A3_CLIP.models.clip_tqn import ModelRes, ModelRes512
            image_encoder = ModelRes(res_base_model='resnet50').to(device)
            checkpoint_path = "/home/fywang/Documents/CXRSeg/pretrained/KAD_Models/KAD_224/best_valid.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            image_state_dict = checkpoint['image_encoder']
            image_encoder.load_state_dict(image_state_dict)
            self.encoder = image_encoder
            emb_dim = 768

        elif self.visual_model == "kad_resnet_512":
            from cxrseg.third_party.kad.A3_CLIP.models.clip_tqn import ModelRes, ModelRes512
            image_encoder = ModelRes512(res_base_model='resnet50').to(device)
            checkpoint_path = "/home/fywang/Documents/CXRSeg/pretrained/KAD_Models/KAD_512/best_valid.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            image_state_dict = checkpoint['image_encoder']
            image_encoder.load_state_dict(image_state_dict)
            self.encoder = image_encoder
            emb_dim = 768

        else:
            raise NotImplementedError

        # Freeze the visual model
        if freeze_visual_model:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder_projection = nn.Linear(emb_dim, 768)

        config = GPT2Config.from_pretrained(self.language_model)
        config.add_cross_attention = True
        config.is_decoder = True

        # define distilGPT2 decoder
        if gpt2_ckpt_path:
            config.vocab_size = config.vocab_size + 2
            decoder = GPT2LMHeadModel(config=config)
            ckpt = torch.load(gpt2_ckpt_path)["state_dict"]
            new_ckpt = dict()
            for k, v in ckpt.items():
                new_k = k.replace("decoder.", "")
                new_ckpt[new_k] = v
            # Load pretrained gpt2
            decoder.load_state_dict(new_ckpt, strict=False)
            print({f"Load GPT2 from {gpt2_ckpt_path}"})
        else:
            decoder = GPT2LMHeadModel.from_pretrained(
                self.language_model, config=config)
            # Resize GPT2 embedding to include padding and beginning of sentence token:
            decoder.resize_token_embeddings(config.vocab_size + 2)
            print("Load pretrained distillgpt2")

       # We don't actually want to use the encoder of the EncoderDecoderModel, create a dummy encoder:
        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = EncoderDecoderModel(
                    encoder=dummy_encoder, decoder=decoder)
        self.decoder = Decoder()

    def setup_tokenizer(self):
        # Decoder tokenizer:
        tokenizer = GPT2TokenizerFast.from_pretrained(self.language_model)
        tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", 'pad_token': '[PAD]'})

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(tokenizer, k + "_id")}')
            else:
                for i, j in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')
        return tokenizer

    def encoder_forward(self, images):
        '''
        Forward step of image encoder
        '''
        if self.dataset_name == "iu_xray":
            images_per_example = images.shape[1]
            images_shape = images.size()
            images = images.view(-1, *images_shape[-3:])

        if "cvt" in self.visual_model:
            image_features = self.encoder(images)['last_hidden_state']
            image_features = image_features.permute(0, 2, 1)
            image_features = self.encoder_compact(image_features)
        elif self.visual_model == "resnet_50":
            _, image_features = self.encoder(images)
        elif self.visual_model == "vit_base_patch16_384":
            image_features = self.encoder(images, return_features=True)
            image_features = image_features[:, 1:]
        elif self.visual_model in ["biovil", "biovil_t"]:
            image_features = self.encoder(images).patch_embeddings
            image_features = rearrange(image_features, "b d h w -> b (h w) d")
        elif self.visual_model in ["gloria_chexpert", "gloria"]:
            _, image_features = self.encoder(images, get_local=True)
            image_features = rearrange(image_features, "b d h w -> b (h w) d")
        elif self.visual_model == "medclip_cnn":
            image_features = self.encoder(images)
            image_features = rearrange(image_features, "b d h w -> b (h w) d")
        elif self.visual_model == "medclip_vit":
            image_features = self.encoder(images)['last_hidden_state']
        elif self.visual_model == "our_medclip":
            image_features = self.encoder(images)
            image_features = rearrange(image_features, "b d h w -> b (h w) d")
        elif self.visual_model == "medklip":
            image_features = self.encoder.module.image_encoder(images)
        elif self.visual_model in ["kad_resnet_224", "kad_resnet_512"]:
            image_features, _ = self.encoder(images)
        else:
            raise NotImplementedError
        
        del images

        # compact the image features
        image_features = self.encoder_projection(image_features)

        # convert back to each image
        if self.dataset_name == "iu_xray":
            image_features_size = image_features.size()
            image_features = image_features.view(
                image_features_size[0] // images_per_example,
                image_features_size[-2] * images_per_example,
                image_features_size[-1],
            )
        else:
            image_features = image_features.clone()

        encoder_outputs = BaseModelOutput(last_hidden_state=image_features)

        return encoder_outputs

    def forward(self, images, input_ids, attention_mask):

        encoder_outputs = self.encoder_forward(images)

        # Teacher forcing: labels are given as input
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits

    def generate(self, num_beams, images):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        encoder_outputs = self.encoder_forward(images)

        outputs = self.decoder.encoder_decoder.generate(
            max_length=self.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )
        return outputs['sequences']

    def configure_optimizers(self) -> Any:
        # This optimizer actually helps
        grouped_parameters = [
            {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
            {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
            {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
        ]

        optimiser = {'optimizer': torch.optim.AdamW(
            grouped_parameters, lr=self.decoder_lr)}

        return optimiser
