from abc import abstractmethod
import clip
import mobileclip
import torch.nn as nn
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from ultralytics.utils import LOGGER

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(texts):
        pass
    
    @abstractmethod
    def encode_text(texts, dtype):
        pass

class CLIP(TextModel):
    def __init__(self, device):
        super().__init__()
        self.model = clip.load("ViT-B/32", device=device)[0]
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        return clip.tokenize(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats
        
class MobileCLIP(TextModel):
    def __init__(self, device):
        super().__init__()
        self.model = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='mobileclip_blt.pt', device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_b')
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        max_len = text_tokens.argmax(dim=-1).max().item() + 1
        text_tokens = text_tokens[..., :max_len]
        return text_tokens

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

def build_text_model(variant, device=None):
    LOGGER.info(f"Build text model {variant}")
    if variant == 'clip':
        return CLIP(device)
    elif variant == 'mobileclip':
        return MobileCLIP(device)
    else:
        print("Variant not found")
        assert(False)