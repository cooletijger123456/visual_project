import clip
import torch
from ultralytics.nn.text_model import build_text_model

model = clip.load("ViT-B/32")[0]
text = ["dog", "person"]

txt_feats = model.encode_text(clip.tokenize(text).cuda())
txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
print(txt_feats)

model = build_text_model('clip', device='cuda')
txt_feats = model.encode_text(model.tokenize(text))
print(txt_feats)

print(txt_feats.max(dim=-1))

import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

text = ["a dog and a cat", "person"]
text_tokens = tokenizer(text)

with torch.no_grad(), torch.cuda.amp.autocast():
    txt_feats = model.encode_text(text_tokens)
    txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
    print(txt_feats)

model = build_text_model('mobileclip')
txt_feats = model.encode_text(model.tokenize(text))
print(txt_feats)
print(txt_feats.max(dim=-1))