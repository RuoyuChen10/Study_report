import sys
sys.path.append('/data1/yjgroup/tym/lab_sync_mac/CLIP/CLIP/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

class ViT_CLIP(nn.Module):
    def __init__(self, class_num = 8):
        super(ViT_CLIP, self).__init__()
        self.encoder = clip_model.encode_image

        self.fc = nn.Linear(768, class_num)
        # softmax 1 * 1 * 1000

    def forward(self, x):

        out = self.encoder(x) # 222
        out = self.fc(out.float())

        return out