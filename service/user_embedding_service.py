# service/user_embedding_service.py
import yaml
import torch
from model.user_embed_model import UserEmbeddingModel

class UserEmbeddingService:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = UserEmbeddingModel(self.config)
        self.model.eval()

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def infer(self, content_embeds, profile_feats):
        with torch.no_grad():
            return self.model(content_embeds, profile_feats)
