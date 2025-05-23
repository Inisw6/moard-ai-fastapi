# model/user_embed_model.py
import torch
import torch.nn as nn

class UserEmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.content_rnn = nn.GRU(
            input_size=config['content_embed_dim'],
            hidden_size=config['rnn_hidden_dim'],
            batch_first=True
        )
        self.profile_fc = nn.Linear(config['profile_input_dim'], config['profile_embed_dim'])

        self.output_fc = nn.Linear(
            config['rnn_hidden_dim'] + config['profile_embed_dim'],
            config['user_embed_dim']
        )

    def forward(self, content_embeds, profile_feats):
        _, h_n = self.content_rnn(content_embeds)  # [1, B, rnn_hidden_dim]
        content_repr = h_n.squeeze(0)              # [B, rnn_hidden_dim]
        profile_embed = self.profile_fc(profile_feats)  # [B, profile_embed_dim]

        x = torch.cat([content_repr, profile_embed], dim=1)
        user_embed = self.output_fc(x)             # [B, user_embed_dim]
        return user_embed
