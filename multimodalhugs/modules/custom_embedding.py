import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    """
    Embedding layer separated into:
      - old_embeddings (frozen)
      - new_embeddings (trainable)
    where they are only explicitly created in __init__,
    and the copy/initialization logic is performed in build_module().

    If num_new_token = 0, the new_embeddings layer is not created and
    everything is mapped to old_embeddings.
    """

    def __init__(self, used_size, num_new_token, emb_dim):
        """
        used_size: actual number of tokens in the old vocabulary (without padding)
        num_new_token: number of new tokens added
        emb_dim: embedding dimension
        """
        super().__init__()

        # Layer for old tokens (frozen)
        self.old_embeddings = nn.Embedding(used_size, emb_dim)

        # If there are no new tokens, do not create the new_embeddings layer
        self.new_embeddings = None
        if num_new_token > 0:
            self.new_embeddings = nn.Embedding(num_new_token, emb_dim)

        # Store sizes to use in forward
        self.used_size = used_size
        self.num_new_token = num_new_token
        self.emb_dim = emb_dim

    @classmethod
    def build_module(cls, old_embs_weight, backbone_used_vocab_size, num_new_token, emb_dim):
        """
        Class method to build the layer with the following logic:
          1. Determine used_size = min(backbone_used_vocab_size, old_embs_weight.size(0))
          2. Create an instance of CustomEmbedding
          3. Copy the pretrained weights into old_embeddings
          4. (Optional) Initialize new_embeddings (average, specific token, etc.) if it exists
        """

        # 1) Determine how many embeddings were actually used
        old_emb_total = old_embs_weight.size(0)
        used_size = min(backbone_used_vocab_size, old_emb_total)

        # 2) Create an instance of the class
        module = cls(used_size=used_size, num_new_token=num_new_token, emb_dim=emb_dim)

        # 3) Copy the pretrained weights into the 'old' (frozen) part
        module.old_embeddings.weight.data[:] = old_embs_weight[:used_size]

        return module

    def forward(self, input_ids):
        """
        input_ids: tensor [batch_size, seq_len].
        If num_new_token > 0:
          - If token_id < used_size => use old_embeddings
          - If token_id >= used_size => use new_embeddings (reindexed)
        If num_new_token = 0:
          - Everything is mapped to old_embeddings (clamped to [0..used_size-1])
        """
        # If there are no new tokens, return everything from old_embeddings
        if self.num_new_token == 0:
            old_input_ids = input_ids.clamp_min(0).clamp_max(self.used_size - 1)
            return self.old_embeddings(old_input_ids)

        # Otherwise, we have new tokens
        mask_new = (input_ids >= self.used_size)

        # Adjust old tokens
        old_input_ids = input_ids.clamp_min(0).clamp_max(self.used_size - 1)

        # Reindex new tokens
        new_input_ids = (input_ids - self.used_size).clamp_min(0).clamp_max(self.num_new_token - 1)

        old_embeds = self.old_embeddings(old_input_ids)
        new_embeds = self.new_embeddings(new_input_ids)

        # Combine according to the mask
        output_embeds = torch.where(mask_new.unsqueeze(-1), new_embeds, old_embeds)

        return output_embeds
