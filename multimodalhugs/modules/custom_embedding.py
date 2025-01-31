import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    """
    Capa de embeddings separada en:
      - old_embeddings (congelada)
      - new_embeddings (entrenable)
    donde sólo se crean de manera explícita en __init__,
    y la lógica de copiado/inicialización se hace en build_module().

    Si num_new_token = 0, no se crea la capa new_embeddings y
    todo queda mapeado a old_embeddings.
    """

    def __init__(self, used_size, num_new_token, emb_dim):
        """
        used_size: número real de tokens del vocabulario antiguo (sin 'relleno')
        num_new_token: número de nuevos tokens añadidos
        emb_dim: dimensión de los embeddings
        """
        super().__init__()

        # Capa para tokens antiguos (congelada)
        self.old_embeddings = nn.Embedding(used_size, emb_dim)

        # Si no hay tokens nuevos, no creamos la capa new_embeddings
        self.new_embeddings = None
        if num_new_token > 0:
            self.new_embeddings = nn.Embedding(num_new_token, emb_dim)

        # Almacenar tamaños para usar en forward
        self.used_size = used_size
        self.num_new_token = num_new_token
        self.emb_dim = emb_dim

    @classmethod
    def build_module(cls, old_embs_weight, backbone_used_vocab_size, num_new_token, emb_dim):
        """
        Método de clase para construir la capa con la lógica de:
          1. Determinar used_size = min(backbone_used_vocab_size, old_embs_weight.size(0))
          2. Crear la instancia de CustomEmbedding
          3. Copiar los pesos antiguos en old_embeddings
          4. (Opcional) Inicializar new_embeddings (avg, token concreto, etc.), si existe
        """

        # 1) Determinar cuántos embeddings se usaron realmente
        old_emb_total = old_embs_weight.size(0)
        used_size = min(backbone_used_vocab_size, old_emb_total)

        # 2) Crear instancia de la clase
        module = cls(used_size=used_size, num_new_token=num_new_token, emb_dim=emb_dim)

        # 3) Copiar los pesos preentrenados a la parte "vieja" (congelada)
        module.old_embeddings.weight.data[:] = old_embs_weight[:used_size]

        return module

    def forward(self, input_ids):
        """
        input_ids: tensor [batch_size, seq_len].
        Si num_new_token > 0:
          - Si token_id < used_size => old_embeddings
          - Si token_id >= used_size => new_embeddings (reindexado)
        Si num_new_token = 0:
          - Todo se mapea a old_embeddings (clampeado a [0..used_size-1])
        """
        # Si no hay tokens nuevos, devolvemos todo de old_embeddings
        if self.num_new_token == 0:
            old_input_ids = input_ids.clamp_min(0).clamp_max(self.used_size - 1)
            return self.old_embeddings(old_input_ids)

        # De lo contrario, sí tenemos tokens nuevos
        mask_new = (input_ids >= self.used_size)

        # Ajustamos tokens viejos
        old_input_ids = input_ids.clamp_min(0).clamp_max(self.used_size - 1)

        # Reindexamos tokens nuevos
        new_input_ids = (input_ids - self.used_size).clamp_min(0).clamp_max(self.num_new_token - 1)

        old_embeds = self.old_embeddings(old_input_ids)
        new_embeds = self.new_embeddings(new_input_ids)

        # Combinar según la máscara
        output_embeds = torch.where(mask_new.unsqueeze(-1), new_embeds, old_embeds)

        return output_embeds
