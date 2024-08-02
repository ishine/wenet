from typing import Optional, Dict
import torch
from torch.nn import TransformerDecoder
from wenet.experimental.dmels.dmels_quantizer import DmelsQuantizer

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.common import IGNORE_ID


class DmelsAsrModel(ASRModel):

    def __init__(self,
                 vocab_size: int,
                 encoder: BaseEncoder,
                 decoder: TransformerDecoder,
                 quantizer: DmelsQuantizer,
                 ctc: CTC,
                 ctc_weight: float = 0.5,
                 ignore_id: int = IGNORE_ID,
                 reverse_weight: float = 0,
                 lsm_weight: float = 0,
                 length_normalized_loss: bool = False,
                 special_tokens: Optional[dict] = None,
                 apply_non_blank_embedding: bool = False):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens,
                         apply_non_blank_embedding)
        self.quantizer = quantizer

        bits = self.quantizer.bits
        self.speech_tokens_embed = torch.nn.Embedding(
            bits,
            # TODO: change later
            32)
        # TODO(Mddct): why encoder.output_size?
        self.speech_linear = torch.nn.Linear(32 * bits, encoder.output_size())

    def forward(self, batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)  # (B, T, D)
        B, T, _ = speech.shape
        speech_tokens = self.quantizer(speech.transpose(1, 2)).transpose(
            1, 2)  # (B,T,D)

        # TODO(Mddct): span mask, can work with w2vec mask?

        embed = self.speech_tokens_embed(speech_tokens)  # (B,T,D,d)
        embed = embed.view(B, T, -1)  # (B, T, Dxd)
        embed = self.speech_linear(embed)  # (B, T, encoder_dim)

        batch['feats'] = embed
        return super().forward(batch, device)