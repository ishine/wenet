from typing import List, Dict, Tuple, Optional

import torch

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.MFFDED.encoder import LayerFusionEncoder
from wenet.utils.common import IGNORE_ID, add_sos_eos, reverse_pad_list
from wenet.transformer.search import (
    ctc_greedy_search,
    ctc_prefix_beam_search,
    attention_beam_search,
    attention_rescoring,
    DecodeResult,
)
from wenet.utils.context_graph import ContextGraph


class MFFDED(ASRModel):

    def __init__(
        self,
        vocab_size: int,
        encoder: LayerFusionEncoder,
        ctc_encoder: TransformerEncoder,
        att_encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: Optional[dict] = None,
        apply_non_blank_embedding: bool = False,
    ):
        super().__init__(
            vocab_size,
            encoder,
            decoder,
            ctc,
            ctc_weight,
            ignore_id,
            reverse_weight,
            lsm_weight,
            length_normalized_loss,
            special_tokens,
            apply_non_blank_embedding,
        )
        self.ctc_encoder = ctc_encoder
        self.att_encoder = att_encoder

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch["feats"].to(device)
        speech_lengths = batch["feats_lengths"].to(device)
        text = batch["target"].to(device)
        text_lengths = batch["target_lengths"].to(device)

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # 1. Shared Layer Fusion Encoder
        encoder_out, encoder_mask, layer_feats = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC Encoder
        ctc_encoder_out, ctc_encoder_mask = self.ctc_encoder(
            encoder_out, encoder_out_lens
        )
        ctc_encoder_out_lens = ctc_encoder_mask.squeeze(1).sum(1)

        # 2b. CTC Decoder
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(
                ctc_encoder_out, ctc_encoder_out_lens, text, text_lengths
            )
        else:
            loss_ctc, ctc_probs = None, None

        # 3. fusion CTC Decoder's probs and Shared Encoder's layer fusion features.
        fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(
                layer_feats.shape[-1] + ctc_probs.shape[-1],
                layer_feats.shape[-1],
            ),
            torch.nn.ReLU(),
        ).to(device)
        fusion_feats = torch.concat((layer_feats, ctc_probs), dim=2)
        fusion_feats = fusion_layer(fusion_feats)

        # 4a. Attention Encoder
        att_encoder_in = torch.concat((encoder_out, fusion_feats), dim=2)
        att_encoder_out, att_encoder_mask = self.att_encoder(
            att_encoder_in, encoder_out_lens
        )

        # 4b. Attention Decoder
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            att_encoder_out, att_encoder_mask = self.filter_blank_embedding(
                ctc_probs, att_encoder_out
            )
        att_decoder_in = torch.concat((att_encoder_out, fusion_feats), dim=2)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                att_decoder_in,
                att_encoder_mask,
                text,
                text_lengths,
                {"langs": batch["langs"], "tasks": batch["tasks"]},
            )
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
        }

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Layer Fusion Encoder
        assert simulate_streaming is False
        encoder_out, encoder_mask, layer_feats = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask, layer_feats

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
    ) -> Dict[str, List[DecodeResult]]:
        """Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # 1. Shared Layer Fusion Encoder
        encoder_out, encoder_mask, layer_feats = self._forward_encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming,
        )
        encoder_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC Encoder
        ctc_encoder_out, ctc_encoder_mask = self.ctc_encoder(
            encoder_out, encoder_out_lens
        )

        # 2. CTC
        ctc_probs = self.ctc_logprobs(ctc_encoder_out, blank_penalty, blank_id)

        # 3. fusion CTC Decoder's probs and Shared Encoder's layer fusion features.
        fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(
                layer_feats.shape[-1] + ctc_probs.shape[-1],
                layer_feats.shape[-1],
            ),
            torch.nn.ReLU(),
        ).to(speech.device)
        fusion_feats = torch.concat((layer_feats, ctc_probs), dim=2)
        fusion_feats = fusion_layer(fusion_feats)

        # 4a. Attention Encoder
        att_encoder_in = torch.concat((encoder_out, fusion_feats), dim=2)
        att_encoder_out, att_encoder_mask = self.att_encoder(
            att_encoder_in, encoder_out_lens
        )
        att_encoder_out = torch.concat((att_encoder_out, fusion_feats), dim=2)

        results = {}
        if "attention" in methods:
            results["attention"] = attention_beam_search(
                self,
                att_encoder_out,
                att_encoder_mask,
                beam_size,
                length_penalty,
                infos,
            )
        if "ctc_greedy_search" in methods:
            results["ctc_greedy_search"] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id
            )
        if "ctc_prefix_beam_search" in methods:
            ctc_prefix_result = ctc_prefix_beam_search(
                ctc_probs, encoder_lens, beam_size, context_graph, blank_id
            )
            results["ctc_prefix_beam_search"] = ctc_prefix_result
        if "attention_rescoring" in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if "ctc_prefix_beam_search" in results:
                ctc_prefix_result = results["ctc_prefix_beam_search"]
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph, blank_id
                )
            if self.apply_non_blank_embedding:
                att_encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, att_encoder_out
                )
            results["attention_rescoring"] = attention_rescoring(
                self,
                ctc_prefix_result,
                att_encoder_out,
                encoder_lens,
                ctc_weight,
                reverse_weight,
                infos,
            )
        return results
