from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer

from .policy import NearestNeighborProjector, TokenCodec


@dataclass(frozen=True)
class GenericTokenizerCodec(TokenCodec):
    tokenizer: object

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


def load_generic_codec_and_projector(
    model_id: str = "distilgpt2",
) -> tuple[GenericTokenizerCodec, NearestNeighborProjector]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    embedding_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()
    projector = NearestNeighborProjector(embedding_matrix)
    return GenericTokenizerCodec(tokenizer), projector
