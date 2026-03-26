import types

import pytest
import torch

from faster_qwen3_tts.generate import fast_generate
from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.sampling import apply_repetition_penalty, sample_logits


def test_repetition_penalty_uses_all_history():
    logits = torch.zeros(1, 1, 10)
    logits[..., 7] = 1.0
    logits[..., 8] = -1.0

    others = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    history = [7] + [others[i % len(others)] for i in range(1, 60)]
    history = torch.tensor(history, dtype=torch.long)

    out = apply_repetition_penalty(logits.clone(), history, repetition_penalty=1.1)
    assert pytest.approx(out[0, 0, 7].item(), rel=1e-6) == 1.0 / 1.1
    assert pytest.approx(out[0, 0, 8].item(), rel=1e-6) == -1.0 * 1.1


def test_sample_logits_respects_explicit_generator_seed():
    logits = torch.tensor([[0.1, 0.2, 3.0]], dtype=torch.float32)

    first_gen = torch.Generator(device=logits.device)
    first_gen.manual_seed(1234)
    first = [
        int(
            sample_logits(
                logits,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                generator=first_gen,
            ).item()
        )
        for _ in range(8)
    ]

    second_gen = torch.Generator(device=logits.device)
    second_gen.manual_seed(1234)
    second = [
        int(
            sample_logits(
                logits,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                generator=second_gen,
            ).item()
        )
        for _ in range(8)
    ]

    third_gen = torch.Generator(device=logits.device)
    third_gen.manual_seed(1235)
    third = [
        int(
            sample_logits(
                logits,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                generator=third_gen,
            ).item()
        )
        for _ in range(8)
    ]

    assert first == second
    assert first != third


def test_predictor_graph_request_seed_splits_graph_and_eager_generators():
    predictor_graph = PredictorGraph.__new__(PredictorGraph)
    predictor_graph.device = "cpu"
    predictor_graph.eager_generator = torch.Generator(device="cpu")
    predictor_graph.device_index = None

    initial_cpu_state = torch.random.get_rng_state()
    predictor_graph.set_request_seed(1234)
    first_graph = torch.randint(0, 1000, (8,))
    first_eager = torch.randint(0, 1000, (8,), generator=predictor_graph.eager_generator)

    predictor_graph.set_request_seed(1234)
    second_graph = torch.randint(0, 1000, (8,))
    second_eager = torch.randint(0, 1000, (8,), generator=predictor_graph.eager_generator)

    torch.random.set_rng_state(initial_cpu_state)

    assert torch.equal(first_graph, second_graph)
    assert torch.equal(first_eager, second_eager)
    assert not torch.equal(first_graph, first_eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast_generate syncs.")
def test_min_new_tokens_suppresses_early_eos():
    class DummyConfig:
        codec_eos_token_id = 1
        num_code_groups = 16
        vocab_size = 5

    class DummyCodePredictor:
        def __init__(self, vocab, hidden, num_codebooks, device):
            self._embeds = torch.nn.ModuleList(
                [torch.nn.Embedding(vocab, hidden).to(device) for _ in range(num_codebooks)]
            )

        def get_input_embeddings(self):
            return list(self._embeds)

    class FixedCodecHead(torch.nn.Module):
        def __init__(self, vocab, eos_id):
            super().__init__()
            self.vocab = vocab
            self.eos_id = eos_id

        def forward(self, x):
            logits = torch.full((x.shape[0], self.vocab), -10.0, device=x.device)
            logits[:, self.eos_id] = 10.0
            logits[:, 0] = 5.0
            return logits

    class DummyTalker:
        def __init__(self, hidden=4, device="cuda"):
            self.config = DummyConfig()
            self.code_predictor = DummyCodePredictor(
                self.config.vocab_size, hidden, self.config.num_code_groups - 1, device
            )
            self._embed = torch.nn.Embedding(self.config.vocab_size, hidden).to(device)
            self.codec_head = FixedCodecHead(self.config.vocab_size, self.config.codec_eos_token_id).to(device)
            self.rope_deltas = torch.zeros(1, 1, device=device)

        def get_input_embeddings(self):
            return self._embed

        def forward(self, inputs_embeds, attention_mask=None, **kwargs):
            device = inputs_embeds.device
            logits = torch.full((1, 1, self.config.vocab_size), -10.0, device=device)
            logits[..., self.config.codec_eos_token_id] = 10.0
            logits[..., 0] = 5.0
            past_hidden = torch.zeros(1, 1, inputs_embeds.shape[-1], device=device)
            past_kv = [(torch.zeros(1, 1, 1, 1, device=device), torch.zeros(1, 1, 1, 1, device=device))]
            return types.SimpleNamespace(
                past_key_values=past_kv,
                past_hidden=past_hidden,
                generation_step=0,
                logits=logits,
            )

    class DummyPredictorGraph:
        def run(self, pred_input):
            return torch.zeros(15, dtype=torch.long, device=pred_input.device)

    class DummyTalkerGraph:
        max_seq_len = 8

        def prefill_kv(self, past_key_values):
            return 1

        def set_generation_state(self, attention_mask, rope_deltas):
            return None

        def run(self, input_embeds, position):
            return input_embeds

    talker = DummyTalker()
    tie = torch.zeros(1, 3, 4, device="cuda")
    tam = torch.ones(1, 3, dtype=torch.long, device="cuda")
    tth = torch.zeros(1, 1, 4, device="cuda")
    tpe = torch.zeros(1, 1, 4, device="cuda")

    codec_ids, _ = fast_generate(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=talker.config,
        predictor_graph=DummyPredictorGraph(),
        talker_graph=DummyTalkerGraph(),
        max_new_tokens=3,
        min_new_tokens=2,
        do_sample=False,
    )

    assert codec_ids is not None
    assert codec_ids.shape[0] >= 2
    eos_id = talker.config.codec_eos_token_id
    assert (codec_ids[:2, 0] == eos_id).sum().item() == 0
