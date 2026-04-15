import torch
import unittest

from scripts.eval.extract_model_for_hf import (
    LATENT_TOKENS,
    add_latent_tokens,
    checkpoint_vocab_size,
    normalize_checkpoint_state_dict,
    split_coconut_state_dict,
)


class FakeTokenizer:
    def __init__(self, base_tokens, eos_token="<eos>", pad_token=None):
        self._ids = {token: idx for idx, token in enumerate(base_tokens)}
        self.eos_token = eos_token
        self.pad_token = pad_token

    def __len__(self):
        return len(self._ids)

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self._ids:
                self._ids[token] = len(self._ids)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token)


class ExtractModelForHfHelpersTest(unittest.TestCase):
    def test_normalize_checkpoint_state_dict_strips_common_wrappers(self):
        state = {
            "module._orig_mod.base_causallm.model.embed_tokens.weight": torch.zeros(4, 2),
            "_orig_mod.embedding.weight": torch.zeros(4, 2),
        }

        normalized = normalize_checkpoint_state_dict(state)

        self.assertEqual(
            sorted(normalized),
            [
                "base_causallm.model.embed_tokens.weight",
                "embedding.weight",
            ],
        )

    def test_split_coconut_state_dict_separates_auxiliary_wrapper_weights(self):
        state = {
            "base_causallm.model.embed_tokens.weight": torch.zeros(5, 3),
            "base_causallm.lm_head.weight": torch.zeros(5, 3),
            "embedding.weight": torch.zeros(5, 3),
        }

        base_model_state, aux_state = split_coconut_state_dict(state)

        self.assertEqual(
            sorted(base_model_state),
            [
                "lm_head.weight",
                "model.embed_tokens.weight",
            ],
        )
        self.assertEqual(sorted(aux_state), ["embedding.weight"])

    def test_checkpoint_vocab_size_prefers_known_embedding_keys(self):
        base_model_state = {
            "model.language_model.embed_tokens.weight": torch.zeros(8, 3),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(3, 3),
        }

        embed_key, vocab_size = checkpoint_vocab_size(base_model_state)

        self.assertEqual(embed_key, "model.language_model.embed_tokens.weight")
        self.assertEqual(vocab_size, 8)

    def test_add_latent_tokens_grows_tokenizer_to_expected_vocab(self):
        tokenizer = FakeTokenizer(["a", "b", "c"])

        token_ids = add_latent_tokens(tokenizer, expected_vocab_size=6)

        self.assertEqual(len(tokenizer), 6)
        self.assertEqual(set(token_ids), set(LATENT_TOKENS))
        self.assertEqual(token_ids["<|start-latent|>"], 3)
        self.assertEqual(token_ids["<|end-latent|>"], 4)
        self.assertEqual(token_ids["<|latent|>"], 5)

    def test_add_latent_tokens_raises_when_vocab_size_still_mismatched(self):
        tokenizer = FakeTokenizer(["a", "b", "c"])

        with self.assertRaisesRegex(
            ValueError, "Tokenizer length does not match checkpoint"
        ):
            add_latent_tokens(tokenizer, expected_vocab_size=7)


if __name__ == "__main__":
    unittest.main()
