"""Offline regressions for token constraints and JSON context."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, cast
from unittest.mock import Mock, patch

import numpy as np
from llm_sdk import Small_LLM_Model

from src.decoder import Decoder
from src.parsing import FunctionDefinition, TypeSchema
from src.vocab import Vocab, _load_vocab


class DecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "vocab.json"
        tokens = {chr(i) if i < 128 else f"byte_{i}": i
                  for i in range(256)}
        self.path.write_text(json.dumps(tokens), encoding="utf-8")
        self.model = Mock(spec=Small_LLM_Model)
        self.model.get_path_to_vocab_file.return_value = str(self.path)
        self.model.encode.side_effect = lambda s: np.array(
            [list(s.encode("utf-8"))]
        )
        self.model.decode.side_effect = lambda ids: bytes(ids).decode(
            "utf-8", errors="replace"
        )
        self.decoder = Decoder(
            functions={}, vocab=Vocab(cast(Small_LLM_Model, self.model))
        )

    def emit(self, text: str) -> None:
        tokens = iter(text.encode("utf-8"))

        def logits(ids: List[int]) -> List[float]:
            result = [0.0] * 256
            result[next(tokens, ord(","))] = 10.0
            return result

        self.model.get_logits_from_input_ids.side_effect = logits

    def context(self) -> str:
        return self.decoder.vocab.decode(self.decoder.ids)

    def test_all_impossible_allowed_tokens_raise(self) -> None:
        self.model.get_logits_from_input_ids.return_value = [-np.inf] * 256
        with self.assertRaises(ValueError):
            self.decoder.pick([65, 66])

    def test_nan_does_not_beat_a_finite_allowed_token(self) -> None:
        scores = [0.0] * 256
        scores[65] = float("nan")
        scores[66] = 1.0
        self.model.get_logits_from_input_ids.return_value = scores
        self.assertEqual(self.decoder.pick([65, 66]), 66)

    def test_strings_preserve_spaces_and_unicode(self) -> None:
        for value in ["  hello", "café 😀", 'say "hi"', "line\nend"]:
            with self.subTest(value=value):
                self.decoder.ids = []
                self.emit(json.dumps(value, ensure_ascii=False)[1:])
                result = self.decoder.gen_value(TypeSchema(type="string"))
                self.assertEqual(result, value)
                self.assertEqual(json.loads(self.context()), value)

    def test_invalid_escape_is_repaired_in_context(self) -> None:
        self.emit('bad\\q"')
        result = self.decoder.gen_value(TypeSchema(type="string"))
        self.assertEqual(result, "bad\\q")
        self.assertEqual(json.loads(self.context()), result)

    def test_string_budget_preserves_trailing_backslash(self) -> None:
        self.emit("a\\")
        with patch("src.decoder.MAX_STRING_TOKENS", 2):
            result = self.decoder.gen_value(TypeSchema(type="string"))
        self.assertEqual(result, "a\\")
        self.assertEqual(json.loads(self.context()), result)

    def test_number_context_stays_valid_json(self) -> None:
        for text in ["01,", "1.,", "-01,", "12.50,"]:
            with self.subTest(text=text):
                self.decoder.ids = []
                self.emit(text)
                value = self.decoder.gen_number(integer_only=False)
                self.assertEqual(json.loads(self.context()), value)

    def test_budget_after_minus_repairs_context(self) -> None:
        self.emit("-")
        with patch("src.decoder.MAX_NUMBER_TOKENS", 1):
            self.assertEqual(self.decoder.gen_number(integer_only=True), 0)
        self.assertEqual(json.loads(self.context()), 0)

    def test_function_name_is_json_escaped_in_context(self) -> None:
        fn = FunctionDefinition(
            name='say"hello\\world', description="Example", parameters={},
            returns=TypeSchema(type="null"),
        )
        self.decoder.functions = {fn.name: fn}
        result = self.decoder.run("Say hello")
        context = json.loads(self.context().split("Answer: ", 1)[1])
        self.assertEqual(context["name"], result["name"])

    def test_unicode_digits_are_not_number_tokens(self) -> None:
        self.path.write_text(json.dumps({"0": 48, '"': 34, "²": 178}))
        vocab = Vocab(cast(Small_LLM_Model, self.model))
        self.assertNotIn(178, vocab.digit_ids)

    def test_invalid_vocab_ids_are_rejected(self) -> None:
        for value in [True, -1, "2", 2.5]:
            with self.subTest(value=value):
                self.path.write_text(json.dumps({"0": value}))
                with self.assertRaises(ValueError):
                    _load_vocab(str(self.path))
