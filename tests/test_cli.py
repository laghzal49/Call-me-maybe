"""CLI regressions without downloading a model."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import __main__ as cli
from src.parsing import Prompt


class CliTests(unittest.TestCase):
    def test_empty_input_does_not_load_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "result.json")
            args = argparse.Namespace(
                input="input.json", functions_definition="functions.json",
                output=output, model=cli.DEFAULT_MODEL, verbose=False,
            )
            with (patch.object(cli, "parse_args", return_value=args),
                  patch.object(cli, "parse_prompts", return_value=[]),
                  patch.object(cli, "parse_functions", return_value={}),
                  patch.object(cli, "Small_LLM_Model") as model):
                cli.main()
            model.assert_not_called()
            self.assertEqual(json.loads(Path(output).read_text()), [])

    def test_partial_failure_writes_successes_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "result.json")
            args = argparse.Namespace(
                input="input.json", functions_definition="functions.json",
                output=output, model=cli.DEFAULT_MODEL, verbose=False,
            )
            prompts = [Prompt(prompt="bad"), Prompt(prompt="good")]
            success = {"prompt": "good", "name": "fn", "parameters": {}}
            with (patch.object(cli, "parse_args", return_value=args),
                  patch.object(cli, "parse_prompts", return_value=prompts),
                  patch.object(cli, "parse_functions", return_value={}),
                  patch.object(cli, "Small_LLM_Model"),
                  patch.object(cli, "Vocab"),
                  patch.object(cli, "Decoder") as decoder):
                decoder.return_value.run.side_effect = [
                    ValueError("failed generation"), success,
                ]
                with self.assertRaises(SystemExit) as error:
                    cli.main()
                self.assertEqual(error.exception.code, 1)
            self.assertEqual(json.loads(Path(output).read_text()), [success])
