import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07b_label_subgoal_events.py"
SPEC = importlib.util.spec_from_file_location("label_subgoal_events_07b", SCRIPT)
LABELER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(LABELER)


def annotation(sentence_id):
    return {
        "sentence_id": sentence_id,
        "mathematically_correct": "yes",
        "novel": "no",
        "advances_valid_path": "no",
        "primary_label": "neutral_support",
        "evidence": "",
        "state_change": "",
        "needs_review": False,
    }


class ChunkedLabelingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = {
            "root": root,
            "traces": root / "traces",
            "tables": root / "tables",
        }
        self.paths["traces"].mkdir()
        self.paths["tables"].mkdir()
        self.config = {"labeling": {"chunk_sentences": 2}}
        self.trace = {
            "trace_id": "trace-1",
            "sentences": [
                {
                    "sentence_id": f"S{index:04d}",
                    "text": f"Sentence {index}.",
                    "is_reasoning": True,
                }
                for index in range(5)
            ],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_whole_result_is_not_rewritten(self):
        destination = LABELER.full_result_path(self.paths, self.trace["trace_id"])
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps({
            "trace_id": self.trace["trace_id"],
            "source": "original_whole_request",
            "annotations": [annotation(row["sentence_id"]) for row in self.trace["sentences"]],
        }), encoding="utf-8")
        before = destination.read_bytes()
        result = LABELER.materialize_chunked_result(
            self.trace, self.config, self.paths
        )
        self.assertEqual(result["source"], "original_whole_request")
        self.assertEqual(destination.read_bytes(), before)

    def test_complete_chunks_merge_in_original_order(self):
        chunks = LABELER.sentence_chunks(self.trace["sentences"], 2)
        for index, sentences in enumerate(chunks):
            destination = LABELER.chunk_result_path(
                self.paths, self.trace["trace_id"], index
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "trace_id": self.trace["trace_id"],
                "chunk_index": index,
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                "annotations": [annotation(row["sentence_id"]) for row in sentences],
            }), encoding="utf-8")
        result = LABELER.materialize_chunked_result(
            self.trace, self.config, self.paths
        )
        self.assertEqual(result["source"], "chunked")
        self.assertEqual(
            [row["sentence_id"] for row in result["annotations"]],
            [row["sentence_id"] for row in self.trace["sentences"]],
        )
        self.assertEqual(result["usage"]["total_tokens"], 90)


if __name__ == "__main__":
    unittest.main()
