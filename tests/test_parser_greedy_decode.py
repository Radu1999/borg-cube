"""Tests for the parser's greedy arc decoding."""
import unittest

import torch

from src.models.parser import greedy_decode


class TestGreedyDecode(unittest.TestCase):

    def test_decodes_simple_chain_without_cycles(self):
        # 3 words at sub-word positions 1, 2, 3 (position 0 == ROOT/CLS).
        # Best scoring arcs form a cycle (1->2, 2->1) which must be broken;
        # the decoder should fall back to the next best, non-cycle-forming
        # choice for one of them.
        word_positions = {0: 1, 1: 2, 2: 3}
        scores = torch.full((4, 4), -100.0)
        scores[1, 2] = 10.0  # word0 -> word1 (best, but would cycle)
        scores[2, 1] = 9.0   # word1 -> word0 (best, but would cycle)
        scores[1, 0] = 5.0   # word0 -> ROOT
        scores[2, 0] = 1.0   # word1 -> ROOT
        scores[3, 2] = 8.0   # word2 -> word1
        scores[3, 0] = 0.5   # word2 -> ROOT

        heads = greedy_decode(scores, word_positions)

        # Every word must get exactly one head.
        self.assertEqual(set(heads.keys()), {0, 1, 2})
        # No cycles: following head pointers from any word must reach ROOT.
        for wid in heads:
            visited = set()
            cur = wid
            while heads[cur] != 0:
                self.assertNotIn(cur, visited, "cycle detected")
                visited.add(cur)
                cur = next(w for w, pos in word_positions.items() if pos == heads[cur])

    def test_root_is_used_only_once_when_possible(self):
        # Two words both strongly prefer ROOT as head; only one should win.
        word_positions = {0: 1, 1: 2}
        scores = torch.full((3, 3), -100.0)
        scores[1, 0] = 10.0
        scores[2, 0] = 9.0
        scores[1, 2] = 1.0
        scores[2, 1] = 0.5

        heads = greedy_decode(scores, word_positions)

        root_children = [wid for wid, head_pos in heads.items() if head_pos == 0]
        self.assertEqual(len(root_children), 1)
        self.assertEqual(root_children[0], 0)  # word0 had the higher ROOT score

    def test_fallback_path_still_avoids_cycles_when_possible(self):
        # 4 words whose top edges would all collide into a single cycle,
        # forcing most of them into the fallback assignment path. Even
        # there, the decoder should prefer a cycle-free head whenever one
        # exists for the dependent.
        word_positions = {0: 1, 1: 2, 2: 3, 3: 4}
        scores = torch.full((5, 5), -100.0)
        # A 4-cycle among the words, all with very high (equal) scores.
        scores[1, 2] = 10.0  # word0 -> word1
        scores[2, 3] = 10.0  # word1 -> word2
        scores[3, 4] = 10.0  # word2 -> word3
        scores[4, 1] = 10.0  # word3 -> word0
        # Lower-scoring fallback options that don't extend the cycle.
        scores[1, 0] = 1.0
        scores[2, 0] = 1.0
        scores[3, 0] = 1.0
        scores[4, 0] = 1.0

        heads = greedy_decode(scores, word_positions)

        self.assertEqual(set(heads.keys()), {0, 1, 2, 3})
        # No cycles: following head pointers from any word must reach ROOT.
        for wid in heads:
            visited = set()
            cur = wid
            while heads[cur] != 0:
                self.assertNotIn(cur, visited, "cycle detected")
                visited.add(cur)
                cur = next(w for w, pos in word_positions.items() if pos == heads[cur])

    def test_fallback_does_not_introduce_a_cycle(self):
        # word0 ("A") takes ROOT first (highest score). word1 ("B") and
        # word2 ("C") mutually prefer each other, so whichever is processed
        # second in the main pass would close a 2-cycle and must fall back
        # to a different, cycle-free head instead.
        word_positions = {0: 1, 1: 2, 2: 3}
        scores = torch.full((4, 4), -100.0)
        scores[1, 0] = 100.0  # A -> ROOT (best overall, assigned first)
        scores[2, 3] = 90.0   # B -> C
        scores[3, 2] = 80.0   # C -> B (would close a 2-cycle with B)

        heads = greedy_decode(scores, word_positions)

        self.assertEqual(heads[0], 0)  # A -> ROOT
        self.assertEqual(heads[1], 3)  # B -> C
        self.assertNotEqual(heads[2], 2)  # C must NOT point back to B
        # No cycles: following head pointers from any word must reach ROOT.
        for wid in heads:
            visited = set()
            cur = wid
            while heads[cur] != 0:
                self.assertNotIn(cur, visited, "cycle detected")
                visited.add(cur)
                cur = next(w for w, pos in word_positions.items() if pos == heads[cur])

    def test_every_word_gets_a_head(self):
        word_positions = {0: 1, 1: 2, 2: 3, 3: 4}
        torch.manual_seed(0)
        scores = torch.randn(5, 5)

        heads = greedy_decode(scores, word_positions)

        self.assertEqual(set(heads.keys()), set(word_positions.keys()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_works_with_scores_on_cuda_device(self):
        # Regression test: internal index tensors must be created on the
        # same device as `scores`, otherwise `masked_fill` raises
        # "expected self and mask to be on the same device".
        word_positions = {0: 1, 1: 2, 2: 3}
        scores = torch.randn(4, 4, device="cuda")

        heads = greedy_decode(scores, word_positions)

        self.assertEqual(set(heads.keys()), set(word_positions.keys()))


if __name__ == "__main__":
    unittest.main()
