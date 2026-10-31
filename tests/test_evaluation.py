import unittest
from src.evaluate import evaluate_translations, compute_diversity_metrics

class EvaluationTests(unittest.TestCase):
    def test_all_sentences_contribute(self):
        refs=['The dog is here.', 'A completely different sentence.', 'Another entirely distinct reference.']
        self.assertEqual(evaluate_translations(refs, refs)['bleu'], 100)
        degraded=evaluate_translations([refs[0], 'wrong output', 'wrong output'], refs)
        self.assertLess(degraded['bleu'], 50)
        self.assertLess(degraded['chrf'], 100)

    def test_invalid_lengths_rejected(self):
        for predictions,references in [([], []), (['one'], ['one','two'])]:
            with self.assertRaises(ValueError):evaluate_translations(predictions,references)

    def test_no_cross_sentence_bigrams(self):
        self.assertEqual(compute_diversity_metrics(['one','two'])['distinct_2'],0)

if __name__=='__main__':unittest.main()
