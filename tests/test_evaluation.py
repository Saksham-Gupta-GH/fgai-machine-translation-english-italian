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


class BatchedDecodingTests(unittest.TestCase):
    def test_finished_rows_do_not_generate_extra_words(self):
        import torch
        from src.models import BasicTransformer
        class ScriptedGenerator(torch.nn.Module):
            def __init__(self):super().__init__();self.step=0
            def forward(self, hidden):
                self.step+=1
                logits=torch.zeros((2,8))
                logits[0,3 if self.step==1 else 4]=10
                logits[1,3 if self.step==3 else 5]=10
                return logits
        model=BasicTransformer(vocab_size=8,d_model=8,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=16)
        model.generator=ScriptedGenerator()
        result=model.translate_greedy(torch.tensor([[2,4,3],[2,5,3]]),max_len=4)
        self.assertEqual(result[0].tolist(),[2,3,3,3])
        self.assertEqual(result[1].tolist(),[2,5,5,3])

if __name__=='__main__':unittest.main()
