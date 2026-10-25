import tempfile
from pathlib import Path
import unittest
import torch
from semantic_diffusion.model import SemanticDiT, alpha_bar
from train_semantic_diffusion import loss_fn, lr_at, add_confusion, metrics, atomic_save
from utils.losses import Lovasz_softmax


class SemanticDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.model = SemanticDiT(classes=3, dim=32, depth=2, heads=4, patch=(2, 4))
        self.x = torch.randn(2, 5, 16, 32)
        self.valid = torch.ones(2, 16, 32)

    def test_corruption_and_label_codes(self):
        labels = torch.randint(1, 4, (2, 16, 32))
        labels[:, 0] = 0
        clean = self.model.label_codes(labels)
        self.assertEqual(clean[:, :, 0].abs().sum().item(), 0)
        noise = torch.randn_like(clean)
        torch.testing.assert_close(self.model.corrupt(clean, torch.zeros(2), noise), clean)
        torch.testing.assert_close(self.model.corrupt(clean, torch.ones(2), noise), noise)
        self.assertEqual(alpha_bar(torch.zeros(1)).item(), 1)

    def test_validity_loss_gradients_and_condition(self):
        labels = torch.randint(1, 4, (2, 16, 32))
        labels[:, 0] = 0
        z = torch.randn(2, 3, 16, 32)
        cond = self.model.encode(self.x, self.valid)
        logits = self.model.denoise(z, torch.ones(2), cond)
        loss = loss_fn(logits, labels, self.valid, Lovasz_softmax(ignore=-1), 1.)
        loss.backward()
        for module in (self.model.stem, self.model.state, self.model.position, self.model.decoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        alternate = self.model.denoise(z, torch.ones(2), self.model.encode(self.x.flip(-1), self.valid))
        self.assertFalse(torch.allclose(logits, alternate))
        changed = logits.detach().clone()
        changed[:, :, 0] = 100 * torch.randn_like(changed[:, :, 0])
        torch.testing.assert_close(loss.detach(), loss_fn(changed, labels, self.valid, Lovasz_softmax(ignore=-1), 1.))

    def test_sampling_and_reload(self):
        self.model.eval()
        a = self.model.sample(self.x, self.valid, steps=3, seed=7)
        b = self.model.sample(self.x, self.valid, steps=3, seed=7)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pt'
            atomic_save({'model': self.model.state_dict()}, path)
            restored = SemanticDiT(classes=3, dim=32, depth=2, heads=4, patch=(2, 4)).eval()
            restored.load_state_dict(torch.load(path, weights_only=True)['model'])
            torch.testing.assert_close(a, restored.sample(self.x, self.valid, steps=3, seed=7))
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_empty_targets_and_point_metrics(self):
        logits = torch.randn(2, 3, 16, 32, requires_grad=True)
        loss = loss_fn(logits, torch.zeros(2, 16, 32, dtype=torch.long), self.valid, Lovasz_softmax(ignore=-1), 1.)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        cm = torch.zeros(3, 3, dtype=torch.long)
        add_confusion(cm, torch.tensor([0, 1, 2, 2]), torch.tensor([1, 2, 3, 0]))
        self.assertEqual(metrics(cm)['miou'], 1.)

    def test_schedule_and_partial_windows(self):
        self.assertAlmostEqual(lr_at(3, 12, 4, 0.01), 0.01)
        self.assertAlmostEqual(lr_at(11, 12, 4, 0.01), 0.)
        # Attention must mask padding for token grids not divisible by window.
        x = torch.randn(1, 5, 18, 36)
        out = self.model.sample(x, torch.ones(1, 18, 36), steps=1)
        self.assertEqual(out.shape, (1, 3, 18, 36))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
