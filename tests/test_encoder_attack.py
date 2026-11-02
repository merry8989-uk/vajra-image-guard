"""Tests for the real encoder attack. Skipped when torch is not installed.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 -m unittest tests.test_encoder_attack -v

No VAE download and no GPU: the PGD loop is tested against a tiny convolution
standing in for an encoder, which is enough to pin down the properties that
matter (the ball, the loss going down, determinism, masking).
"""
import importlib
import os
import sys
import unittest

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class EncoderAttackTest(unittest.TestCase):
    def setUp(self):
        self.sa = importlib.import_module("sd_encoder_attack")
        torch.manual_seed(0)
        # a fixed linear-ish "encoder": 8x8 stride conv, so latent is small
        self.conv = torch.nn.Conv2d(3, 4, 8, stride=8)
        for p in self.conv.parameters():
            p.requires_grad_(False)
        self.encode = lambda x: self.conv(x)

    def _image(self, n=32):
        return (torch.rand(1, 3, n, n) * 2 - 1)

    def test_respects_the_epsilon_ball(self):
        x = self._image()
        z = self.encode(torch.zeros_like(x))
        eps = 0.1
        adv = self.sa.encoder_attack(self.encode, x, z, steps=8, eps=eps, step_size=0.02)
        self.assertLessEqual(float((adv - x).abs().max()), eps + 1e-6)

    def test_reduces_the_latent_distance(self):
        x = self._image()
        z = self.encode(torch.zeros_like(x))
        before = float((self.encode(x) - z).pow(2).sum())
        adv = self.sa.encoder_attack(self.encode, x, z, steps=20, eps=0.2, step_size=0.05)
        after = float((self.encode(adv) - z).pow(2).sum())
        self.assertLess(after, before * 0.9, f"loss {before:.3f} -> {after:.3f}")

    def test_mask_freezes_the_rest_of_the_image(self):
        x = self._image()
        z = self.encode(torch.zeros_like(x))
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, :16, :] = 1.0            # only the top half may change
        adv = self.sa.encoder_attack(self.encode, x, z, steps=6, eps=0.1,
                                     step_size=0.02, mask=mask)
        untouched = (adv - x)[:, :, 16:, :]
        self.assertEqual(float(untouched.abs().max()), 0.0)
        self.assertGreater(float((adv - x).abs().max()), 0.0)

    def test_output_is_deterministic_for_a_fixed_seed(self):
        x = self._image()
        z = self.encode(torch.zeros_like(x))
        torch.manual_seed(7)
        a = self.sa.encoder_attack(self.encode, x, z, steps=4, eps=0.1, step_size=0.02)
        torch.manual_seed(7)
        b = self.sa.encoder_attack(self.encode, x, z, steps=4, eps=0.1, step_size=0.02)
        self.assertTrue(torch.equal(a, b))

    def test_tensor_round_trip_keeps_pixels(self):
        from PIL import Image
        import numpy as np
        arr = (np.random.default_rng(0).random((40, 24, 3)) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        back = self.sa.to_image(self.sa.to_tensor(img, None))
        np.testing.assert_array_equal(np.asarray(back), arr)

    def test_size_is_multiple_of_eight(self):
        from PIL import Image
        import numpy as np
        img = Image.fromarray((np.random.default_rng(1).random((37, 53, 3)) * 255).astype(np.uint8))
        t = self.sa.to_tensor(img, None)
        self.assertEqual(t.shape[2] % 8, 0)
        self.assertEqual(t.shape[3] % 8, 0)

    def test_psnr_sanity(self):
        import numpy as np
        a = np.zeros((8, 8, 3), np.uint8)
        self.assertEqual(self.sa.psnr(a, a), float("inf"))
        self.assertLess(self.sa.psnr(a, a + 10), self.sa.psnr(a, a + 5))


if __name__ == "__main__":
    unittest.main()
