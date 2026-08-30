import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

from rl.environment import XiangqiEnv


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are optional training dependencies")
class XiangqiModelTests(unittest.TestCase):
    def test_autoregressive_policy_respects_legal_action_mask(self):
        from rl.model import build_model

        env = XiangqiEnv()
        observation, info = env.reset()
        model = build_model(channels=32, blocks=1, pair_width=16)
        obs = torch.as_tensor(observation).unsqueeze(0)
        mask = torch.as_tensor(info["action_mask"]).unsqueeze(0)
        source_logits, values = model(obs, mask)
        self.assertEqual(tuple(source_logits.shape), (1, 90))
        self.assertEqual(tuple(values.shape), (1,))
        action, sampled_log_prob, _ = model.act(obs, mask, deterministic=True)
        self.assertTrue(bool(mask[0, action.item()]))
        log_prob, entropy, evaluated_values = model.evaluate_actions(obs, mask, action)
        self.assertTrue(torch.allclose(log_prob, sampled_log_prob))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertEqual(tuple(evaluated_values.shape), (1,))


if __name__ == "__main__":
    unittest.main()
