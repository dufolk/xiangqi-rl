import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

from rl.environment import ACTION_SIZE, XiangqiEnv


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are optional training dependencies")
class XiangqiModelTests(unittest.TestCase):
    def test_paired_policy_respects_legal_action_mask(self):
        from rl.model import build_model

        env = XiangqiEnv()
        observation, info = env.reset()
        model = build_model(channels=32, blocks=1, pair_width=16)
        obs = torch.as_tensor(observation).unsqueeze(0)
        mask = torch.as_tensor(info["action_mask"]).unsqueeze(0)
        logits, values = model(obs, mask)
        self.assertEqual(tuple(logits.shape), (1, ACTION_SIZE))
        self.assertEqual(tuple(values.shape), (1,))
        illegal_logits = logits[0, ~mask[0]]
        self.assertTrue(torch.eq(illegal_logits, torch.finfo(logits.dtype).min).all())
        action, _, _ = model.act(obs, mask, deterministic=True)
        self.assertTrue(bool(mask[0, action.item()]))


if __name__ == "__main__":
    unittest.main()
