import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional training dependency")
class CheckpointTests(unittest.TestCase):
    def test_restore_checkpoint_recovers_model_optimizer_and_counters(self):
        from rl.model import build_model
        from rl.train import restore_checkpoint

        device = torch.device("cpu")
        source_model = build_model(channels=16, blocks=1, pair_width=8).to(device)
        source_optimizer = torch.optim.Adam(source_model.parameters(), lr=2.5e-4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "xiangqi_resume.pt"
            torch.save({
                "model": source_model.state_dict(),
                "optimizer": source_optimizer.state_dict(),
                "update": 17,
                "global_step": 4096,
                "completed_episodes": 23,
                "observation_shape": (16, 10, 9),
                "action_size": 8100,
                "policy_format": "autoregressive_cross_attention_pointer_v1",
            }, checkpoint_path)
            restored_model = build_model(channels=16, blocks=1, pair_width=8).to(device)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
            restored = restore_checkpoint(torch, restored_model, restored_optimizer, device, checkpoint_path)

        self.assertEqual(restored, {"update": 17, "global_step": 4096, "completed_episodes": 23})
        for source, recovered in zip(source_model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(source, recovered))
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 2.5e-4)


if __name__ == "__main__":
    unittest.main()
