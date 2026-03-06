import tempfile
import unittest
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from value_function.config import VFTrainConfig
from value_function.encoder import MockObservationEncoder
from value_function.engine import load_checkpoint, save_checkpoint, step_forward
from value_function.model import ValueFunctionHead


class TestModelEncoderEngine(unittest.TestCase):
    def test_mock_encoder_and_head_shape(self):
        encoder = MockObservationEncoder(embedding_dim=64)
        head = ValueFunctionHead(embed_dim=64, hidden_dim=32, num_bins=201)

        images = [[Image.new("RGB", (32, 32), color="red")] for _ in range(4)]
        texts = ["pick up cube"] * 4
        embedding = encoder.encode_batch(images=images, texts=texts, device=torch.device("cpu"))
        logits = head(embedding)

        self.assertEqual(tuple(embedding.shape), (4, 64))
        self.assertEqual(tuple(logits.shape), (4, 201))

    def test_step_forward_and_checkpoint_resume(self):
        device = torch.device("cpu")
        encoder = MockObservationEncoder(embedding_dim=32)
        head = ValueFunctionHead(embed_dim=32, hidden_dim=16, num_bins=201)

        batch = {
            "images": [[Image.new("RGB", (16, 16), color="blue")] for _ in range(3)],
            "texts": ["task a", "task b", "task c"],
            "target_bins": torch.tensor([10, 20, 30], dtype=torch.long),
            "target_values": torch.tensor([-0.9, -0.8, -0.7], dtype=torch.float32),
        }

        logits, target_bins, target_values, loss = step_forward(encoder, head, batch, device)
        self.assertEqual(tuple(logits.shape), (3, 201))
        self.assertEqual(tuple(target_bins.shape), (3,))
        self.assertEqual(tuple(target_values.shape), (3,))
        self.assertGreater(float(loss.item()), 0.0)

        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        cfg = VFTrainConfig(dataset_path="/tmp/fake")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = save_checkpoint(
                output_dir=tmpdir,
                step=7,
                config=cfg,
                head=head,
                encoder=encoder,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            resumed_step = load_checkpoint(ckpt, head=head, encoder=encoder, optimizer=optimizer, scheduler=scheduler)
            self.assertEqual(resumed_step, 7)


if __name__ == "__main__":
    unittest.main()
