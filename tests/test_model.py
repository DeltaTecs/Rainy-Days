import unittest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


class ResMLPTests(unittest.TestCase):
    @unittest.skipUnless(_has_torch(), "torch not installed")
    def test_forward_shape_and_gradients(self) -> None:
        import torch

        from ml.modis_cloud_seeding.model import ResMLP, ResMLPConfig

        config = ResMLPConfig(
            input_dim=5,
            hidden_dim=16,
            num_blocks=2,
            num_classes=3,
            dropout=0.1,
        )
        model = ResMLP(config)
        batch = torch.randn(4, config.input_dim)
        output = model(batch)
        self.assertEqual(output.shape, (4, config.num_classes))

        loss = output.sum()
        loss.backward()
        has_grad = any(parameter.grad is not None for parameter in model.parameters())
        self.assertTrue(has_grad)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
