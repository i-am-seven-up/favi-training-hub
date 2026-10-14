import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from transformers import CLIPVisionConfig, CLIPVisionModel

from favi_hub.models import Classifier
from favi_hub.train import restore_checkpoint, save_checkpoint

pytestmark = pytest.mark.model


def tiny(method):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    encoder = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    ))
    lora = {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    return Classifier(encoder, method, 42, lora)


@pytest.mark.parametrize("method", ["frozen", "lora"])
def test_only_intended_parameters_update(method):
    model = tiny(method)
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 3, 16, 16)), torch.tensor([0, 1, 2]))
    loss.backward()
    optimizer.step()
    changed = [name for name, parameter in model.named_parameters() if not torch.equal(before[name], parameter)]
    assert any(name.startswith("head.") for name in changed)
    assert all(name.startswith("head.") or (method == "lora" and "lora_" in name) for name in changed)
    if method == "lora":
        assert any("lora_" in name for name in changed)
    else:
        assert not model.encoder.training


def test_head_initialization_matches_between_methods():
    frozen, lora = tiny("frozen"), tiny("lora")
    assert torch.equal(frozen.head.weight, lora.head.weight)


def test_binary_head_has_two_outputs():
    model = Classifier(tiny("frozen").encoder, "frozen", 42, num_classes=2)
    assert model.head.out_features == 2


def test_resume_restores_trainable_optimizer_scheduler_and_rng(tmp_path):
    model = tiny("lora")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    pixels, labels = torch.randn(3, 3, 16, 16), torch.tensor([0, 1, 2])

    def step(current_model, current_optimizer, current_scheduler):
        current_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(current_model(pixels), labels).backward()
        current_optimizer.step()
        current_scheduler.step()

    step(model, optimizer, scheduler)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, {"run": "same"})
    expected_rng = torch.rand(4)
    step(model, optimizer, scheduler)
    expected_output = model(pixels).detach()

    restored = tiny("lora")
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1 / (step + 1))
    assert restore_checkpoint(
        checkpoint, restored, restored_optimizer, restored_scheduler, {"run": "same"}
    ) == 1
    assert torch.equal(torch.rand(4), expected_rng)
    step(restored, restored_optimizer, restored_scheduler)
    torch.testing.assert_close(restored(pixels), expected_output, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler, {"run": "other"})
