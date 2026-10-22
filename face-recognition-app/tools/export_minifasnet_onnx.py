"""Dev-only exporter; requires PyTorch and a local Silent-Face-Anti-Spoofing clone."""

import argparse
from collections import OrderedDict
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="path to Silent-Face-Anti-Spoofing clone")
    parser.add_argument("--weights", type=Path, required=True, help="local MiniFASNet .pth file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.repo.is_dir() or not args.weights.is_file():
        parser.error("--repo and --weights must point to existing local paths.")

    sys.path.insert(0, str(args.repo.resolve()))
    try:
        import torch
        from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
        from src.utility import get_kernel, parse_model_name
    except ImportError as error:
        parser.error(f"Could not import the upstream model definition or PyTorch: {type(error).__name__}.")

    height, width, model_type, _ = parse_model_name(args.weights.name)
    if (height, width) != (80, 80):
        parser.error("Only 80x80 MiniFASNet weights are supported by this service.")
    models = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }
    if model_type not in models:
        parser.error(f"Unsupported upstream model type: {model_type}")
    model = models[model_type](conv6_kernel=get_kernel(height, width))
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("module."):
        state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    model.load_state_dict(state)
    model.eval()
    output = args.output or args.weights.with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, 80, 80), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        str(output),
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )
    print(f"Exported {output}")
    print("Expected input: 1x3x80x80 float32; crop is OpenCV BGR; ToTensor-equivalent uint8 / 255; real class index 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())