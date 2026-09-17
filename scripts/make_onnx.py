import os, sys, argparse
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{code_dir}/../")
import torch
from bridgedepth.bridgedepth import BridgeDepth


class BridgeDepthOnnx(torch.nn.Module):
    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = BridgeDepth.from_pretrained(model)

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor, fx_baseline: torch.Tensor) -> torch.Tensor:
        assert left_image.shape == right_image.shape
        assert left_image.ndim == 3
        assert left_image.dtype == right_image.dtype == torch.uint8
        assert fx_baseline.numel() == 1
        inputs = {"img1": self._preproc(left_image), "img2": self._preproc(right_image)}
        disp = self._model(inputs)["disp_pred"].squeeze(0).float()
        depth = fx_baseline.reshape(1, 1) / disp.clamp_min(1e-4)
        depth = torch.clamp(depth, 0., 65535.)
        assert depth.shape == left_image.shape[:2]
        return depth

    @staticmethod
    def _preproc(x):
        return x.to(torch.float16).permute(2, 0, 1).unsqueeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=f"{code_dir}/../onnx/", help="Path to save results.")
    parser.add_argument("--model_name", choices=["rvc", "rvc_pretrain", "eth3d_pretrain", "middlebury_pretrain"], default="rvc_pretrain")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

    pretrained_model_name_or_path = args.model_name
    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path)
        pretrained_model_name_or_path = args.checkpoint_path
        model_name = os.path.splitext(os.path.basename(pretrained_model_name_or_path))[0]
    else:
        model_name = f"bridge_{args.model_name}"

    device = torch.device(args.device)
    model = BridgeDepthOnnx(pretrained_model_name_or_path)
    model = model.to(device).eval()
    shape = (args.height, args.width, 3)
    img1 = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    img2 = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    fx_baseline = torch.tensor([10.], dtype=torch.float32, device=device)

    opset_version = args.opset
    output_file = os.path.join(args.save_dir, f"{model_name}_{args.height}x{args.width}_opset{opset_version}.onnx")

    print(f"try to export the ONNX (opset {opset_version})...")
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=args.amp):
        torch.onnx.export(
            model,
            args=(img1, img2, fx_baseline),
            f=output_file,
            input_names=["left", "right", "fx_baseline"],
            output_names=["depth"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
            dynamic_shapes=None,
            verify=False,
            profile=False,
            verbose=False,
        )
    print(f"success! ONNX file saved at {output_file}")
