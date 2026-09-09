import os, sys, argparse
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import torch
from bridgedepth.bridgedepth import BridgeDepth


class BridgeDepthOnnx(BridgeDepth):
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        assert img1.shape == img2.shape
        assert img1.ndim == 3
        assert img1.dtype == img2.dtype == torch.uint8
        inputs = {'img1': self._preproc(img1), 'img2': self._preproc(img2)}
        results = super().forward(inputs)
        disp = results['disp_pred'].squeeze(0).to(torch.float32).clamp_min(1e-3)
        assert disp.shape == img1.shape[:2]
        return disp

    @staticmethod
    def _preproc(x):
        return x.permute(2, 0, 1).unsqueeze(0).contiguous().float()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=f'{code_dir}/../onnx/', help='Path to save results.')
    parser.add_argument('--model_name', choices=['rvc', 'rvc_pretrain', 'eth3d_pretrain', 'middlebury_pretrain'], default='rvc_pretrain')
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--device', type=str, default='cuda')
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
    model = BridgeDepthOnnx.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device).eval()
    shape = (args.height, args.width, 3)
    img1 = torch.zeros(shape, dtype=torch.uint8, device=device)
    img2 = torch.zeros(shape, dtype=torch.uint8, device=device)

    opset_version = 17
    output_file = os.path.join(args.save_dir, f"{model_name}_opset{opset_version}.onnx")

    print(f"try to export the ONNX (opset {opset_version})...")
    with torch.no_grad(), torch.amp.autocast(device.type):
        torch.onnx.export(
            model,
            args=(img1, img2),
            f=output_file,
            input_names=["left", "right"],
            output_names=["disp"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
            dynamic_shapes=None,
            verify=False,
            profile=False,
            verbose=False,
        )
    print(f"success! ONNX file saved at {output_file}")
