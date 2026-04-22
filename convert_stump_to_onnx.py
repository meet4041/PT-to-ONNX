from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Ultralytics .pt model to ONNX.")
    parser.add_argument("--weights", default="stump_detection.pt", help="Path to the .pt model.")
    parser.add_argument(
        "--output",
        default="stump_detection.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[224, 224],
        help="Inference image size. Use one value (224) or two values (224 224).",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--device", default="cpu", help="Export device, e.g. cpu or 0.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the ONNX graph.")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX.")
    return parser.parse_args()


def normalize_imgsz(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("--imgsz accepts one value or two values only.")


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=normalize_imgsz(args.imgsz),
        opset=args.opset,
        batch=args.batch,
        device=args.device,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    )

    exported_path = Path(exported).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != output:
        shutil.move(str(exported_path), str(output))

    print(output)


if __name__ == "__main__":
    main()
