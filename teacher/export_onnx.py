"""Export the trained student to ONNX for the browser app (onnxruntime-web).

  .venv/bin/python export_onnx.py [checkpoints/student.pt] [../models/anime_student.onnx]
Without a checkpoint (--untrained) it exports random weights — only useful for testing the app's plumbing.
"""
import sys
from pathlib import Path
import torch
from student import Student, build

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ckpt = Path(args[0] if args else "checkpoints/student.pt")
out = Path(args[1] if len(args) > 1 else "../models/anime_student.onnx")

if "--untrained" in sys.argv:
    model = Student(24)
else:
    st = torch.load(ckpt, map_location="cpu")
    model = build(st); model.load_state_dict(st["model"])
model.eval()

out.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(model, torch.randn(1, 3, 288, 512), str(out), input_names=["x"], output_names=["y"],
                  opset_version=17, dynamo=False,
                  dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
if "--fp16" in sys.argv:   # half-precision weights/math; inputs and outputs stay float32 for the app
    import onnx
    from onnxconverter_common import float16
    onnx.save(float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True), str(out))
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
