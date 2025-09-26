"""模型优化：FP16量化和ONNX转换"""

import torch
from pathlib import Path
from ultralytics import YOLO


def quantize_fp16(model_path: str, output_path: str = None):
    """
    将模型转换为FP16精度

    参数:
        model_path: 输入.pt模型路径
        output_path: FP16模型输出路径（可选）
    """
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_fp16.pt"

    # 加载并转换为FP16
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint['model'] = checkpoint['model'].half()

    # 保存FP16模型
    torch.save(checkpoint, output_path)

    # 打印结果
    original_size = model_path.stat().st_size / (1024 * 1024)
    fp16_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ FP16模型已保存: {output_path}")
    print(f"大小: {original_size:.1f}MB → {fp16_size:.1f}MB (减少{(1-fp16_size/original_size)*100:.1f}%)")

    return output_path


def convert_to_onnx(model_path: str, output_path: str = None, fp16: bool = False, imgsz: int = 640):
    """
    将YOLOv8模型转换为ONNX格式

    参数:
        model_path: 输入.pt模型路径
        output_path: ONNX模型输出路径（可选）
        fp16: 是否启用FP16精度
        imgsz: 输入图片尺寸
    """
    model_path = Path(model_path)
    if output_path is None:
        suffix = "_fp16.onnx" if fp16 else ".onnx"
        output_path = model_path.parent / f"{model_path.stem}{suffix}"

    # 加载模型并导出为ONNX
    model = YOLO(str(model_path))
    model.export(
        format='onnx',
        imgsz=imgsz,
        half=fp16,
        simplify=True,
        dynamic=False,
        opset=13
    )

    # 移动到目标位置
    exported_path = model_path.parent / f"{model_path.stem}.onnx"
    if exported_path.exists() and str(exported_path) != str(output_path):
        exported_path.rename(output_path)

    # 打印结果
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ ONNX模型已保存: {output_path}")
    print(f"大小: {onnx_size:.1f}MB | FP16: {fp16} | 图片尺寸: {imgsz}")

    return output_path


if __name__ == "__main__":

    # 输入模型路径
    model_path = "fall_detection/models/best.pt"

    # 输出路径：直接保存到fall_detection/models目录
    output_dir = Path("fall_detection/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 直接生成FP16量化的ONNX模型
    output_path = output_dir / "best_fp16.onnx"
    print("正在转换为FP16 ONNX模型...")
    convert_to_onnx(model_path, output_path=str(output_path), fp16=True)

    print(f"\n✅ 完成！FP16 ONNX模型已保存到: {output_path}")