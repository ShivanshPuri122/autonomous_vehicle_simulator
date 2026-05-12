from ultralytics import YOLO
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    
    print(f"Exporting model: {model_path} to ONNX")
    model = YOLO(model_path)
    
    try:
        # Export to ONNX (FP32)
        onnx_path = model.export(format="onnx", simplify=True)
        print(f"Successfully exported to {onnx_path}")
        
        # Quantize to INT8
        int8_onnx_path = onnx_path.replace(".onnx", "_int8.onnx")
        print(f"Quantizing {onnx_path} to {int8_onnx_path} (INT8)")
        quantize_dynamic(
            onnx_path,
            int8_onnx_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Successfully quantized to {int8_onnx_path}")
    except Exception as e:
        print(f"Failed to export/quantize {model_path} with error: {e}")

if __name__ == "__main__":
    base_dir = r"d:\Projects\autonomous_vehicle_simulator"
    
    lane_model = os.path.join(base_dir, "models", "lane_detection", "weights", "best.pt")
    obj_model = os.path.join(base_dir, "models", "object_detection", "weights", "best.pt")
    
    export_model(lane_model)
    export_model(obj_model)
