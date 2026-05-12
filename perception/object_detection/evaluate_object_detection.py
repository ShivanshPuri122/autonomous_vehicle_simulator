from ultralytics import YOLO
import os

def evaluate_model():
    base_dir = r"d:\Projects\autonomous_vehicle_simulator"
    model_path = os.path.join(base_dir, "models", "object_detection", "weights", "best.pt")
    data_path = os.path.join(base_dir, "config", "object_data.yaml")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print(f"Evaluating Object Detection Model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation on test split
    results = model.val(data=data_path, split='test', plots=True, project=os.path.join(base_dir, 'test_results'), name='object_detection_eval')
    print("\nEvaluation Results:", results.results_dict)

if __name__ == "__main__":
    evaluate_model()
