import os
import argparse
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--data', type=str, default='/opt/ml/input/data/train/data.yaml')  
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))

    args = parser.parse_args()

    model = YOLO('yolov11n.pt') 

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    save_path = os.path.join(args.model_dir, 'yolov11_best.pt')
    torch.save(torch.load(best_model_path), save_path)
    print(f"YOLOv11 model saved to: {save_path}")

if __name__ == '__main__':
    main()
