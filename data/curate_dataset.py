import os
import cv2
import dlib
import random
import yaml

random.seed(42)
face_detector = dlib.get_frontal_face_detector()

def convert_to_yolo(size, box):
    img_w, img_h = size
    dw = 1. / img_w
    dh = 1. / img_h
    x = (box.left() + box.right()) / 2.0
    y = (box.top() + box.bottom()) / 2.0
    w = box.right() - box.left()
    h = box.bottom() - box.top()
    return (x * dw, y * dh, w * dw, h * dh)

def prepare_directories(base_path):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, 'dataset/cropped_faces', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'dataset/yolo_format/images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'dataset/yolo_format/labels', split), exist_ok=True)

def process_faces(input_root, output_root, val_ratio=0.1):
    for class_name in sorted(os.listdir(input_root)):
        class_path = os.path.join(input_root, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = face_detector(gray, 1)

            if not detections:
                continue

            h, w = img.shape[:2]
            split = 'val' if random.random() < val_ratio else 'train'

            yolo_labels = []
            for i, det in enumerate(detections):
                x, y, w_rel, h_rel = convert_to_yolo((w, h), det)
                yolo_labels.append(f"0 {x:.6f} {y:.6f} {w_rel:.6f} {h_rel:.6f}")  

                top = max(det.top(), 0)
                bottom = min(det.bottom(), h)
                left = max(det.left(), 0)
                right = min(det.right(), w)

                if bottom > top and right > left:
                    crop_img = img[top:bottom, left:right]
                    cropped_face_dir = os.path.join(output_root, 'dataset/cropped_faces', split, class_name)
                    os.makedirs(cropped_face_dir, exist_ok=True)
                    crop_save_path = os.path.join(cropped_face_dir, f"{os.path.splitext(img_name)[0]}_{i}.jpg")
                    cv2.imwrite(crop_save_path, crop_img)

            # Save image + label in YOLO format
            yolo_image_path = os.path.join(output_root, 'dataset/yolo_format/images', split, f"{class_name}_{img_name}")
            yolo_label_dir = os.path.join(output_root, 'dataset/yolo_format/labels', split)
            os.makedirs(yolo_label_dir, exist_ok=True)
            yolo_label_path = os.path.join(yolo_label_dir, f"{class_name}_{os.path.splitext(img_name)[0]}.txt")

            cv2.imwrite(yolo_image_path, img)
            with open(yolo_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

def write_data_yaml(output_root):
    data_yaml = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['face']
    }

    yaml_path = os.path.join(output_root, 'dataset/yolo_format', 'face_data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"✅ data.yaml created at {yaml_path}")

if __name__ == "__main__":
    input_root = r"raw\input\folder\path" # Make sure to give original single full body images.
    output_root = r"raw\output\folder\path" # This will be the output folder where the dataset will be saved.

    prepare_directories(output_root)
    process_faces(input_root, output_root, val_ratio=0.1)
    write_data_yaml(output_root)
