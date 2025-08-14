import os
import cv2
import torch
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords, save_one_box
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator

def detect_from_video(video_path, model_weights, output_dir="inference_output", conf_thres=0.25, img_size=640):
    os.makedirs(output_dir, exist_ok=True)
    device = select_device('')
    model = DetectMultiBackend(model_weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    model.warmup(imgsz=(1, 3, img_size, img_size))  # warmup

    dataset = LoadImages(video_path, img_size=img_size, stride=stride, auto=pt)

    for path, img, im0s, vid_cap, _ in dataset:
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

        for i, det in enumerate(pred):
            p = Path(path)
            frame_name = p.stem
            save_path = os.path.join(output_dir, f"{frame_name}.jpg")
            label_path = os.path.join(output_dir, f"{frame_name}.txt")
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                with open(label_path, "w") as f:
                    for *xyxy, conf, cls in det:
                        annotator.box_label(xyxy, label=f"{names[int(cls)]} {conf:.2f}")
                        x1, y1, x2, y2 = map(int, xyxy)
                        xc = (x1 + x2) / 2 / im0.shape[1]
                        yc = (y1 + y2) / 2 / im0.shape[0]
                        w = (x2 - x1) / im0.shape[1]
                        h = (y2 - y1) / im0.shape[0]
                        f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            cv2.imwrite(save_path, annotator.result())

    print(f"Detections saved to {output_dir}")


if __name__ == "__main__":
    video_file = "path/to/your/video.mp4"
    weights_path = "runs/train/exp3/weights/best.pt"
    detect_from_video(video_file, weights_path)
