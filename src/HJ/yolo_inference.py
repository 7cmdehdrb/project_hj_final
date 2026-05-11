import cv2
from ultralytics import YOLO

# Initialize YOLO model
model_path = "/home/irol/ros2_ws/src/HJ/yolo/best.pt"
model = YOLO(model_path)

# Perform inference
source = "/home/irol/ros2_ws/src/HJ/saved_virtual_images/virtual_camera_1775096914_204359081.png"

# Run the model on the image
results = model(source)
boxes = results[0].boxes

if boxes is not None and len(boxes) > 0:
    boxes_np = boxes.xyxy.cpu().numpy()
    
    im = cv2.imread(source)
    
    num_boxes = len(boxes_np)
    print(f"인식된 상자(Box) 개수: {num_boxes}개")
    
    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Bounding Box 그리기 (초록색, 두께 2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Bounding Box 좌표 텍스트 추가 (빨간색)
        label = f"[{x1}, {y1}, {x2}, {y2}]"
        cv2.putText(im, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("result", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("인식된 상자가 없습니다.")
    im = cv2.imread(source)
    cv2.imshow("result", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
