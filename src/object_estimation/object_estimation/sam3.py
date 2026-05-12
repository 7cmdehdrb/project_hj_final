import cv2

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.plotting import Annotator, colors

# Initialize predictors
overrides = dict(conf=0.50, task="segment", mode="predict", model="/home/irol/ros2_ws/src/HJ/2. SAM3/sam3.pt", verbose=False)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor2 = SAM3SemanticPredictor(overrides=overrides)

# Extract features from the first predictor
source = "/home/irol/ros2_ws/src/HJ/saved_virtual_images/virtual_camera_1778483639_1932249.png"
predictor.set_image(source)
src_shape = cv2.imread(source).shape[:2]

# Setup second predictor and reuse features
predictor2.setup_model()

# Perform inference using shared features with text prompt
masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, text=["front face of the box"])

# Perform inference using shared features with bounding box prompt
# masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, bboxes=[[439, 437, 524, 709]])

# Visualize results
if boxes is not None:
    if masks is not None:
        boxes = boxes.cpu().numpy()
        
    im = cv2.imread(source)
    
    num_boxes = len(boxes)
    print(f"인식된 상자(Box) 개수: {num_boxes}개")
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Bounding Box 그리기 (초록색, 두께 2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Bounding Box 좌표 텍스트 추가 (빨간색)
        label = f"[{x1}, {y1}, {x2}, {y2}]"
        cv2.putText(im, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("result", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()