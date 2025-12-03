import cv2
import torch

# Load the TorchScript model
model = torch.jit.load("best.torchscript.pt")
model.eval()

# Define class names - matching your data.yaml
class_names = {
    0: "Green Buoy",
    1: "Red Buoy",
    2: "Yellow East Buoy",
    3: "Yellow North Buoy",
    4: "Yellow South Buoy",
    5: "Yellow West Buoy"
}

# Define colors for each class (BGR format for OpenCV)
class_colors = {
    0: (0, 255, 0),      # Green
    1: (0, 0, 255),      # Red
    2: (0, 255, 255),    # Yellow (Cyan for visibility)
    3: (0, 200, 255),    # Yellow North (Orange-ish)
    4: (0, 255, 200),    # Yellow South (Yellow-green)
    5: (0, 150, 255)     # Yellow West (Gold)
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR→RGB, HWC→CHW
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        output = model(img)
    
    # Handle the output format
    if isinstance(output, (list, tuple)):
        preds = output[0]
    else:
        preds = output
    
    # Remove batch dimension if present
    if preds.dim() == 3:
        preds = preds[0]
    
    annotated = frame.copy()
    
    # Filter predictions by confidence threshold
    conf_threshold = 0.25
    preds = preds[preds[:, 4] > conf_threshold]
    
    for pred in preds:
        # Extract coordinates, confidence, and class
        x1, y1, x2, y2 = pred[:4].int().tolist()
        conf = pred[4].item()
        cls = int(pred[5].item())
        
        # Scale coordinates back to original frame size
        h, w = frame.shape[:2]
        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)
        
        # Get class name and color
        class_name = class_names.get(cls, f"Class {cls}")
        color = class_colors.get(cls, (255, 255, 255))  # Default to white
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label = f"{class_name}: {conf:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            2
        )

    cv2.imshow("YOLOv5 Buoy Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()