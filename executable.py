import cv2
from orchestration import build_workflow

# Load invoice image
image = cv2.imread("path/to/invoice.png")

# Initialize pipeline
pipeline = build_workflow()

# Create initial state
state = {
    "image": image,
    "ocr_texts": [],
    "marker_detections": [],
    "agent_results": {},
    "final_output": {}
}

# Run extraction
final_state = pipeline.invoke(state)

# Get results
result = final_state["final_output"]
print(result)