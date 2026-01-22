import torch
import argparse
import logging
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

torch.serialization.add_safe_globals([argparse.Namespace])

logger = logging.getLogger("SignStampAgent")

@dataclass
class MarkerDetection:
    """Standardized schema for stamps and signatures."""
    type: str
    confidence: float
    bbox: List[float]
    normalized_bbox: List[float]

class SignStampAgent:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        logger.info(f"Initializing RF-DETR Agent with checkpoint: {checkpoint_path}")
        self.wrapper = RFDETRBase(pretrain_weights=checkpoint_path, device=device)
        current_obj = self.wrapper
        found_engine = False
        for _ in range(3):
            if hasattr(current_obj, 'eval'):
                current_obj.eval(); found_engine = True; break
            elif hasattr(current_obj, 'model'):
                current_obj = current_obj.model
            else: break
        
        if found_engine: logger.info("Engine set to eval mode.")
        self.class_map = {0: "stamp", 1: "signature"}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced Document Preprocessing Pipeline:
        1. Denoising: Preserves handwriting edges while smoothing background noise.
        2. Contrast Enhancement: Makes faint stamps/signatures distinct.
        """
        # A. Edge-Preserving Denoising (Bilateral Filter)
        # 9=diameter, 75=color sigma, 75=space sigma
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # B. Local Contrast Enhancement (CLAHE)
        # Convert to LAB to enhance only luminosity
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        final_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return final_bgr

    def _normalize(self, bbox: np.ndarray, w: int, h: int) -> List[float]:
        return [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]

    def detect_markers(self, image: np.ndarray, threshold: float = 0.5) -> List[MarkerDetection]:
        h, w = image.shape[:2]
     
        logger.info("SignStamp Node: Applying Edge-Preserving Enhancement...")
        processed_img = self._preprocess(image)
        

        img_rgb = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            detections = self.wrapper.predict(img_rgb, threshold=threshold)
        
        results = []
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            if class_id in self.class_map:
                xyxy = detections.xyxy[i]
                results.append(MarkerDetection(
                    type=self.class_map[class_id],
                    confidence=float(detections.confidence[i]),
                    bbox=xyxy.tolist(),
                    normalized_bbox=self._normalize(xyxy, w, h)
                ))
            
        return results
    
if __name__ == "__main__":
    import cv2
    CHECKPOINT = r"C:\Users\Abhi9\OneDrive\Documents\Convolve\stampDetectionModel\checkpoint_best_ema.pth"
    TEST_IMG = r"C:\Users\Abhi9\OneDrive\Documents\Convolve\train\172847544_1_pg23.png"
    
    agent = SignStampAgent(CHECKPOINT)
    img = cv2.imread(TEST_IMG)
    markers = agent.detect_markers(img)
    
    for m in markers:
        print(asdict(m))