"""
Dental X-Ray Vision Module
===========================
YOLOv11-based dental condition detection and analysis.

Detects: Caries, Deep Caries, Periapical Lesions, Impacted Teeth, 
         Root Canal Treatments, Crowns, Implants, Missing Teeth

Based on: DentalXrayAI (SubGlitch1) - Enhanced with YOLOv11
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Vision features will be limited.")


class DentalCondition(str, Enum):
    """Dental conditions that can be detected."""
    CARIES = "Caries"
    DEEP_CARIES = "Deep Caries"
    PERIAPICAL_LESION = "Periapical Lesion"
    IMPACTED = "Impacted"
    ROOT_CANAL = "Root Canal"
    CROWN = "Crown"
    IMPLANT = "Implant"
    MISSING = "Missing Tooth"
    FILLING = "Filling"
    BONE_LOSS = "Bone Loss"


class Severity(str, Enum):
    """Severity levels for detected conditions."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class Detection:
    """Represents a single detection from the model."""
    condition: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    tooth_number: Optional[int] = None
    severity: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Complete analysis result for a dental X-ray."""
    image_path: str
    detections: List[Detection]
    summary: Dict[str, Any]
    recommendations: List[str]
    risk_score: float  # 0-100
    processing_time_ms: float
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "detections": [d.to_dict() for d in self.detections],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "risk_score": self.risk_score,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DentalVisionAnalyzer:
    """
    Dental X-Ray Vision Analyzer using YOLOv11.
    
    Features:
    - Multi-class dental condition detection
    - Severity assessment based on detection confidence and size
    - Tooth numbering (FDI notation)
    - Risk scoring algorithm
    - Treatment recommendations
    """
    
    # Class mapping for dental conditions
    CLASS_NAMES = {
        0: DentalCondition.CARIES,
        1: DentalCondition.DEEP_CARIES,
        2: DentalCondition.PERIAPICAL_LESION,
        3: DentalCondition.IMPACTED,
        4: DentalCondition.ROOT_CANAL,
        5: DentalCondition.CROWN,
        6: DentalCondition.IMPLANT,
        7: DentalCondition.FILLING,
    }
    
    # Severity thresholds based on confidence and lesion size
    SEVERITY_THRESHOLDS = {
        "confidence": {
            Severity.MILD: (0.3, 0.5),
            Severity.MODERATE: (0.5, 0.7),
            Severity.SEVERE: (0.7, 0.85),
            Severity.CRITICAL: (0.85, 1.0),
        }
    }
    
    # Risk weights for different conditions
    CONDITION_RISK_WEIGHTS = {
        DentalCondition.CARIES: 15,
        DentalCondition.DEEP_CARIES: 35,
        DentalCondition.PERIAPICAL_LESION: 45,
        DentalCondition.IMPACTED: 25,
        DentalCondition.ROOT_CANAL: 10,  # Treatment, not condition
        DentalCondition.CROWN: 5,  # Treatment, not condition
        DentalCondition.IMPLANT: 5,  # Treatment, not condition
        DentalCondition.FILLING: 5,  # Treatment, not condition
        DentalCondition.BONE_LOSS: 40,
        DentalCondition.MISSING: 20,
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize the Dental Vision Analyzer.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self.model_version = "yolov11n-dental-v1.0"
        
        if model_path and YOLO_AVAILABLE:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load YOLO model from path."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for vision features")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.model_version = f"yolov11-dental-{Path(model_path).stem}"
    
    def _assess_severity(self, confidence: float, bbox_area: float, image_area: float) -> Severity:
        """Assess severity based on confidence and relative lesion size."""
        relative_size = bbox_area / image_area if image_area > 0 else 0
        
        # Combine confidence and size for severity assessment
        severity_score = (confidence * 0.7) + (min(relative_size * 100, 1.0) * 0.3)
        
        if severity_score >= 0.85:
            return Severity.CRITICAL
        elif severity_score >= 0.7:
            return Severity.SEVERE
        elif severity_score >= 0.5:
            return Severity.MODERATE
        else:
            return Severity.MILD
    
    def _estimate_tooth_number(self, bbox: List[float], image_width: int) -> Optional[int]:
        """
        Estimate tooth number based on position (FDI notation).
        This is a simplified estimation based on X-ray position.
        """
        center_x = (bbox[0] + bbox[2]) / 2
        relative_pos = center_x / image_width
        
        # Simplified mapping (panoramic X-ray assumption)
        # Left side: 18-11 (upper), 48-41 (lower)
        # Right side: 21-28 (upper), 31-38 (lower)
        
        if relative_pos < 0.25:
            return 18 - int(relative_pos * 32)  # Upper left molars
        elif relative_pos < 0.5:
            return 11 + int((relative_pos - 0.25) * 28)  # Upper right
        elif relative_pos < 0.75:
            return 31 + int((relative_pos - 0.5) * 28)  # Lower right
        else:
            return 48 - int((relative_pos - 0.75) * 32)  # Lower left
    
    def _calculate_risk_score(self, detections: List[Detection]) -> float:
        """Calculate overall oral health risk score (0-100)."""
        if not detections:
            return 0.0
        
        total_risk = 0.0
        for det in detections:
            condition = DentalCondition(det.condition) if det.condition in [c.value for c in DentalCondition] else None
            if condition:
                weight = self.CONDITION_RISK_WEIGHTS.get(condition, 10)
                severity_multiplier = {
                    Severity.MILD.value: 0.5,
                    Severity.MODERATE.value: 1.0,
                    Severity.SEVERE.value: 1.5,
                    Severity.CRITICAL.value: 2.0,
                }.get(det.severity, 1.0)
                total_risk += weight * severity_multiplier * det.confidence
        
        # Normalize to 0-100 scale
        return min(total_risk, 100.0)
    
    def _generate_recommendations(self, detections: List[Detection]) -> List[str]:
        """Generate treatment recommendations based on detections."""
        recommendations = []
        conditions_found = set()
        
        for det in detections:
            conditions_found.add(det.condition)
        
        # Priority-based recommendations
        if DentalCondition.PERIAPICAL_LESION.value in conditions_found:
            recommendations.append(
                "🚨 URGENT: Periapical lesion detected. Recommend immediate endodontic "
                "evaluation. Root canal therapy or extraction may be required."
            )
        
        if DentalCondition.DEEP_CARIES.value in conditions_found:
            recommendations.append(
                "⚠️ HIGH PRIORITY: Deep caries approaching pulp. Urgent restorative "
                "treatment needed. May require pulp capping or root canal therapy."
            )
        
        if DentalCondition.CARIES.value in conditions_found:
            recommendations.append(
                "📋 Dental caries detected. Schedule restorative treatment (filling). "
                "Consider fluoride therapy for prevention."
            )
        
        if DentalCondition.IMPACTED.value in conditions_found:
            recommendations.append(
                "🦷 Impacted tooth identified. Surgical evaluation recommended. "
                "Consider extraction to prevent complications."
            )
        
        if DentalCondition.BONE_LOSS.value in conditions_found:
            recommendations.append(
                "⚠️ Bone loss detected. Periodontal evaluation required. "
                "Consider scaling, root planing, or surgical intervention."
            )
        
        # General recommendations based on risk
        if not recommendations:
            recommendations.append(
                "✅ No significant pathology detected. Continue regular dental "
                "check-ups every 6 months."
            )
        
        return recommendations
    
    def _generate_summary(self, detections: List[Detection]) -> Dict[str, Any]:
        """Generate analysis summary statistics."""
        summary = {
            "total_findings": len(detections),
            "conditions_breakdown": {},
            "severity_breakdown": {
                Severity.MILD.value: 0,
                Severity.MODERATE.value: 0,
                Severity.SEVERE.value: 0,
                Severity.CRITICAL.value: 0,
            },
            "teeth_affected": [],
            "urgent_attention_needed": False,
        }
        
        for det in detections:
            # Count by condition
            if det.condition not in summary["conditions_breakdown"]:
                summary["conditions_breakdown"][det.condition] = 0
            summary["conditions_breakdown"][det.condition] += 1
            
            # Count by severity
            if det.severity:
                summary["severity_breakdown"][det.severity] += 1
            
            # Track affected teeth
            if det.tooth_number:
                summary["teeth_affected"].append(det.tooth_number)
            
            # Check for urgent conditions
            if det.severity in [Severity.SEVERE.value, Severity.CRITICAL.value]:
                summary["urgent_attention_needed"] = True
            if det.condition in [DentalCondition.PERIAPICAL_LESION.value, 
                                 DentalCondition.DEEP_CARIES.value]:
                summary["urgent_attention_needed"] = True
        
        summary["teeth_affected"] = list(set(summary["teeth_affected"]))
        return summary
    
    def analyze(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_annotated: bool = False
    ) -> AnalysisResult:
        """
        Analyze a dental X-ray image.
        
        Args:
            image: Path to image, numpy array, or PIL Image
            return_annotated: Whether to return annotated image
            
        Returns:
            AnalysisResult with detections and recommendations
        """
        import time
        start_time = time.time()
        
        # Load image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
        elif isinstance(image, Image.Image):
            image_path = "uploaded_image"
            img = np.array(image)
            if img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            image_path = "numpy_array"
            img = image
        
        height, width = img.shape[:2]
        image_area = height * width
        detections = []
        
        # Run YOLO inference if model is loaded
        if self.model is not None:
            results = self.model.predict(
                img,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        bbox = box.xyxy[0].tolist()
                        
                        # Get condition name
                        condition = self.CLASS_NAMES.get(cls_id, f"Unknown-{cls_id}")
                        if isinstance(condition, DentalCondition):
                            condition = condition.value
                        
                        # Calculate bbox area
                        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        # Assess severity
                        severity = self._assess_severity(confidence, bbox_area, image_area)
                        
                        # Estimate tooth number
                        tooth_number = self._estimate_tooth_number(bbox, width)
                        
                        detection = Detection(
                            condition=condition,
                            confidence=round(confidence, 4),
                            bbox=[round(b, 2) for b in bbox],
                            tooth_number=tooth_number,
                            severity=severity.value
                        )
                        detections.append(detection)
        else:
            # Demo mode - return simulated results for testing
            detections = self._generate_demo_detections(width, height)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Generate summary and recommendations
        summary = self._generate_summary(detections)
        recommendations = self._generate_recommendations(detections)
        risk_score = self._calculate_risk_score(detections)
        
        return AnalysisResult(
            image_path=image_path,
            detections=detections,
            summary=summary,
            recommendations=recommendations,
            risk_score=round(risk_score, 2),
            processing_time_ms=round(processing_time, 2),
            model_version=self.model_version
        )
    
    def _generate_demo_detections(self, width: int, height: int) -> List[Detection]:
        """Generate demo detections for testing without model."""
        import random
        
        demo_conditions = [
            (DentalCondition.CARIES.value, 0.78, Severity.MODERATE.value),
            (DentalCondition.DEEP_CARIES.value, 0.65, Severity.SEVERE.value),
            (DentalCondition.FILLING.value, 0.92, Severity.MILD.value),
        ]
        
        detections = []
        for i, (condition, conf, severity) in enumerate(demo_conditions):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            
            detections.append(Detection(
                condition=condition,
                confidence=conf,
                bbox=[x1, y1, x1 + 100, y1 + 100],
                tooth_number=random.choice([14, 15, 16, 24, 25, 26, 36, 37, 46, 47]),
                severity=severity
            ))
        
        return detections
    
    def annotate_image(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Detection],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw detection annotations on the image.
        
        Args:
            image: Input image
            detections: List of detections to draw
            output_path: Optional path to save annotated image
            
        Returns:
            Annotated image as numpy array
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Color map for conditions
        colors = {
            DentalCondition.CARIES.value: (0, 165, 255),  # Orange
            DentalCondition.DEEP_CARIES.value: (0, 0, 255),  # Red
            DentalCondition.PERIAPICAL_LESION.value: (0, 0, 139),  # Dark Red
            DentalCondition.IMPACTED.value: (255, 0, 255),  # Magenta
            DentalCondition.ROOT_CANAL.value: (0, 255, 255),  # Yellow
            DentalCondition.CROWN.value: (255, 255, 0),  # Cyan
            DentalCondition.IMPLANT.value: (0, 255, 0),  # Green
            DentalCondition.FILLING.value: (255, 165, 0),  # Light Blue
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(b) for b in det.bbox]
            color = colors.get(det.condition, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{det.condition} ({det.confidence:.2f})"
            if det.tooth_number:
                label += f" #{det.tooth_number}"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img


# Convenience function for quick analysis
def analyze_dental_xray(
    image_path: str,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Quick function to analyze a dental X-ray.
    
    Args:
        image_path: Path to the dental X-ray image
        model_path: Optional path to custom YOLO model
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = DentalVisionAnalyzer(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    result = analyzer.analyze(image_path)
    return result.to_dict()


if __name__ == "__main__":
    # Demo usage
    print("Dental Vision Analyzer - Demo Mode")
    print("=" * 50)
    
    analyzer = DentalVisionAnalyzer()
    
    # Create a dummy image for testing
    dummy_img = np.zeros((800, 1200, 3), dtype=np.uint8) + 50
    
    result = analyzer.analyze(dummy_img)
    print(result.to_json())
