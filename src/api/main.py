"""
Dental AI REST API
===================
FastAPI-based REST API for dental treatment planning.

Provides endpoints for:
- X-ray analysis
- Treatment planning
- Cost estimation
- Guidelines search
"""

import os
import json
import base64
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.dental_analyzer import DentalVisionAnalyzer
from rag.dental_rag import DentalGuidelinesRAG


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Dental AI Treatment Planner API",
    description="""
    AI-powered dental treatment planning API with:
    - 🦷 X-ray analysis using YOLOv11
    - 📋 Evidence-based treatment planning with RAG
    - 💰 Cost estimation for Dubai market
    - 📚 Clinical guidelines search
    
    **Author:** Kannan @ iBritz.co.uk
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vision_analyzer = DentalVisionAnalyzer()
rag_system = DentalGuidelinesRAG()


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for X-ray analysis."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    confidence_threshold: float = Field(0.25, ge=0.1, le=0.9)


class TreatmentRequest(BaseModel):
    """Request model for treatment planning."""
    condition: str = Field(..., min_length=1, max_length=100)
    severity: str = Field("moderate")
    tooth_number: Optional[int] = Field(None, ge=11, le=48)


class CostRequest(BaseModel):
    """Request model for cost estimation."""
    treatments: List[str] = Field(..., min_items=1, max_items=10)
    location: str = Field("Dubai")


class GuidelinesRequest(BaseModel):
    """Request model for guidelines search."""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(3, ge=1, le=10)


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    risk_score: float
    detections: list
    summary: dict
    recommendations: list
    processing_time_ms: float


class TreatmentResponse(BaseModel):
    """Response model for treatment plan."""
    success: bool
    condition: str
    severity: str
    primary_treatment: dict
    alternatives: list
    follow_up: list
    prognosis: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    components: dict


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Dental AI Treatment Planner API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components={
            "vision": "ready",
            "rag": "ready",
            "api": "ready"
        }
    )


@app.post("/api/v1/analyze", response_model=dict)
async def analyze_xray(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.25)
):
    """
    Analyze a dental X-ray image.
    
    Upload an image file or provide base64 encoded image data.
    Returns detected conditions, risk score, and recommendations.
    """
    try:
        if file:
            # Read uploaded file
            contents = await file.read()
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(contents))
        elif image_base64:
            # Decode base64
            from PIL import Image
            import io
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Analyze
        result = vision_analyzer.analyze(image)
        
        return {
            "success": True,
            "risk_score": result.risk_score,
            "detections": [d.to_dict() for d in result.detections],
            "summary": result.summary,
            "recommendations": result.recommendations,
            "processing_time_ms": result.processing_time_ms,
            "model_version": result.model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/treatment-plan", response_model=dict)
async def get_treatment_plan(request: TreatmentRequest):
    """
    Generate a treatment plan for a dental condition.
    
    Returns evidence-based treatment recommendations with cost estimates.
    """
    try:
        plan = rag_system.generate_treatment_plan(
            condition=request.condition,
            severity=request.severity,
            patient_factors={"tooth_number": request.tooth_number} if request.tooth_number else None
        )
        
        return {
            "success": True,
            "condition": plan.condition,
            "severity": plan.severity,
            "primary_treatment": plan.primary_treatment.to_dict(),
            "alternatives": [t.to_dict() for t in plan.alternative_treatments],
            "follow_up": plan.follow_up_recommendations,
            "prognosis": plan.prognosis,
            "confidence": plan.confidence_score,
            "sources": plan.guidelines_sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cost-estimate", response_model=dict)
async def get_cost_estimate(request: CostRequest):
    """
    Get cost estimates for dental treatments.
    
    Returns price ranges based on Dubai/UAE market rates.
    """
    TREATMENT_COSTS = {
        "Fluoride Treatment": {"min": 150, "max": 400},
        "Dental Sealant": {"min": 200, "max": 450},
        "Composite Filling": {"min": 400, "max": 1000},
        "Root Canal - Anterior": {"min": 1500, "max": 3500},
        "Root Canal - Molar": {"min": 2500, "max": 5500},
        "Crown - Zirconia": {"min": 2500, "max": 5000},
        "Crown - PFM": {"min": 1500, "max": 3500},
        "Extraction - Simple": {"min": 300, "max": 800},
        "Extraction - Surgical": {"min": 1200, "max": 3500},
        "Dental Implant": {"min": 5000, "max": 12000},
        "Deep Cleaning": {"min": 800, "max": 2500},
    }
    
    estimates = []
    total_min, total_max = 0, 0
    
    for treatment in request.treatments:
        matched = None
        for name, costs in TREATMENT_COSTS.items():
            if treatment.lower() in name.lower() or name.lower() in treatment.lower():
                matched = (name, costs)
                break
        
        if matched:
            name, costs = matched
            estimates.append({
                "treatment": name,
                "min": costs["min"],
                "max": costs["max"],
                "currency": "AED"
            })
            total_min += costs["min"]
            total_max += costs["max"]
        else:
            estimates.append({
                "treatment": treatment,
                "min": None,
                "max": None,
                "note": "Price unavailable"
            })
    
    return {
        "success": True,
        "estimates": estimates,
        "total": {"min": total_min, "max": total_max, "currency": "AED"},
        "location": request.location,
        "disclaimer": "Prices are estimates and may vary by clinic."
    }


@app.post("/api/v1/search-guidelines", response_model=dict)
async def search_guidelines(request: GuidelinesRequest):
    """
    Search dental clinical guidelines.
    
    Returns relevant evidence-based recommendations from professional organizations.
    """
    try:
        results = rag_system.retrieve_guidelines(
            condition=request.query,
            k=request.max_results
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/complete-diagnosis", response_model=dict)
async def complete_diagnosis(
    file: UploadFile = File(...),
    include_costs: bool = Form(True),
    patient_notes: Optional[str] = Form(None)
):
    """
    Complete diagnosis workflow: analyze X-ray and generate treatment plans.
    
    This endpoint combines:
    1. X-ray analysis
    2. Treatment planning for detected conditions
    3. Cost estimation
    4. Comprehensive recommendations
    """
    try:
        # Read and analyze image
        contents = await file.read()
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(contents))
        
        analysis = vision_analyzer.analyze(image)
        
        # Generate treatment plans
        treatment_plans = []
        for detection in analysis.detections:
            plan = rag_system.generate_treatment_plan(
                condition=detection.condition,
                severity=detection.severity or "moderate"
            )
            treatment_plans.append({
                "detection": detection.to_dict(),
                "treatment": plan.to_dict()
            })
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "risk_score": analysis.risk_score,
            "summary": analysis.summary,
            "detections": [d.to_dict() for d in analysis.detections],
            "treatment_plans": treatment_plans,
            "recommendations": analysis.recommendations,
            "patient_notes": patient_notes,
            "processing_time_ms": analysis.processing_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
