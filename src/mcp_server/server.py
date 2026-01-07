"""
Dental AI MCP Server
======================
Model Context Protocol server for AI-powered dental treatment planning.

Integrates:
- YOLOv11 dental X-ray analysis
- RAG-based clinical guidelines
- Treatment planning and cost estimation

Compatible with: Claude, ChatGPT, and other MCP-enabled AI assistants
"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.dental_analyzer import DentalVisionAnalyzer, AnalysisResult
from rag.dental_rag import DentalGuidelinesRAG, TreatmentPlan


# ============================================================================
# MCP Server Initialization
# ============================================================================

mcp = FastMCP(
    "dental_ai_mcp",
    version="1.0.0",
    description="AI-powered dental treatment planning with X-ray analysis and clinical guidelines"
)

# Global instances
_vision_analyzer: Optional[DentalVisionAnalyzer] = None
_rag_system: Optional[DentalGuidelinesRAG] = None


def get_vision_analyzer() -> DentalVisionAnalyzer:
    """Get or create vision analyzer instance."""
    global _vision_analyzer
    if _vision_analyzer is None:
        model_path = os.environ.get("DENTAL_MODEL_PATH")
        _vision_analyzer = DentalVisionAnalyzer(
            model_path=model_path,
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))
        )
    return _vision_analyzer


def get_rag_system() -> DentalGuidelinesRAG:
    """Get or create RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = DentalGuidelinesRAG()
    return _rag_system


# ============================================================================
# Input Models (Pydantic)
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class AnalyzeXrayInput(BaseModel):
    """Input model for dental X-ray analysis."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    image_path: Optional[str] = Field(
        default=None,
        description="Path to dental X-ray image file (PNG, JPG, DICOM)"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded dental X-ray image"
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.1,
        le=0.9,
        description="Minimum confidence threshold for detections (0.1-0.9)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for structured data"
    )


class GetTreatmentPlanInput(BaseModel):
    """Input model for treatment plan generation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    condition: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Dental condition (e.g., 'Caries', 'Deep Caries', 'Periapical Lesion', 'Impacted')"
    )
    severity: str = Field(
        default="moderate",
        description="Severity level: 'mild', 'moderate', 'severe', or 'critical'"
    )
    tooth_number: Optional[int] = Field(
        default=None,
        ge=11,
        le=48,
        description="FDI tooth number (11-48)"
    )
    patient_age: Optional[int] = Field(
        default=None,
        ge=1,
        le=120,
        description="Patient age for treatment considerations"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class SearchGuidelinesInput(BaseModel):
    """Input model for clinical guidelines search."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query for dental clinical guidelines"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Filter by specific condition"
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of results to return"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class GetCostEstimateInput(BaseModel):
    """Input model for treatment cost estimation."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    treatments: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of treatment names to estimate costs for"
    )
    location: str = Field(
        default="Dubai",
        description="Location for pricing (e.g., 'Dubai', 'Abu Dhabi')"
    )
    currency: str = Field(
        default="AED",
        description="Currency for cost display"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class CompleteDiagnosisInput(BaseModel):
    """Input model for complete diagnosis and treatment planning."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    image_path: Optional[str] = Field(
        default=None,
        description="Path to dental X-ray image"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image"
    )
    patient_notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Additional clinical notes or patient complaints"
    )
    include_cost_estimate: bool = Field(
        default=True,
        description="Include treatment cost estimates"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="dental_analyze_xray",
    annotations={
        "title": "Analyze Dental X-Ray",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def dental_analyze_xray(params: AnalyzeXrayInput) -> str:
    """
    Analyze a dental X-ray image using AI to detect conditions.
    
    Detects dental conditions including:
    - Caries (cavities)
    - Deep caries
    - Periapical lesions
    - Impacted teeth
    - Root canal treatments
    - Crowns and implants
    
    Args:
        params (AnalyzeXrayInput): Input parameters including image path or base64 data
        
    Returns:
        str: Analysis results with detected conditions, risk score, and recommendations
    """
    try:
        analyzer = get_vision_analyzer()
        
        # Handle image input
        if params.image_base64:
            import numpy as np
            from PIL import Image
            import io
            
            image_data = base64.b64decode(params.image_base64)
            image = Image.open(io.BytesIO(image_data))
            result = analyzer.analyze(image)
        elif params.image_path:
            if not os.path.exists(params.image_path):
                return f"Error: Image file not found: {params.image_path}"
            result = analyzer.analyze(params.image_path)
        else:
            return "Error: Either 'image_path' or 'image_base64' must be provided"
        
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return result.to_json()
        else:
            return _format_analysis_markdown(result)
            
    except Exception as e:
        return f"Error analyzing X-ray: {str(e)}"


@mcp.tool(
    name="dental_get_treatment_plan",
    annotations={
        "title": "Get Dental Treatment Plan",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def dental_get_treatment_plan(params: GetTreatmentPlanInput) -> str:
    """
    Generate a comprehensive dental treatment plan based on clinical guidelines.
    
    Provides evidence-based treatment recommendations using RAG from:
    - American Dental Association (ADA) guidelines
    - American Association of Endodontists (AAE)
    - FDI World Dental Federation
    
    Args:
        params (GetTreatmentPlanInput): Condition, severity, and patient factors
        
    Returns:
        str: Complete treatment plan with options, costs, and prognosis
    """
    try:
        rag = get_rag_system()
        
        plan = rag.generate_treatment_plan(
            condition=params.condition,
            severity=params.severity,
            patient_factors={
                "tooth_number": params.tooth_number,
                "age": params.patient_age
            } if params.tooth_number or params.patient_age else None
        )
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(plan.to_dict(), indent=2)
        else:
            return _format_treatment_plan_markdown(plan)
            
    except Exception as e:
        return f"Error generating treatment plan: {str(e)}"


@mcp.tool(
    name="dental_search_guidelines",
    annotations={
        "title": "Search Clinical Guidelines",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def dental_search_guidelines(params: SearchGuidelinesInput) -> str:
    """
    Search dental clinical guidelines and evidence-based recommendations.
    
    Searches knowledge base of dental treatment guidelines from major
    professional organizations for evidence-based recommendations.
    
    Args:
        params (SearchGuidelinesInput): Search query and filters
        
    Returns:
        str: Relevant clinical guidelines and recommendations
    """
    try:
        rag = get_rag_system()
        
        results = rag.retrieve_guidelines(
            condition=params.condition or params.query,
            severity=None,
            k=params.max_results
        )
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(results, indent=2)
        else:
            return _format_guidelines_markdown(results, params.query)
            
    except Exception as e:
        return f"Error searching guidelines: {str(e)}"


@mcp.tool(
    name="dental_get_cost_estimate",
    annotations={
        "title": "Get Treatment Cost Estimate",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def dental_get_cost_estimate(params: GetCostEstimateInput) -> str:
    """
    Get cost estimates for dental treatments in UAE/Dubai market.
    
    Provides estimated price ranges for common dental procedures
    based on Dubai/UAE market rates.
    
    Args:
        params (GetCostEstimateInput): List of treatments and location
        
    Returns:
        str: Cost estimates with price ranges
    """
    # Cost database for Dubai market (AED)
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
        "Teeth Whitening": {"min": 1000, "max": 3000},
        "Veneer - Porcelain": {"min": 2000, "max": 4500},
    }
    
    estimates = []
    total_min = 0
    total_max = 0
    
    for treatment in params.treatments:
        # Find matching treatment
        matched = None
        treatment_lower = treatment.lower()
        for name, costs in TREATMENT_COSTS.items():
            if treatment_lower in name.lower() or name.lower() in treatment_lower:
                matched = (name, costs)
                break
        
        if matched:
            name, costs = matched
            estimates.append({
                "treatment": name,
                "min_cost": costs["min"],
                "max_cost": costs["max"],
                "currency": params.currency
            })
            total_min += costs["min"]
            total_max += costs["max"]
        else:
            estimates.append({
                "treatment": treatment,
                "min_cost": None,
                "max_cost": None,
                "note": "Price not available - please consult clinic"
            })
    
    result = {
        "estimates": estimates,
        "total_range": {
            "min": total_min,
            "max": total_max,
            "currency": params.currency
        },
        "location": params.location,
        "disclaimer": "Prices are estimates and may vary by clinic. Insurance coverage may apply."
    }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    else:
        return _format_cost_estimate_markdown(result)


@mcp.tool(
    name="dental_complete_diagnosis",
    annotations={
        "title": "Complete Diagnosis and Treatment Planning",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def dental_complete_diagnosis(params: CompleteDiagnosisInput) -> str:
    """
    Perform complete dental diagnosis with X-ray analysis and treatment planning.
    
    This is the main workflow tool that:
    1. Analyzes the dental X-ray
    2. Generates treatment plans for detected conditions
    3. Provides cost estimates
    4. Compiles comprehensive recommendations
    
    Args:
        params (CompleteDiagnosisInput): Image and patient information
        
    Returns:
        str: Complete diagnostic report with treatment plans
    """
    try:
        analyzer = get_vision_analyzer()
        rag = get_rag_system()
        
        # Step 1: Analyze X-ray
        if params.image_base64:
            import numpy as np
            from PIL import Image
            import io
            
            image_data = base64.b64decode(params.image_base64)
            image = Image.open(io.BytesIO(image_data))
            analysis = analyzer.analyze(image)
        elif params.image_path:
            if not os.path.exists(params.image_path):
                return f"Error: Image file not found: {params.image_path}"
            analysis = analyzer.analyze(params.image_path)
        else:
            return "Error: Either 'image_path' or 'image_base64' must be provided"
        
        # Step 2: Generate treatment plans for each detection
        treatment_plans = []
        for detection in analysis.detections:
            plan = rag.generate_treatment_plan(
                condition=detection.condition,
                severity=detection.severity or "moderate"
            )
            treatment_plans.append({
                "detection": detection.to_dict(),
                "treatment_plan": plan.to_dict()
            })
        
        # Step 3: Compile cost estimates
        treatments_needed = []
        for plan_data in treatment_plans:
            treatments_needed.append(plan_data["treatment_plan"]["primary_treatment"]["name"])
        
        cost_data = None
        if params.include_cost_estimate and treatments_needed:
            # Get unique treatments
            unique_treatments = list(set(treatments_needed))
            cost_input = GetCostEstimateInput(
                treatments=unique_treatments,
                response_format=ResponseFormat.JSON
            )
            cost_result = await dental_get_cost_estimate(cost_input)
            cost_data = json.loads(cost_result)
        
        # Step 4: Compile complete report
        report = {
            "report_date": datetime.now().isoformat(),
            "analysis_summary": analysis.summary,
            "risk_score": analysis.risk_score,
            "detections": [d.to_dict() for d in analysis.detections],
            "treatment_plans": treatment_plans,
            "cost_estimates": cost_data,
            "overall_recommendations": analysis.recommendations,
            "patient_notes": params.patient_notes,
            "model_version": analysis.model_version,
            "processing_time_ms": analysis.processing_time_ms
        }
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(report, indent=2)
        else:
            return _format_complete_report_markdown(report)
            
    except Exception as e:
        return f"Error in diagnosis: {str(e)}"


# ============================================================================
# Formatting Helpers
# ============================================================================

def _format_analysis_markdown(result: AnalysisResult) -> str:
    """Format analysis result as markdown."""
    lines = [
        "# 🦷 Dental X-Ray Analysis Report",
        "",
        f"**Risk Score:** {result.risk_score}/100",
        f"**Processing Time:** {result.processing_time_ms}ms",
        f"**Model:** {result.model_version}",
        "",
        "## Findings",
        ""
    ]
    
    if not result.detections:
        lines.append("✅ No significant pathology detected.")
    else:
        for i, det in enumerate(result.detections, 1):
            severity_icon = {
                "mild": "🟢",
                "moderate": "🟡",
                "severe": "🟠",
                "critical": "🔴"
            }.get(det.severity, "⚪")
            
            lines.append(f"### {i}. {det.condition}")
            lines.append(f"- **Severity:** {severity_icon} {det.severity}")
            lines.append(f"- **Confidence:** {det.confidence*100:.1f}%")
            if det.tooth_number:
                lines.append(f"- **Tooth:** #{det.tooth_number}")
            lines.append("")
    
    lines.append("## Summary")
    lines.append(f"- Total findings: {result.summary['total_findings']}")
    lines.append(f"- Urgent attention needed: {'Yes' if result.summary['urgent_attention_needed'] else 'No'}")
    
    lines.append("")
    lines.append("## Recommendations")
    for rec in result.recommendations:
        lines.append(f"- {rec}")
    
    return "\n".join(lines)


def _format_treatment_plan_markdown(plan: TreatmentPlan) -> str:
    """Format treatment plan as markdown."""
    lines = [
        "# 📋 Dental Treatment Plan",
        "",
        f"**Condition:** {plan.condition}",
        f"**Severity:** {plan.severity}",
        f"**Confidence:** {plan.confidence_score*100:.0f}%",
        "",
        "## Recommended Treatment",
        "",
        f"### {plan.primary_treatment.name}",
        f"*Category: {plan.primary_treatment.category}*",
        "",
        plan.primary_treatment.description,
        "",
        "**Indications:**"
    ]
    
    for ind in plan.primary_treatment.indications:
        lines.append(f"- {ind}")
    
    lines.append("")
    lines.append("**Procedure Steps:**")
    for i, step in enumerate(plan.primary_treatment.procedure_steps, 1):
        lines.append(f"{i}. {step}")
    
    cost = plan.primary_treatment.estimated_cost_range
    lines.append("")
    lines.append(f"**Estimated Cost:** {cost['currency']} {cost['min']:,} - {cost['max']:,}")
    lines.append(f"**Recovery Time:** {plan.primary_treatment.recovery_time}")
    lines.append(f"**Success Rate:** {plan.primary_treatment.success_rate}%")
    
    if plan.alternative_treatments:
        lines.append("")
        lines.append("## Alternative Treatments")
        for alt in plan.alternative_treatments:
            lines.append(f"- **{alt.name}** ({alt.category})")
    
    lines.append("")
    lines.append("## Follow-Up")
    for fu in plan.follow_up_recommendations:
        lines.append(f"- {fu}")
    
    lines.append("")
    lines.append(f"## Prognosis")
    lines.append(plan.prognosis)
    
    return "\n".join(lines)


def _format_guidelines_markdown(results: List[Dict], query: str) -> str:
    """Format guidelines search results as markdown."""
    lines = [
        "# 📚 Clinical Guidelines Search Results",
        "",
        f"**Query:** {query}",
        f"**Results:** {len(results)}",
        ""
    ]
    
    for i, result in enumerate(results, 1):
        lines.append(f"## Result {i}")
        lines.append(f"**Relevance:** {result['relevance_score']*100:.0f}%")
        lines.append(f"**Condition:** {result['metadata'].get('condition', 'N/A')}")
        lines.append("")
        lines.append(result['content'][:1000])
        if len(result['content']) > 1000:
            lines.append("...")
        lines.append("")
        lines.append("---")
    
    return "\n".join(lines)


def _format_cost_estimate_markdown(data: Dict) -> str:
    """Format cost estimates as markdown."""
    lines = [
        "# 💰 Treatment Cost Estimates",
        "",
        f"**Location:** {data['location']}",
        "",
        "## Individual Treatments",
        ""
    ]
    
    for est in data['estimates']:
        if est.get('min_cost') is not None:
            lines.append(f"- **{est['treatment']}:** {est['currency']} {est['min_cost']:,} - {est['max_cost']:,}")
        else:
            lines.append(f"- **{est['treatment']}:** {est.get('note', 'Price unavailable')}")
    
    total = data['total_range']
    lines.append("")
    lines.append(f"## Total Estimate")
    lines.append(f"**Range:** {total['currency']} {total['min']:,} - {total['max']:,}")
    
    lines.append("")
    lines.append(f"*{data['disclaimer']}*")
    
    return "\n".join(lines)


def _format_complete_report_markdown(report: Dict) -> str:
    """Format complete diagnosis report as markdown."""
    lines = [
        "# 🏥 Complete Dental Diagnosis Report",
        "",
        f"**Date:** {report['report_date'][:10]}",
        f"**Risk Score:** {report['risk_score']}/100",
        "",
        "---",
        ""
    ]
    
    # Summary
    lines.append("## Executive Summary")
    summary = report['analysis_summary']
    lines.append(f"- **Total Findings:** {summary['total_findings']}")
    lines.append(f"- **Urgent Attention:** {'⚠️ Yes' if summary['urgent_attention_needed'] else '✅ No'}")
    
    if summary.get('conditions_breakdown'):
        lines.append("- **Conditions Found:**")
        for cond, count in summary['conditions_breakdown'].items():
            lines.append(f"  - {cond}: {count}")
    
    # Detections and Treatment Plans
    if report['treatment_plans']:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Findings and Treatment Plans")
        
        for i, plan_data in enumerate(report['treatment_plans'], 1):
            det = plan_data['detection']
            plan = plan_data['treatment_plan']
            
            lines.append(f"")
            lines.append(f"### {i}. {det['condition']}")
            lines.append(f"**Severity:** {det['severity']} | **Confidence:** {det['confidence']*100:.0f}%")
            if det.get('tooth_number'):
                lines.append(f"**Tooth:** #{det['tooth_number']}")
            
            lines.append("")
            lines.append(f"**Recommended Treatment:** {plan['primary_treatment']['name']}")
            lines.append(f"- {plan['primary_treatment']['description']}")
            cost = plan['primary_treatment']['estimated_cost_range']
            lines.append(f"- **Cost:** {cost['currency']} {cost['min']:,} - {cost['max']:,}")
    
    # Cost Summary
    if report.get('cost_estimates'):
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Cost Summary")
        total = report['cost_estimates']['total_range']
        lines.append(f"**Total Estimated Cost:** {total['currency']} {total['min']:,} - {total['max']:,}")
    
    # Recommendations
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    for rec in report['overall_recommendations']:
        lines.append(f"- {rec}")
    
    # Patient Notes
    if report.get('patient_notes'):
        lines.append("")
        lines.append("## Clinical Notes")
        lines.append(report['patient_notes'])
    
    # Disclaimer
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This report is generated by AI and should be reviewed by a qualified dental professional. "
                 "It is not intended to replace professional medical advice, diagnosis, or treatment.*")
    
    return "\n".join(lines)


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dental AI MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable_http"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    if args.transport == "streamable_http":
        mcp.run(transport="streamable_http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
