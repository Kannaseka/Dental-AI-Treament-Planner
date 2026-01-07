"""
Dental AI Gradio Demo
======================
Interactive web interface for dental treatment planning.
"""

import os
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.dental_analyzer import DentalVisionAnalyzer
from rag.dental_rag import DentalGuidelinesRAG


# Initialize components
vision_analyzer = DentalVisionAnalyzer()
rag_system = DentalGuidelinesRAG()


def analyze_xray(image, confidence_threshold):
    """Analyze uploaded dental X-ray."""
    if image is None:
        return "Please upload an image", "", ""
    
    try:
        # Analyze image
        result = vision_analyzer.analyze(image)
        
        # Format detections
        detections_text = "## 🔍 Detected Conditions\n\n"
        if not result.detections:
            detections_text += "✅ No significant pathology detected.\n"
        else:
            for i, det in enumerate(result.detections, 1):
                severity_color = {
                    "mild": "🟢",
                    "moderate": "🟡", 
                    "severe": "🟠",
                    "critical": "🔴"
                }.get(det.severity, "⚪")
                
                detections_text += f"### {i}. {det.condition}\n"
                detections_text += f"- **Severity:** {severity_color} {det.severity}\n"
                detections_text += f"- **Confidence:** {det.confidence*100:.1f}%\n"
                if det.tooth_number:
                    detections_text += f"- **Tooth #:** {det.tooth_number}\n"
                detections_text += "\n"
        
        # Risk score
        risk_text = f"## 📊 Risk Assessment\n\n**Overall Risk Score:** {result.risk_score}/100\n\n"
        
        if result.risk_score >= 70:
            risk_text += "⚠️ **HIGH RISK** - Immediate dental consultation recommended\n"
        elif result.risk_score >= 40:
            risk_text += "🟡 **MODERATE RISK** - Schedule dental appointment soon\n"
        else:
            risk_text += "🟢 **LOW RISK** - Continue regular dental check-ups\n"
        
        # Recommendations
        rec_text = "## 💡 Recommendations\n\n"
        for rec in result.recommendations:
            rec_text += f"- {rec}\n"
        
        return detections_text, risk_text, rec_text
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def get_treatment_plan(condition, severity):
    """Generate treatment plan for a condition."""
    if not condition:
        return "Please select a condition"
    
    try:
        plan = rag_system.generate_treatment_plan(condition, severity)
        
        output = f"# 📋 Treatment Plan for {condition}\n\n"
        output += f"**Severity:** {severity}\n"
        output += f"**Confidence:** {plan.confidence_score*100:.0f}%\n\n"
        
        output += "## Recommended Treatment\n\n"
        output += f"### {plan.primary_treatment.name}\n"
        output += f"*{plan.primary_treatment.category}*\n\n"
        output += f"{plan.primary_treatment.description}\n\n"
        
        output += "**Procedure Steps:**\n"
        for i, step in enumerate(plan.primary_treatment.procedure_steps, 1):
            output += f"{i}. {step}\n"
        
        cost = plan.primary_treatment.estimated_cost_range
        output += f"\n**Estimated Cost:** {cost['currency']} {cost['min']:,} - {cost['max']:,}\n"
        output += f"**Success Rate:** {plan.primary_treatment.success_rate}%\n"
        output += f"**Recovery Time:** {plan.primary_treatment.recovery_time}\n\n"
        
        output += "## Prognosis\n"
        output += f"{plan.prognosis}\n\n"
        
        output += "## Follow-Up\n"
        for fu in plan.follow_up_recommendations:
            output += f"- {fu}\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


def search_guidelines(query):
    """Search clinical guidelines."""
    if not query:
        return "Please enter a search query"
    
    try:
        results = rag_system.retrieve_guidelines(query, k=3)
        
        output = f"# 📚 Guidelines for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"## Result {i}\n"
            output += f"**Relevance:** {result['relevance_score']*100:.0f}%\n\n"
            output += f"{result['content'][:800]}...\n\n"
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(
    title="🦷 Dental AI Treatment Planner",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; margin-bottom: 20px; }
    .footer { text-align: center; margin-top: 20px; font-size: 0.9em; color: #666; }
    """
) as demo:
    
    gr.Markdown("""
    # 🦷 Dental AI Treatment Planner
    ### AI-Powered Dental Diagnosis & Treatment Planning
    
    Upload dental X-rays for AI analysis, get treatment recommendations based on clinical guidelines,
    and estimate treatment costs for the Dubai market.
    
    ---
    """)
    
    with gr.Tabs():
        # Tab 1: X-Ray Analysis
        with gr.Tab("📸 X-Ray Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Upload Dental X-Ray",
                        type="pil",
                        height=400
                    )
                    confidence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    analyze_btn = gr.Button("🔍 Analyze X-Ray", variant="primary")
                
                with gr.Column(scale=1):
                    detections_output = gr.Markdown(label="Detections")
                    risk_output = gr.Markdown(label="Risk Assessment")
                    recommendations_output = gr.Markdown(label="Recommendations")
            
            analyze_btn.click(
                fn=analyze_xray,
                inputs=[image_input, confidence_slider],
                outputs=[detections_output, risk_output, recommendations_output]
            )
        
        # Tab 2: Treatment Planning
        with gr.Tab("📋 Treatment Planning"):
            with gr.Row():
                with gr.Column(scale=1):
                    condition_dropdown = gr.Dropdown(
                        choices=[
                            "Caries",
                            "Deep Caries",
                            "Periapical Lesion",
                            "Impacted",
                            "Root Canal",
                            "Crown",
                            "Bone Loss"
                        ],
                        label="Select Condition",
                        value="Caries"
                    )
                    severity_dropdown = gr.Dropdown(
                        choices=["mild", "moderate", "severe", "critical"],
                        label="Severity",
                        value="moderate"
                    )
                    treatment_btn = gr.Button("📋 Generate Treatment Plan", variant="primary")
                
                with gr.Column(scale=2):
                    treatment_output = gr.Markdown(label="Treatment Plan")
            
            treatment_btn.click(
                fn=get_treatment_plan,
                inputs=[condition_dropdown, severity_dropdown],
                outputs=treatment_output
            )
        
        # Tab 3: Guidelines Search
        with gr.Tab("📚 Clinical Guidelines"):
            with gr.Row():
                with gr.Column(scale=1):
                    search_input = gr.Textbox(
                        label="Search Guidelines",
                        placeholder="e.g., root canal treatment, caries management..."
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary")
                
                with gr.Column(scale=2):
                    guidelines_output = gr.Markdown(label="Search Results")
            
            search_btn.click(
                fn=search_guidelines,
                inputs=search_input,
                outputs=guidelines_output
            )
        
        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Dental AI Treatment Planner
            
            This application uses advanced AI to assist dental professionals with:
            
            ### 🔍 X-Ray Analysis
            - YOLOv11-based object detection for dental conditions
            - Detects: Caries, Deep Caries, Periapical Lesions, Impacted Teeth, etc.
            - Risk scoring and severity assessment
            
            ### 📋 Treatment Planning
            - Evidence-based recommendations using RAG technology
            - Guidelines from ADA, AAE, FDI World Dental Federation
            - Cost estimates for Dubai/UAE market
            
            ### 🏥 Designed For
            - Dental clinics in Dubai and UAE
            - ROZE BioHealth and BioDental Clinics
            - Dental professionals seeking AI-assisted diagnosis
            
            ---
            
            ### Technology Stack
            - **Vision:** YOLOv11 (Ultralytics)
            - **RAG:** LangChain + ChromaDB
            - **MCP:** Model Context Protocol for AI integration
            - **API:** FastAPI + Gradio
            
            ### Disclaimer
            ⚠️ This tool is designed to assist dental professionals and should not replace 
            professional clinical judgment. All diagnoses should be confirmed by qualified 
            dental practitioners.
            
            ---
            
            **Author:** Kannan @ iBritz.co.uk  
            **Version:** 1.0.0  
            **License:** MIT
            """)
    
    gr.Markdown("""
    ---
    <div class="footer">
    🦷 Dental AI Treatment Planner | Built with YOLOv11, MCP & RAG | © 2025 iBritz.co.uk
    </div>
    """)


def main():
    """Launch the Gradio demo."""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
