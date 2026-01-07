"""
Dental AI CLI
==============
Command-line interface for Dental AI Treatment Planner.
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

app = typer.Typer(
    name="dental-ai",
    help="🦷 AI-Powered Dental Treatment Planning CLI",
    add_completion=False
)
console = Console()


@app.command()
def analyze(
    image: str = typer.Argument(..., help="Path to dental X-ray image"),
    confidence: float = typer.Option(0.25, "--confidence", "-c", help="Confidence threshold"),
    output: str = typer.Option(None, "--output", "-o", help="Output file for results"),
    json_format: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    🔍 Analyze a dental X-ray image.
    
    Detects dental conditions and provides risk assessment.
    """
    from src.vision.dental_analyzer import DentalVisionAnalyzer
    
    if not os.path.exists(image):
        console.print(f"[red]Error:[/red] Image file not found: {image}")
        raise typer.Exit(1)
    
    console.print(Panel("🦷 Analyzing Dental X-Ray...", style="blue"))
    
    analyzer = DentalVisionAnalyzer(confidence_threshold=confidence)
    result = analyzer.analyze(image)
    
    if json_format:
        console.print(result.to_json())
    else:
        # Display results as rich table
        console.print(f"\n[bold]Risk Score:[/bold] {result.risk_score}/100\n")
        
        if result.detections:
            table = Table(title="Detected Conditions")
            table.add_column("Condition", style="cyan")
            table.add_column("Severity", style="yellow")
            table.add_column("Confidence", style="green")
            table.add_column("Tooth #", style="magenta")
            
            for det in result.detections:
                table.add_row(
                    det.condition,
                    det.severity or "N/A",
                    f"{det.confidence*100:.1f}%",
                    str(det.tooth_number) if det.tooth_number else "N/A"
                )
            
            console.print(table)
        else:
            console.print("[green]✅ No significant pathology detected.[/green]")
        
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            console.print(f"  • {rec}")
    
    if output:
        with open(output, 'w') as f:
            f.write(result.to_json())
        console.print(f"\n[green]Results saved to:[/green] {output}")


@app.command()
def treatment(
    condition: str = typer.Argument(..., help="Dental condition"),
    severity: str = typer.Option("moderate", "--severity", "-s", help="Severity level"),
    json_format: bool = typer.Option(False, "--json", "-j", help="Output as JSON")
):
    """
    📋 Generate a treatment plan for a dental condition.
    
    Provides evidence-based recommendations from clinical guidelines.
    """
    from src.rag.dental_rag import DentalGuidelinesRAG
    
    console.print(Panel(f"📋 Generating Treatment Plan for {condition}...", style="blue"))
    
    rag = DentalGuidelinesRAG()
    plan = rag.generate_treatment_plan(condition, severity)
    
    if json_format:
        import json
        console.print(json.dumps(plan.to_dict(), indent=2))
    else:
        console.print(f"\n[bold]Condition:[/bold] {plan.condition}")
        console.print(f"[bold]Severity:[/bold] {plan.severity}")
        console.print(f"[bold]Confidence:[/bold] {plan.confidence_score*100:.0f}%\n")
        
        console.print("[bold cyan]Recommended Treatment:[/bold cyan]")
        console.print(f"  {plan.primary_treatment.name}")
        console.print(f"  [dim]{plan.primary_treatment.description}[/dim]\n")
        
        cost = plan.primary_treatment.estimated_cost_range
        console.print(f"[bold]Estimated Cost:[/bold] {cost['currency']} {cost['min']:,} - {cost['max']:,}")
        console.print(f"[bold]Success Rate:[/bold] {plan.primary_treatment.success_rate}%\n")
        
        console.print("[bold]Prognosis:[/bold]")
        console.print(f"  {plan.prognosis}\n")
        
        console.print("[bold]Follow-up:[/bold]")
        for fu in plan.follow_up_recommendations:
            console.print(f"  • {fu}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query for guidelines"),
    max_results: int = typer.Option(3, "--max", "-m", help="Maximum results")
):
    """
    📚 Search dental clinical guidelines.
    
    Retrieves relevant evidence-based recommendations.
    """
    from src.rag.dental_rag import DentalGuidelinesRAG
    
    console.print(Panel(f"📚 Searching Guidelines: {query}", style="blue"))
    
    rag = DentalGuidelinesRAG()
    results = rag.retrieve_guidelines(query, k=max_results)
    
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}[/bold cyan] (Relevance: {result['relevance_score']*100:.0f}%)")
        console.print(f"[dim]{result['content'][:500]}...[/dim]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload")
):
    """
    🌐 Start the REST API server.
    
    Runs FastAPI server for API access.
    """
    import uvicorn
    
    console.print(Panel(f"🌐 Starting API Server at http://{host}:{port}", style="green"))
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


@app.command()
def demo():
    """
    🎨 Launch the interactive Gradio demo.
    
    Opens web interface for testing.
    """
    console.print(Panel("🎨 Launching Gradio Demo...", style="green"))
    
    from src.api.demo import main
    main()


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport type"),
    port: int = typer.Option(8001, "--port", "-p", help="HTTP port (for streamable_http)")
):
    """
    🤖 Start the MCP server.
    
    Enables integration with Claude, ChatGPT, and other AI assistants.
    """
    console.print(Panel(f"🤖 Starting MCP Server ({transport})...", style="green"))
    
    from src.mcp_server.server import mcp as mcp_server
    
    if transport == "streamable_http":
        mcp_server.run(transport="streamable_http", port=port)
    else:
        mcp_server.run()


@app.command()
def version():
    """
    ℹ️ Show version information.
    """
    console.print(Panel.fit(
        "[bold cyan]🦷 Dental AI Treatment Planner[/bold cyan]\n\n"
        "Version: 1.0.0\n"
        "Author: Kannan @ iBritz.co.uk\n"
        "License: MIT",
        title="About"
    ))


if __name__ == "__main__":
    app()
