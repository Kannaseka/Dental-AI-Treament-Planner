# 🦷 Dental AI Treatment Planner

<div align="center">

![Dental AI Banner](https://img.shields.io/badge/Dental%20AI-Treatment%20Planner-blue?style=for-the-badge&logo=tooth)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

**AI-Powered Dental Treatment Planning with YOLOv11, MCP Server, and RAG**

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[MCP Integration](#-mcp-integration) •
[API Reference](#-api-reference) •
[Contributing](#-contributing)

</div>

---

## 🌟 Overview

Dental AI Treatment Planner is an intelligent system that combines computer vision, retrieval-augmented generation (RAG), and the Model Context Protocol (MCP) to provide comprehensive dental diagnosis and treatment planning.

### Key Capabilities

- **🔍 X-Ray Analysis**: YOLOv11-based detection of dental conditions from panoramic and periapical X-rays
- **📋 Treatment Planning**: Evidence-based recommendations from clinical guidelines (ADA, AAE, FDI)
- **💰 Cost Estimation**: Treatment cost estimates for Dubai/UAE market
- **🤖 MCP Integration**: Seamless integration with Claude, ChatGPT, and other AI assistants
- **🌐 REST API**: FastAPI-based API for web and mobile applications
- **🎨 Interactive UI**: Gradio-powered demo interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dental AI Treatment Planner                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   YOLOv11    │───▶│  MCP Server  │◀───│  RAG Engine  │       │
│  │  Detection   │    │  (FastMCP)   │    │ (LangChain)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Dental X-ray │    │   Claude/    │    │  Clinical    │       │
│  │   Analysis   │    │   ChatGPT    │    │  Guidelines  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  OUTPUT: Diagnosis + Treatment Plan + Cost Estimate              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔬 Dental Condition Detection

Detects multiple dental conditions from X-ray images:

| Condition | Description | Severity Levels |
|-----------|-------------|-----------------|
| Caries | Tooth decay/cavities | Mild, Moderate |
| Deep Caries | Decay approaching pulp | Severe, Critical |
| Periapical Lesion | Infection at root tip | Critical |
| Impacted Tooth | Teeth unable to erupt | Moderate |
| Root Canal | Endodontic treatment | Treatment |
| Crown | Dental crown restoration | Treatment |
| Implant | Dental implant | Treatment |
| Bone Loss | Periodontal bone loss | Severe |

### 📚 Clinical Guidelines Integration

RAG-powered retrieval from:
- American Dental Association (ADA)
- American Association of Endodontists (AAE)
- FDI World Dental Federation
- European Society of Endodontology (ESE)

### 🤖 MCP Tools

| Tool | Description |
|------|-------------|
| `dental_analyze_xray` | Analyze dental X-ray images |
| `dental_get_treatment_plan` | Generate evidence-based treatment plans |
| `dental_search_guidelines` | Search clinical guidelines |
| `dental_get_cost_estimate` | Get treatment cost estimates |
| `dental_complete_diagnosis` | Full diagnosis workflow |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Kannaseka/Dental-AI-Treament-Planner.git
cd dental-ai-treatment-planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or using uv (faster)
uv pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

---

## 🚀 Usage

### 1. Command Line Interface

```bash
# Analyze an X-ray
dental-ai analyze --image path/to/xray.jpg

# Get treatment plan
dental-ai treatment --condition "Deep Caries" --severity "severe"

# Start MCP server
dental-mcp --transport stdio
```

### 2. Python API

```python
from src.vision.dental_analyzer import DentalVisionAnalyzer
from src.rag.dental_rag import DentalGuidelinesRAG

# Initialize analyzer
analyzer = DentalVisionAnalyzer()

# Analyze X-ray
result = analyzer.analyze("dental_xray.jpg")
print(f"Risk Score: {result.risk_score}")
print(f"Detections: {len(result.detections)}")

# Get treatment plan
rag = DentalGuidelinesRAG()
plan = rag.generate_treatment_plan("Caries", "moderate")
print(f"Treatment: {plan.primary_treatment.name}")
print(f"Cost: {plan.primary_treatment.estimated_cost_range}")
```

### 3. REST API

```bash
# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn src.api.main:app --reload
```

**API Endpoints:**

```bash
# Analyze X-ray
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@dental_xray.jpg"

# Get treatment plan
curl -X POST "http://localhost:8000/api/v1/treatment-plan" \
  -H "Content-Type: application/json" \
  -d '{"condition": "Caries", "severity": "moderate"}'

# Get cost estimate
curl -X POST "http://localhost:8000/api/v1/cost-estimate" \
  -H "Content-Type: application/json" \
  -d '{"treatments": ["Root Canal - Molar", "Crown - Zirconia"]}'
```

### 4. Gradio Demo

```bash
# Start interactive demo
python src/api/demo.py

# Opens at http://localhost:7860
```

---

## 🔌 MCP Integration

### With Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "dental-ai": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/dental-ai-treatment-planner"
    }
  }
}
```

### With Claude Code

```bash
# Add MCP server
claude mcp add dental-ai -- python -m src.mcp_server.server
```

### Example MCP Conversation

```
User: Analyze this dental X-ray and create a treatment plan

Claude: [Uses dental_analyze_xray tool]
I've analyzed the X-ray. Here are my findings:

**Risk Score: 65/100**

**Detected Conditions:**
1. Deep Caries (Tooth #36)
   - Severity: Severe
   - Confidence: 78%

**Recommended Treatment:**
Stepwise Caries Excavation
- Estimated Cost: AED 800 - 1,800
- Success Rate: 85%

**Recommendations:**
- Schedule treatment within 1-2 weeks
- Consider root canal if pulp exposure occurs
```

---

## 💰 Cost Reference (Dubai Market)

| Treatment | Cost Range (AED) |
|-----------|------------------|
| Fluoride Treatment | 150 - 400 |
| Composite Filling | 400 - 1,000 |
| Root Canal (Anterior) | 1,500 - 3,500 |
| Root Canal (Molar) | 2,500 - 5,500 |
| Crown (Zirconia) | 2,500 - 5,000 |
| Dental Implant | 5,000 - 12,000 |
| Surgical Extraction | 1,200 - 3,500 |

---

## 📁 Project Structure

```
dental-ai-treatment-planner/
├── src/
│   ├── vision/
│   │   └── dental_analyzer.py    # YOLOv11 X-ray analysis
│   ├── rag/
│   │   └── dental_rag.py         # Clinical guidelines RAG
│   ├── mcp_server/
│   │   └── server.py             # MCP server implementation
│   └── api/
│       ├── main.py               # FastAPI REST API
│       └── demo.py               # Gradio demo interface
├── data/
│   ├── models/                   # YOLO model weights
│   ├── guidelines/               # Clinical guidelines PDFs
│   └── sample_xrays/            # Sample X-ray images
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_vision.py -v
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
export DENTAL_MODEL_PATH=/path/to/dental_yolo.pt
export CONFIDENCE_THRESHOLD=0.25

# API configuration
export API_HOST=0.0.0.0
export API_PORT=8000

# RAG configuration
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🛣️ Roadmap

- [x] YOLOv11 dental condition detection
- [x] RAG-based treatment planning
- [x] MCP server implementation
- [x] FastAPI REST API
- [x] Gradio demo interface
- [ ] DICOM image support
- [ ] Multi-language support (Arabic)
- [ ] Integration with PACS systems
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/GCP)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

# Kannan Sekar

**AI/ML Engineer | Computer Vision & LLM Applications**

Building production AI systems for healthcare and e-commerce. 15+ years of software engineering experience with a focus on deploying scalable AI solutions.

📍 Dubai, UAE | Open to remote opportunities  
🔗 [LinkedIn](https://linkedin.com/in/kannansekar) | [Email](mailto:kannasekarr@gmail.com)

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv11
- [Anthropic](https://anthropic.com/) for MCP and Claude
- [LangChain](https://langchain.com/) for RAG framework
- [Dentex Dataset](https://universe.roboflow.com/dentex) for training data
- ADA, AAE, FDI for clinical guidelines

---

## ⚠️ Disclaimer

This tool is designed to assist dental professionals and should not replace professional clinical judgment. All diagnoses and treatment plans should be confirmed by qualified dental practitioners. Not intended for direct patient use without professional supervision.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the dental AI community

</div>
