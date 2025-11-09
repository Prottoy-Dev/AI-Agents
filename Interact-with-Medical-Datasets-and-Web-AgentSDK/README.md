# 🏥 Multi-Tool AI Agent for Medical Datasets and Web

A sophisticated AI agent system that can interact with medical datasets and search the web for general medical information. The system intelligently routes questions to specialized agents based on the query type.

## 🌟 Features

- **🤖 Multi-Agent Architecture**: Intelligent routing between database and web search agents
- **📊 Medical Database Analysis**: Query and analyze three medical datasets:
  - Heart Disease Dataset (1,025 patients)
  - Breast Cancer Dataset
  - Diabetes Dataset
- **🔍 Risk Assessment**: Analyze patient characteristics for disease risk evaluation
- **🌐 Web Search Integration**: Access general medical knowledge via Tavily search
- **📈 Statistical Analysis**: Generate insights from patient data with natural language queries

## 🏗️ Architecture

### Agent Hierarchy
```
Main Medical Assistant
├── 📊 Database Agent (Medical Database Specialist)
│   ├── Heart Disease DB Query Tool
│   ├── Cancer DB Query Tool
│   └── Diabetes DB Query Tool
└── 🌐 Web Agent (Medical Web Knowledge Assistant)
    └── Tavily Medical Search Tool
```

### Routing Logic
- **Database Agent** handles:
  - Dataset-specific questions
  - Statistical analysis requests
  - Risk assessments based on patient characteristics
  - Numerical data analysis
  - Data-driven predictions

- **Web Agent** handles:
  - General medical knowledge
  - Treatment options and procedures
  - Latest medical research
  - Health advice not requiring data analysis

## 🚀 Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "F:/Courses/Ostad/Assignments/Multi-Tool AI Agent to Interact with Medical Datasets and Web/medical_ai_agent"
   ```

2. **Create and activate virtual environment:**
   ```powershell
   # Create venv
   python -m venv venv
   
   # Activate (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration:**
   Create a `.env` file in the project root:
   ```env
   # GitHub Models API (Free tier: 150 requests/day)
   OPENAI_API_BASE=https://models.github.ai/v1
   OPENAI_API_KEY=your_github_token_here
   MODEL_NAME=gpt-4o-mini
   
   # Tavily Search API (Optional - for web search)
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## 💾 Database Setup

The project includes pre-configured SQLite databases:
- `databases/heart_disease.db` - Heart disease patient data
- `databases/cancer.db` - Breast cancer patient data  
- `databases/diabetes.db` - Diabetes patient data

Databases are automatically loaded from CSV files in the `data/` directory.

## 🎯 Usage

### Run the Agent
```bash
python agent.py
```

### Example Interactions

#### 1. General Medical Knowledge (→ Web Agent)
```
Q: "What are the common treatments for diabetes?"
A: 🌐 WEB AGENT: Searching for general medical information...
   [Provides comprehensive treatment options]
```

#### 2. Dataset Queries (→ Database Agent)
```
Q: "How many patients are in the heart disease dataset?"
A: 📊 ROUTING TO DATABASE AGENT: Dataset-specific question
   🔍 DATABASE AGENT: Now analyzing your question...
   There are 1,025 patients in the heart disease dataset.
```

#### 3. Risk Assessment (→ Database Agent)
```
Q: "My age is 58, I am male, what is my risk of heart disease?"
A: 📊 ROUTING TO DATABASE AGENT: Risk assessment based on patient characteristics
   🔍 DATABASE AGENT: Now analyzing your question...
   [Provides statistical analysis based on similar patients]
```

## 🔧 Configuration

### API Providers

**GitHub Models (Default - Free)**
- Rate limit: 150 requests per 24 hours
- Models: gpt-4o-mini, gpt-4o, etc.
- Good for development and testing

**Alternative APIs:**
```env
# OpenAI
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_key
MODEL_NAME=gpt-4o-mini

# Azure OpenAI
OPENAI_API_BASE=https://your-resource.openai.azure.com
OPENAI_API_KEY=your_azure_key
MODEL_NAME=your-deployment-name
```

### Model Selection
Supported models:
- `gpt-4o-mini` (recommended, cost-effective)
- `gpt-4o` (more capable, higher cost)
- `gpt-4` (legacy)

## 📁 Project Structure

```
medical_ai_agent/
├── agent.py                 # Main agent implementation
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── data/                   # CSV datasets
│   ├── diabetes.csv
│   ├── heart disease dataset.csv
│   └── The Cancer data.csv
├── databases/              # SQLite databases
│   ├── csv_to_sqlite.py   # Database conversion script
│   ├── diabetes.db
│   ├── heart_disease.db
│   └── cancer.db
└── venv/                   # Virtual environment
```

## 🛠️ Development

### Adding New Datasets
1. Place CSV file in `data/` directory
2. Update `csv_to_sqlite.py` to include new dataset
3. Create new query function in `agent.py`
4. Add to appropriate agent's tools

### Customizing Agents
Modify agent instructions in `agent.py` to change:
- Routing logic
- Response format
- Analysis approach

## 🚨 Troubleshooting

### Rate Limit Errors
```
Error code: 429 - Rate limit exceeded
```
**Solution**: Wait for rate limit reset (24 hours for GitHub Models) or switch API providers.

### Environment Issues
```
Activation script not found
```
**Solution**: Ensure virtual environment is created and use full path:
```powershell
"F:\...\medical_ai_agent\venv\Scripts\Activate.ps1"
```

### Database Connection Issues
**Solution**: Verify database files exist in `databases/` directory and run `csv_to_sqlite.py` if needed.

## 📊 Dataset Information

- **Heart Disease**: 1,025 patients with 14 attributes
- **Breast Cancer**: Diagnostic features for breast cancer classification  
- **Diabetes**: Patient data for diabetes risk assessment

*Note: All datasets are for educational purposes and contain de-identified/synthetic data.*

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Educational Use**: This project is part of Ostad coursework and is available for educational and learning purposes.

## 🙋‍♀️ Support

For questions or issues:
1. Check the troubleshooting section
2. Review agent logs for routing decisions
3. Verify API keys and rate limits

---

**Built with** 🤖 Agents Framework, 🦜 LangChain, and 💖 for medical AI