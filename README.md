# Coal Grade Quality MCP Server

MCP (Model Context Protocol) server that exposes a trained coal ore 
quality classifier as an LLM-accessible tool — allowing Claude and other 
MCP-compatible clients to classify coal images into quality grades .



## What This Does

Upload a coal ore image → get a quality grade back through natural language.

Claude interprets the result, explains its reasoning, and flags any uncertainty 


## Grades

| Grade | Quality |
|-------|---------|
| grade_a | Highest quality — high carbon content |
| grade_b | Medium quality — moderate carbon content |
| grade_c | Lower quality — visible impurities |
| reject | Not suitable for industrial use |



## How It Works

Coal image (base64)
↓
Non-Local Means Denoising
↓
Color Histogram Extraction (96 features)
↓
StandardScaler normalization
↓
Logistic Regression prediction
↓
Grade + Confidence



## MCP PRIMITIVES


| Primitive | Name                      | Purpose                                                | Invocation      |
| --------- | ------------------------- | ------------------------------------------------------ | --------------- |
| Tool      | `grade_coal`              | Runs the ML model on a coal image → grade + confidence | LLM-controlled  |
| Resource  | `coal://info`             | Grade definitions as reference context                 | Host-controlled |
| Prompt    | `classify-with-reasoning` | Structured reasoning template for deep analysis        | User-controlled |



### Connect Claude Desktop

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "coal-grade-server": {
      "command": "path/to/venv/Scripts/python.exe",
      "args": ["path/to/server.py"]
    }
  }
}
```

