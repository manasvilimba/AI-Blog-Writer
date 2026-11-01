
# AI Blog Writer

An AI-powered technical blog generation application built using Python, LangGraph, LangChain, and Streamlit.

## Features

- AI-powered blog planning and generation
- Intelligent routing to determine whether web research is required
- Web research integration using Tavily Search
- Structured blog outlines with sections, goals, and bullet points
- Parallel section generation using LangGraph
- Automatic merging of generated sections
- AI-generated technical diagrams and images
- Markdown output for generated blogs

## Tech Stack

- Python
- LangGraph
- LangChain
- Groq API
- Tavily Search API
- Google Gemini API
- Streamlit
- Pydantic

## Project Structure

```text
AIBLOGWRT/
│
├── AIBLOG.py          # Backend and LangGraph workflow
├── AIBLOG_FD.py       # Streamlit frontend
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from Git
└── README.md          # Project documentation
```

## Setup and Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd AIBLOGWRT
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Create a `.env` file in the project directory and add your API keys:

```env
NEW_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

Never share your actual API keys publicly.

## Run the Application

```bash
streamlit run AIBLOG_FD.py
```

## Workflow

1. User enters a blog topic.
2. Router determines whether research is required.
3. Tavily retrieves relevant web sources when needed.
4. Orchestrator creates a structured blog plan.
5. Worker nodes generate individual blog sections in parallel.
6. Reducer combines the sections.
7. Image planner identifies useful diagrams.
8. Final blog is generated in Markdown format.

## Author

Manasvi

## License

This project is available for educational and learning purposes.
