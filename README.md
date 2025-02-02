# Essay Writer with Mistral AI and Tavily/Google Search

This project automates essay writing using **Mistral AI** for content generation
and **Tavily API** (with fallback to **Google Custom Search**) for research.
The workflow is orchestrated by **LangGraph**, ensuring a seamless process from
planning to final draft.

---

## Features

- **Plan Node**: Generates a detailed essay outline.
- **Research Node**: Gathers relevant content using Tavily (or Google as a fallback).
- **Generate Node**: Writes the essay draft.
- **Reflect Node**: Provides critique and recommendations.
- **Revise Loop**: Repeats drafting and refining until the essay is polished.

---

## Requirements

- Python 3.8+
- Required packages: See `requirements.txt`.

---

## Setup

### 1. Clone the Repository

Clone the repository to your local machine:

git clone https://github.com/bengtfrost/langgraph_essay_writer.git

cd essay-writer

### 2. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

### 3. Set Up Environment Variables

Rename `.env.example` to `.env`:

cp .env.example .env

Open `.env` in a text editor and replace the placeholders with your API keys:

TAVILY_API_KEY=your_tavily_api_key

MISTRAL_KEY=your_mistral_api_key

GOOGLE_GEMINI_KEY=your_google_gemini_key  # Optional, for Google Gemini fallback

### 4. Export Environment Variables

For local testing, you can export the environment variables:

export TAVILY_API_KEY='your_tavily_api_key'

export MISTRAL_KEY='your_mistral_api_key'

### 5. Start the LiteLLM Server for Mistral

Run the LiteLLM server to connect to the Mistral API:

litellm --model mistral/mistral-large-2407

### 6. Run the Script

Start the essay writer script:

python essay_writer_console.py

---

## Usage

When prompted, enter your essay topic. The script will:

- Generate an outline.
- Perform research using Tavily (or Google as a fallback).
- Write the essay draft.
- Provide critique and refine the draft.

The final essay will be displayed in the console.

---

## Dependencies

- **Mistral AI**: For generating essay content (via LiteLLM server).
- **Tavily API**: For research (fallback to Google Custom Search if needed).
- **LangGraph**: For workflow orchestration.

---

## File Structure

essay-writer/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── essay_writer_console.py # Main script
├── .env.example            # Template for environment variables
└── LICENSE                 # License file

---

## Environment Variables

The `.env` file should contain the following variables:

```plaintext
TAVILY_API_KEY=your_tavily_api_key
MISTRAL_KEY=your_mistral_api_key
GOOGLE_GEMINI_KEY=your_google_gemini_key  # Optional
```

- **TAVILY_API_KEY**: Your API key for Tavily. Sign up at Tavily.
- **MISTRAL_KEY**: Your API key for Mistral AI. Sign up at Mistral AI.
- **GOOGLE_GEMINI_KEY**: Your API key for Google Gemini (optional). Sign up at Google Cloud.











