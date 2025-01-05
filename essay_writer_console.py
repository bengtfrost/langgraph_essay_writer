from dotenv import load_dotenv
import os
import requests
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from tavily import TavilyClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ = load_dotenv()

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Set up Tavily client
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Google Custom Search API setup
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

def google_search(query, max_results=2):
    """Perform a search using Google Custom Search JSON API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": max_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Google Search failed for query '{query}': {e}")
        return None

def search_with_fallback(query, max_results=2):
    """Search using Tavily, fallback to Google if Tavily fails."""
    logger.info(f"Searching for: {query}")

    # Try Tavily first
    try:
        logger.info("Trying Tavily...")
        search_response = tavily.search(query=query, max_results=max_results)
        results = search_response.get('results', [])
        if results:
            logger.info("Tavily search successful.")
            return [r['content'] for r in results]
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")

    # Fallback to Google if Tavily fails
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        logger.info("Falling back to Google Search...")
        search_response = google_search(query, max_results=max_results)
        if search_response:
            results = search_response.get('items', [])
            if results:
                logger.info("Google Search successful.")
                return [r.get('snippet', '') for r in results]

    logger.warning("Both Tavily and Google Search failed.")
    return []

# Define API endpoint for Mistral model
MISTRAL_API_URL = "http://localhost:4000/v1/completions"

def call_mistral_api(messages):
    """Call the Mistral API."""
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    try:
        response = requests.post(MISTRAL_API_URL, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        logger.error(f"Response: {e.response.text}")
        raise

PLAN_PROMPT = """You are an expert writer tasked with writing a high-level outline of an essay.
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user's request and the initial outline.
If the user provides critique, respond with a revised version of your previous attempts.
Utilize all the information below as needed:

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can
be used when writing the following essay. Generate a list of search queries that will gather
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can
be used when making any requested revisions (as outlined below).
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

def plan_node(state: AgentState):
    """Generate an essay plan."""
    logger.info("Running plan_node...")
    messages = [
        {"role": "system", "content": PLAN_PROMPT},
        {"role": "user", "content": state['task']}
    ]
    response = call_mistral_api(messages)
    return {"plan": response['choices'][0]['text']}

def research_plan_node(state: AgentState):
    """Generate research queries and perform searches."""
    logger.info("Running research_plan_node...")
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_PROMPT},
        {"role": "user", "content": state['task']}
    ]
    response = call_mistral_api(messages)

    queries = [q.strip() for q in response['choices'][0]['text'].split('\n') if q.strip()]
    logger.info(f"Generated research queries: {queries}")

    content = state['content'] or []
    for q in queries[:3]:  # Limit to 3 queries
        search_results = search_with_fallback(q, max_results=2)
        content.extend(search_results)

    return {"content": content}

def generation_node(state: AgentState):
    """Generate the essay draft."""
    logger.info("Running generation_node...")
    content = "\n\n".join(state['content'] or [])
    user_message = {
        "role": "user",
        "content": f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    }

    messages = [
        {"role": "system", "content": WRITER_PROMPT.format(content=content)},
        user_message
    ]

    response = call_mistral_api(messages)

    return {
        "draft": response['choices'][0]['text'],
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    """Generate critique for the essay draft."""
    logger.info("Running reflection_node...")
    messages = [
        {"role": "system", "content": REFLECTION_PROMPT},
        {"role": "user", "content": state['draft']}
    ]

    response = call_mistral_api(messages)

    return {"critique": response['choices'][0]['text']}

def research_critique_node(state: AgentState):
    """Generate research queries based on critique and perform searches."""
    logger.info("Running research_critique_node...")
    messages = [
        {"role": "system", "content": RESEARCH_CRITIQUE_PROMPT},
        {"role": "user", "content": state['critique']}
    ]

    response = call_mistral_api(messages)

    queries = [q.strip() for q in response['choices'][0]['text'].split('\n') if q.strip()]
    logger.info(f"Generated research queries: {queries}")

    content = state['content'] or []

    for q in queries[:3]:  # Limit to 3 queries
        search_results = search_with_fallback(q, max_results=2)
        content.extend(search_results)

    return {"content": content}

def should_continue(state):
    """Determine if the graph should continue or end."""
    if state["revision_number"] > state["max_revisions"]:
        logger.info("Max revisions reached. Ending graph.")
        return END
    logger.info("Continuing to next revision.")
    return "reflect"

builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile()

# Main function to run the script
def main():
    # Ask for the essay topic in the console
    essay_topic = input("Enter the essay topic: ").strip()
    if not essay_topic:
        print("Error: Essay topic cannot be empty.")
        return

    max_revisions = 3
    logger.info("Starting essay generation process...")
    result = graph.invoke({
        "task": essay_topic,
        "max_revisions": max_revisions,
        "revision_number": 0,
        "content": [],
        "plan": "",
        "draft": "",
        "critique": ""
    })

    logger.info("Essay generation complete.")
    print("\nFinal Essay:")
    print(result["draft"])

# Run the script
if __name__ == "__main__":
    main()
