
# agents/prospecting.py
from .base import create_agent, agent_node
from graph.state import GraphState

def create_prospecting_agent(llm, prospecting_tools):
    """Create prospecting agent with user segmentation."""
    return create_agent(
        llm, 
        prospecting_tools, 
        "You are a prospecting expert. Adapt your response based on user type:\n\n"
        "For SALES REPS: Focus on specific prospects with detailed contact info and pain points\n"
        "For DEMAND GEN: Focus on broader market segments, volume metrics, and campaign targets\n\n" 
        "Use 'find_prospects_hybrid' tool and return results exactly as provided.\n"
        "After showing the prospects, add a brief summary:\n\n"
        "**Summary:** [Number] prospects found. Key pattern: [main commonality]. "
        "Primary opportunity: [biggest gap/opportunity]. Recommended approach: [strategy].\n\n"
        "Keep summary under 2 sentences and focus on actionable insights."
    )

def prospecting_node(state: GraphState, agent): 
    return agent_node(state, agent, "ProspectingAgent")