# agents/insights.py  
from .base import create_agent, agent_node
from graph.state import GraphState

def create_insights_agent(llm, insights_tools):
    """Create insights agent with user segmentation."""
    return create_agent(
        llm, 
        insights_tools, 
        "You are a business analyst. Adapt analysis based on user type:\n\n"
        "For SALES REPS: Detailed SWOT, specific talking points, objection handling\n"
        "For DEMAND GEN: Market positioning, segment fit, campaign angles\n\n"
        "Use 'get_prospect_details' and format as:\n"
        "**[Business Name] Analysis**\n"
        "**Digital Scores:** [scores]\n"
        "**SWOT Analysis:** [analysis]\n" 
        "**Recommendation:** [tailored to user type]\n\n"
        "Check user_context to customize recommendations."
    )

def insights_node(state: GraphState, agent): 
    return agent_node(state, agent, "InsightsAgent")