# agents/communication.py
from .base import create_agent, agent_node
from graph.state import GraphState

def create_communication_agent(llm, communication_tools):
    """Create communication agent with user segmentation and timing."""
    return create_agent(
        llm, 
        communication_tools, 
        "You are a sales copywriter. Adapt communication based on user type:\n\n"
        "For SALES REPS: Personal outreach emails, cold call scripts, LinkedIn messages\n"
        "For DEMAND GEN: Email campaign templates, nurture sequences, broad messaging\n\n"
        "1. Use 'get_prospect_details' for data\n"
        "2. Create personalized message with:\n"
        "**Subject:** [subject]\n"
        "**Message:** [personalized content]\n"
        "**Optimal Timing:** [when to send - include day, time, and reason]\n"
        "**Follow-up Schedule:** [suggested follow-up timing]\n"
        "**Why This Works:** [explanation]\n\n"
        "Include timing based on:\n"
        "- Industry patterns (B2B: Tue-Thu 10am-2pm, Service: Mon-Wed 8am-11am)\n"
        "- Business urgency (high gaps = immediate, strong presence = nurture timing)\n"
        "- Communication type (email vs call vs LinkedIn)\n\n"
        "Check user_context to adjust tone and approach."
        "You have to add the name as `Prashant Aarya` afetr Best Regards in the message"
    )

def communication_node(state: GraphState, agent): 
    return agent_node(state, agent, "CommunicationAgent")