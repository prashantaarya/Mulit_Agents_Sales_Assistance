# graph/router.py
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from graph.state import GraphState

def create_router_chain(llm):
    """Create router chain for request routing with user segmentation."""
    router_prompt = PromptTemplate(
        template="""
        You are an expert router that identifies user type and routes requests.
        
        First, identify the user type:
        - SALES REP: Asks about specific prospects, needs detailed analysis, wants personalized outreach
        - DEMAND GEN: Asks about broader campaigns, market segments, lead generation strategies
        
        Then route to appropriate agent:
        - 'prospecting': Finding/searching businesses
        - 'insights': Analysis of specific companies  
        - 'communication': Drafting messages/emails
        - 'end': Goodbye/thank you
        
        Previous context: {conversation_history}
        Current message: {last_message}
        
        Respond with: [USER_TYPE]|[ROUTE]
        Example: SALESREP|insights or DEMANDGEN|prospecting
        """,
        input_variables=["last_message", "conversation_history"],
    )
    
    return router_prompt | llm | StrOutputParser()

def route_requests(state: GraphState, router_chain) -> str:
    """Routes requests with user segmentation."""
    last_message = state['messages'][-1].content
    history = state.get('conversation_history', [])
    
    result = router_chain.invoke({
        "last_message": last_message,
        "conversation_history": str(history[-3:]) if history else "No previous context"
    })
    
    # Parse user type and route
    if '|' in result:
        user_type, route = result.split('|')
        # Store user type in state for agents to use
        if 'user_context' not in state:
            state['user_context'] = {}
        state['user_context']['user_type'] = user_type.strip()
        route = route.strip().lower()
    else:
        route = result.lower()
    
    if 'prospecting' in route:
        return "prospecting"
    elif 'insights' in route:
        return "insights" 
    elif 'communication' in route:
        return "communication"
    else:
        return "end"