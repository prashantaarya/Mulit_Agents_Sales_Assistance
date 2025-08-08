# agents/base.py
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from graph.state import GraphState

def create_agent(llm, tools: list, system_prompt: str):
    """Helper function to create a new agent."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        MessagesPlaceholder(variable_name="messages"), 
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        return_intermediate_steps=True,
        max_iterations=10,  # Limit iterations to prevent loops
        max_execution_time=30,  # 30 second timeout
        early_stopping_method="generate"  # Stop after first valid response
    )
    return executor

def agent_node(state: GraphState, agent: AgentExecutor, name: str):
    """Helper function to invoke an agent and update state."""
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=str(result["output"]), name=name)]}