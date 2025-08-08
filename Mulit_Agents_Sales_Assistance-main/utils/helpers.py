# utils/helpers.py
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate   
from langchain_core.messages import SystemMessage
import os
from langchain_groq import ChatGroq

# Global memory storage (simple in-memory)
conversation_memory = {
    "history": [],
    "user_context": {}
}

def run_conversation(app, query: str):
    """Helper to run a conversation with memory retention."""
    # Add current query to memory
    conversation_memory["history"].append({
        "query": query,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })
    
    # Keep only last 10 interactions
    if len(conversation_memory["history"]) > 10:
        conversation_memory["history"] = conversation_memory["history"][-10:]
    
    # Create context-aware message
    context_msg = f"Previous queries: {[h['query'] for h in conversation_memory['history'][-3:]]}\nCurrent query: {query}"
    
    initial_state = {
        "messages": [HumanMessage(content=context_msg)],
        "conversation_history": conversation_memory["history"],
        "user_context": conversation_memory["user_context"]
    }
    
    output_data = {}

    if app is not None:
        for event in app.stream(initial_state, {"recursion_limit": 5}):
            for key, value in event.items():
                print(f"--- Output from node: {key} ---")
                print(value)
                print("\n" + "="*40 + "\n")
                output_data["agent_out"] = value
                
                # Store response in memory
                if "agent_out" in output_data:
                    conversation_memory["history"][-1]["response"] = str(value.get("messages", [{}])[-1].content if value.get("messages") else "")
    else:
        raise RuntimeError(" App not compiled. Please fix errors above.")

    return output_data

def get_conversation_context():
    """Get current conversation context"""
    return conversation_memory

def clear_conversation_memory():
    """Clear conversation memory"""
    global conversation_memory
    conversation_memory = {"history": [], "user_context": {}}





# def generate_markdown_output(agent_data: dict) -> str:
#     """
#     Uses an LLM to process agent data and return a structured markdown output.
#     """
#     llm = ChatGroq(model="gemma2-9b-it", temperature=0.3)

#     # --- Prompt Setup ---
#     system_message = SystemMessage(
#         content=(
#             "You are a business analyst assistant helping create concise, structured, and professional "
#             "Markdown reports based on lead qualification data. Format your response in clean Markdown."
#         )
#     )

#     human_prompt_template = (
#         "Here is the lead data from our agent:\n\n"
#         "{data}\n\n"
#         "Please structure this information in a clean and readable Markdown format with the following sections:\n"
#         "1. Prospect Overview\n"
#         "2. Current Digital Snapshot\n"
#         "3. Key Gaps Identified\n"
#         "4. Growth Opportunities\n\n"
#         "Use bullet points, bold labels, and emojis where relevant for visual clarity. Keep the tone professional."
#         "you you do not foud any data, please return a message saying 'No data found.'"
#     )

#     # Format agent data
#     formatted_input = "\n".join([f"- **{key}**: {value}" for key, value in agent_data.items()])
#     human_message = HumanMessage(content=human_prompt_template.format(data=formatted_input))

#     # --- Call LLM ---
#     response = llm([system_message, human_message])

#     return response.content
