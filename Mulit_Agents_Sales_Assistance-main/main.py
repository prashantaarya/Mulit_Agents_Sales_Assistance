"""
Main entry point - orchestrates the entire system
"""
import os
from functools import partial
from langchain_groq import ChatGroq
from langchain.tools import Tool
from typing import Optional


# Import all modules
from config.settings import GROQ_API_KEY, MODEL_NAME, TEMPERATURE, DATA_FILE_PATH
from data.processor import process_data
from tools.toolbox import ToolBox
from tools.hybrid_search import HybridSearchToolBox
from agents.prospecting import create_prospecting_agent, prospecting_node
from agents.insights import create_insights_agent, insights_node
from agents.communication import create_communication_agent, communication_node
from graph.router import create_router_chain, route_requests
from graph.workflow import create_workflow
from utils.helpers import run_conversation


class SalesSystem:
    """Main Sales System orchestrator."""
    
    def __init__(self):
        self.llm = None
        self.app = None
        self.enhanced_toolbox = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the complete sales system."""
        print(" Initializing Sales System...")
        
        
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.llm = ChatGroq(temperature=TEMPERATURE, model_name=MODEL_NAME)
        print("LLM initialized successfully.")
        
        
        processed_df = process_data(DATA_FILE_PATH)
        
        if processed_df.empty:
            print("System initialization failed due to data processing error.")
            return
        
        
        try:
            
            toolbox = ToolBox(dataframe=processed_df)
            
           
            self.enhanced_toolbox = HybridSearchToolBox(df=processed_df, llm=self.llm)
            print(" Enhanced tools configured successfully.")
        except Exception as e:
            print(f"Error initializing toolboxes: {e}")
            return
        
#Create tools
        from langchain.tools import tool
        
        @tool
        def find_prospects_hybrid(query: str) -> list[dict]:
            """
            Intelligent prospect finder. Handles complex queries like 'find businesses with low local presence but high SEM spend' or 'Computer Contractors missing Google Places listings'.
            
            Args:
                query: Natural language query describing the prospects you're looking for
                
            Returns:
                List of prospect dictionaries matching the criteria
            """
            return self.enhanced_toolbox.find_prospects_hybrid(query)
        
        @tool 
        def get_prospect_details(prospect_name: str) -> Optional[dict]:
            """
            Get detailed digital marketing analysis and opportunity assessment for a specific prospect.
            
            Args:
                prospect_name: The exact business name of the prospect
                
            Returns:
                Detailed prospect analysis dictionary
            """
            return self.enhanced_toolbox.get_prospect_details(prospect_name)
        
        prospecting_tools = [find_prospects_hybrid]
        insights_tools = [get_prospect_details]
        communication_tools = [get_prospect_details] 
        
        #Create agents
        prospecting_agent = create_prospecting_agent(self.llm, prospecting_tools)
        insights_agent = create_insights_agent(self.llm, insights_tools)
        communication_agent = create_communication_agent(self.llm, communication_tools) 
        
        #Create router
        router_chain = create_router_chain(self.llm)
        
        #Create node functions with partial application
        def prospecting_node_func(state):
            return prospecting_node(state, prospecting_agent)
        
        def insights_node_func(state):
            return insights_node(state, insights_agent)
        
        def communication_node_func(state):
            return communication_node(state, communication_agent)
        
        def route_requests_func(state):
            return route_requests(state, router_chain)
        
        #Build workflow
        self.app = create_workflow(
            prospecting_node_func, 
            insights_node_func, 
            communication_node_func, 
            route_requests_func
        )
        
        print("Graph compiled successfully! The system is ready.")
    
    def run_query(self, query: str):
        """Run a query through the sales system."""
        if self.app is None:
            print(" System not initialized properly.")
            return None
        
        return run_conversation(self.app, query)
    
    def get_system_status(self):
        """Get the current system status."""
        return {
            "LLM": "Ready" if self.llm else "Not initialized",
            "App": "Ready" if self.app else "Not initialized", 
            "Enhanced Toolbox": "Ready" if self.enhanced_toolbox else "Not initialized"
        }

def print_welcome_message():
    """Print welcome message and instructions."""
    print("=" * 80)
    print(" WELCOME TO THE AI SALES SYSTEM ")
    print("=" * 80)
    print("\n What can I help you with today?")
    print("\n EXAMPLE QUERIES:")
    print("   • 'Find computer contractors in Texas'")
    print("   • 'Show me businesses with low local presence'") 
    print("   • 'Get details for [Business Name]'")
    print("   • 'Draft an email for [Business Name]'")
    print("   • 'Find IT companies missing Google Places listings'")
    print("\n COMMANDS:")
    print("   • Type 'help' for more examples")
    print("   • Type 'status' to check system health")
    print("   • Type 'quit' or 'exit' to end session")
    print("\n" + "=" * 80)

def print_help():
    """Print detailed help information."""
    print("\n DETAILED HELP & EXAMPLES")
    print("=" * 50)
    print("\n PROSPECTING QUERIES:")
    print("   • 'Find computer contractors'")
    print("   • 'Show me businesses in California'") 
    print("   • 'Find companies with high SEM spend'")
    print("   • 'Look for businesses with weak local presence'")
    print("   • 'Find IT services companies'")
    
    print("\n INSIGHTS & ANALYSIS:")
    print("   • 'Analyze [Business Name]'")
    print("   • 'Get details for [Business Name]'")
    print("   • 'Show SWOT analysis for [Business Name]'")
    
    print("\n COMMUNICATION:")
    print("   • 'Draft an email for [Business Name]'")
    print("   • 'Write a sales message for [Business Name]'")
    print("   • 'Create outreach content for [Business Name]'")
    
    print("\n TIPS:")
    print("   • Be specific about location, industry, or digital marketing needs")
    print("   • Combine criteria: 'Find Texas contractors with no Google Places'")
    print("   • Ask for analysis after finding prospects")
    print("=" * 50)

def interactive_mode(sales_system):
    """Run the system in interactive mode."""
    print_welcome_message()
    
    while True:
        try:
            # Get user input
            print("\n" + "─" * 80)
            user_input = input("\n Enter your query: ").strip()
            
            # Handle empty input
            if not user_input:
                print("  Please enter a query. Type 'help' for examples.")
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Thank you for using the AI Sales System! Goodbye!")
                break
            
            elif user_input.lower() in ['help', 'h']:
                print_help()
                continue
            
            elif user_input.lower() in ['status', 'health']:
                status = sales_system.get_system_status()
                print("\n SYSTEM STATUS:")
                for component, status_msg in status.items():
                    print(f"   {component}: {status_msg}")
                continue
            
            # Process the query
            print(f"\n Processing: '{user_input}'")
            print(" Please wait...")
            
            try:
                result = sales_system.run_query(user_input)
                
                if result and 'agent_out' in result:
                    response = result['agent_out']['messages'][0].content
                    print("\nRESULT:")
                    print("─" * 40)
                    print(response)
                    print("─" * 40)
                else:
                    print("\n No result returned. Please try a different query.")
                    
            except Exception as e:
                print(f"\n Error processing query: {str(e)}")
                print(" Try rephrasing your query or type 'help' for examples.")
        
        except KeyboardInterrupt:
            print("\n\n Session interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n Unexpected error: {str(e)}")
            print("Please try again or restart the system.")

def run_test_mode(sales_system):
    """Run predefined test queries."""
    print("\nRUNNING TEST MODE...")
    
    test_queries = [
        "Find computer contractors in Texas",
        "Show me businesses with low local presence",  
        "Draft a personalized email for a tech business"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        try:
            result = sales_system.run_query(query)
            if result and 'agent_out' in result:
                print("Result:", result['agent_out']['messages'][0].content[:200] + "...")
            else:
                print("No result returned")
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main function to run the sales system."""
    # Initialize the system
    print("🔧 Initializing AI Sales System...")
    sales_system = SalesSystem()
    
    # Check system status
    status = sales_system.get_system_status()
    
    if not all("" in s for s in status.values()):
        print("\nSYSTEM INITIALIZATION FAILED")
        print("System Status:")
        for component, status_msg in status.items():
            print(f"   {component}: {status_msg}")
        print("\n💡 Please check your configuration and data files.")
        return None
    
    print(" System ready!")
    
    # Ask user for mode
    print("\n  SELECT MODE:")
    print("   1. Interactive Mode (recommended)")
    print("   2. Test Mode (run sample queries)")
    print("   3. Single Query Mode")
    
    while True:
        try:
            mode = input("\nEnter your choice (1/2/3): ").strip()
            
            if mode == '1':
                interactive_mode(sales_system)
                break
            elif mode == '2':
                run_test_mode(sales_system)
                break
            elif mode == '3':
                query = input("\nEnter your query: ").strip()
                if query:
                    print(f"\n Processing: '{query}'")
                    result = sales_system.run_query(query)
                    if result and 'agent_out' in result:
                        print("\n RESULT:")
                        print("─" * 40)
                        print(result['agent_out']['messages'][0].content)
                        print("─" * 40)
                break
            else:
                print(" Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
    
    return sales_system

if __name__ == "__main__":
    system = main()