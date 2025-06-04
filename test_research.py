"""Test script for the research agent."""
import os
from dotenv import load_dotenv
from src.research_agent.web_research_agent import WebResearchAgent

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    if not os.getenv("SEARCH_API_KEY"):
        print("Error: SEARCH_API_KEY not found in environment variables")
        print("Please set your Serper API key in the .env file")
        return
    
    # Initialize the research agent
    agent = WebResearchAgent()
    
    # Test research task
    research_task = "What are the latest developments in large language models in 2024?"
    print(f"\nResearching: {research_task}\n")
    
    # Execute the research task
    result = agent.execute_task(research_task)
    
    # Print results
    if result.success:
        print("\nResearch completed successfully!")
        print("\nMetrics:")
        for key, value in result.metrics.items():
            print(f"{key}: {value}")
            
        # Get the log file path
        log_dir = os.path.join("data", "logs")
        log_files = [f for f in os.listdir(log_dir) if f.startswith("research_agent_")]
        if log_files:
            latest_log = max(log_files)
            print(f"\nLog file saved to: {os.path.join(log_dir, latest_log)}")
            
            # Optionally display log contents
            log_contents = agent.get_latest_log()
            if log_contents:
                print("\nLog file contents:")
                print("-" * 80)
                print(log_contents)
                print("-" * 80)
    else:
        print(f"\nResearch failed: {result.error}")

if __name__ == "__main__":
    main() 