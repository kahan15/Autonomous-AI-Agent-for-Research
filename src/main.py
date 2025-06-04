import argparse
import logging
import time
import yaml
from pathlib import Path
import sys

from research_agent.web_research_agent import WebResearchAgent

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('main')

def main():
    """Main entry point for the research agent."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the research agent')
    parser.add_argument('--task', type=str, required=True, help='Research task to perform')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting research task: {args.task}")
    
    try:
        # Initialize and run the agent
        start_time = time.time()
        agent = WebResearchAgent(config_path=args.config)
        
        # Create plan
        plan = agent.plan(args.task)
        
        # Execute plan
        success_count = 0
        total_steps = len(plan)
        
        for step in plan:
            result = agent.execute(step)
            if agent.reflect(result):
                success_count += 1
        
        # Calculate metrics
        duration = time.time() - start_time
        success_rate = (success_count / total_steps) * 100 if total_steps > 0 else 0
        
        # Log results
        logger.info(f"Results saved to: reports")
        logger.info(f"Task completed successfully in {duration:.2f} seconds")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
    except Exception as e:
        logger.error(f"Error during task execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 