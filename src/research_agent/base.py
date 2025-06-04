from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import yaml
from pathlib import Path
import time

@dataclass
class PlanStep:
    """A step in the research plan."""
    objective: str
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class ExecutionResult:
    """Result of an execution step."""
    success: bool
    output: Any = None
    error: str = None
    metrics: Dict[str, Any] = None

class ResearchAgent(ABC):
    """Base class for the autonomous research agent implementing the Plan-Execute-Reflect workflow."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the research agent."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.data_dir = self._setup_data_dir()
        self.current_plan: List[PlanStep] = []
        self.execution_history: List[ExecutionResult] = []
        self.reflection_notes: List[str] = []
        self.current_task: str = ""

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {str(e)}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("Research Agent")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add formatter to ch
        ch.setFormatter(formatter)
        
        # Add ch to logger
        logger.addHandler(ch)
        
        return logger

    def _setup_data_dir(self) -> Path:
        """Set up data directory structure."""
        try:
            data_config = self.config.get('data', {})
            base_dir = Path(data_config.get('base_dir', 'data'))
            
            # Create required directories
            base_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / data_config.get('cache_dir', 'cache')).mkdir(exist_ok=True)
            (base_dir / data_config.get('report_dir', 'reports')).mkdir(exist_ok=True)
            
            return base_dir
            
        except Exception as e:
            self.logger.error(f"Failed to set up data directory: {str(e)}")
            return Path('data')  # Fallback to basic structure

    @abstractmethod
    def plan(self, task: str) -> List[PlanStep]:
        """Create a research plan."""
        pass

    @abstractmethod
    def execute(self, step: PlanStep) -> ExecutionResult:
        """Execute a research step."""
        pass

    @abstractmethod
    def reflect(self, execution_result: ExecutionResult) -> bool:
        """Reflect on execution results."""
        pass

    def run(self, task: str) -> Dict[str, Any]:
        """Execute the full Plan-Execute-Reflect workflow for a given task."""
        self.logger.info(f"Starting new task: {task}")
        self.current_task = task
        start_time = time.time()
        
        try:
            # Plan phase
            self.logger.info("Planning phase started")
            self.current_plan = self.plan(task)
            
            # Execute phase
            self.logger.info("Execution phase started")
            for step in self.current_plan:
                self.logger.info(f"Executing step: {step.objective}")
                result = self.execute(step)
                self.execution_history.append(result)
                
                if not result.success:
                    self.logger.error(f"Step failed: {result.error}")
                    continue  # Continue with next step instead of breaking
                
                # Reflect phase
                self.logger.info(f"Reflecting on step: {step.objective}")
                self.reflect(result)
            
            # Generate final report
            duration = time.time() - start_time
            report = self._generate_final_report(task, duration)
            
            self.logger.info(f"Task completed in {duration:.2f} seconds")
            return report
            
        except Exception as e:
            self.logger.error(f"Error during task execution: {str(e)}", exc_info=True)
            return {
                "task": task,
                "duration_seconds": time.time() - start_time,
                "iterations": len(self.execution_history),
                "success_rate": 0.0,
                "error": str(e),
                "final_state": "failed"
            }

    def _is_task_complete(self, task: str) -> bool:
        """Check if the task has been completed successfully."""
        if not self.execution_history:
            return False
            
        # Check the last few execution results
        recent_results = self.execution_history[-3:]
        return all(result.success for result in recent_results)

    def _generate_final_report(self, task: str, duration: float) -> Dict[str, Any]:
        """Generate a final report of the agent's work."""
        return {
            "task": task,
            "duration_seconds": duration,
            "iterations": len(self.execution_history),
            "success_rate": sum(1 for r in self.execution_history if r.success) / len(self.execution_history),
            "reflection_notes": self.reflection_notes,
            "final_state": "completed" if self._is_task_complete(task) else "incomplete"
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if exc_type is not None:
            self.logger.error(f"Error during agent execution: {str(exc_val)}")
        self.logger.info("Agent shutting down") 