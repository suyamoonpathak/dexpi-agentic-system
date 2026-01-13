import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger("Observability")

@dataclass
class TraceLog:
    step: str
    input: str
    output: str
    latency: float

class SystemMonitor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemMonitor, cls).__new__(cls)
            cls._instance.traces = []
            cls._instance.total_tokens = 0
        return cls._instance

    def log_step(self, step_name: str, input_data: str, output_data: str, start_time: float):
        duration = time.time() - start_time
        trace = TraceLog(step_name, input_data, str(output_data)[:200] + "...", duration)
        self.traces.append(trace)
        logger.info(f"[Trace] {step_name} completed in {duration:.2f}s")

    def get_traces(self) -> List[Dict]:
        return [t.__dict__ for t in self.traces]

# Global instance
monitor = SystemMonitor()