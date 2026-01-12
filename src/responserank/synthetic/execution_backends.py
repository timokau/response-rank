import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Dict

import psutil

logger = logging.getLogger(__name__)


class ExecutionBackend(ABC):
    def __init__(self, run_learner_func):
        self.run_learner_func = run_learner_func

    @abstractmethod
    def submit_job(self, *args):
        pass

    @abstractmethod
    def wait_for_jobs(self):
        pass

    @abstractmethod
    def get_job_status(self, job_id):
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up any resources or running processes."""
        pass


class ProcessExecutionBackend(ExecutionBackend):
    def __init__(self, num_processes: int, run_learner_func):
        super().__init__(run_learner_func)

        if num_processes > 0:
            self.max_workers = num_processes
            logger.info(f"Using explicitly specified {self.max_workers} workers")
        else:
            system_cpus = os.cpu_count() or 1
            # Check if running under SLURM and use SLURM_CPUS_PER_TASK if available
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus and slurm_cpus.isdigit():
                slurm_cpus = int(slurm_cpus)
                logger.info(
                    f"Overriding system CPU ({system_cpus}) with SLURM_CPUS_PER_TASK: {slurm_cpus}"
                )
                system_cpus = slurm_cpus
            self.max_workers = system_cpus + num_processes
            logger.info(
                f"Using {self.max_workers} workers ({system_cpus} CPUs + {num_processes})"
            )

        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.futures: Dict[str, Future] = {}
        self.job_id_counter = 0

    def submit_job(self, *args):
        # Get CPU usage before submitting
        cpu_percent = psutil.cpu_percent(interval=0.1)

        future = self.executor.submit(self.run_learner_func, *args)
        job_id = str(self.job_id_counter)
        self.futures[job_id] = future
        self.job_id_counter += 1

        # Calculate queued jobs (those not done)
        jobs_queued = sum(1 for f in self.futures.values() if not f.done())
        logger.info(
            f"Submitted job {self.job_id_counter}. Stats: {jobs_queued}/{self.max_workers} jobs active. CPU usage: {cpu_percent:.1f}%"
        )

        return job_id

    def wait_for_jobs(self):
        logger.info(f"Waiting for {len(self.futures)} jobs to complete")
        for future in self.futures.values():
            future.result()  # This will raise any exceptions that occurred during execution
        self.executor.shutdown(wait=True)

    def get_job_status(self, job_id: str) -> bool:
        future = self.futures.get(job_id)
        if future is None:
            return True  # Assume job is complete if job_id is not found
        return future.done()

    def cleanup(self):
        """Cancel all running futures and terminate any active processes."""
        logger.info("Cleaning up processes...")
        self.executor.shutdown(wait=False, cancel_futures=True)
