import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import submitit

from dinov2.utils.cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_user_checkpoint_path,
)


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    slurm_partition = get_slurm_partition()
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--ngpus",
        "--gpus",
        default=2,
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--nodes",
        "--nnodes",
        default=1,
        type=int,
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--timeout",
        default=4319,  
        type=int,
        help="Duration of the job in minutes",
    )
    parser.add_argument(
        "-p",
        "--partition",
        default="a100_short", # default="a100_short,gpu4_short,gpu4_medium,a100_dev,a100_long,gpu8_short,gpu8_medium"
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--cpus-per-task",
        default=16,
        type=int,
        help="Number of CPUs per task",
    )
    parser.add_argument(
        "--mem",
        default=100,
        type=int,
        help="Memory per node in GB",
    )
    return parser


def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = Path("/gpfs/data/shenlab/wz1492/HCC/dinov2/experiments")
    path.mkdir(exist_ok=True)
    return path


def submit_jobs(task_class, args, name: str):
    if not args.output_dir:
        args.output_dir = str(get_shared_folder())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if hasattr(args, 'use_volta32') and args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if hasattr(args, 'comment') and args.comment:
        kwargs["slurm_comment"] = args.comment
    if hasattr(args, 'exclude') and args.exclude:
        kwargs["slurm_exclude"] = args.exclude

    # Add new SLURM parameters
    kwargs["slurm_cpus_per_task"] = args.cpus_per_task
    kwargs["slurm_mem"] = f"{args.mem}GB"

    executor_params = get_slurm_executor_parameters(
        nodes=args.nodes,
        num_gpus_per_node=args.ngpus,
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        **kwargs,
    )
    executor.update_parameters(name=name, **executor_params)

    task = task_class(args)
    job = executor.submit(task)

    logger.info(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(args.output_dir).replace("%J", str(job.job_id))
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")
