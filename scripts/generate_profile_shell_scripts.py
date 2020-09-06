#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

if __name__ == "__main__":

    # program_str = "python -u -m habitat_baselines.run --exp-config habitat_baselines/config/pointnav/ddppo_pointnav.yaml --run-type train"
    program_str = "python -u -m habitat_baselines.run --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_example.yaml --run-type train"

    # Path to Nsight Systems nsys command-line tool
    # nsys_path = "/private/home/eundersander/nsight-systems-2020.3.1/bin/nsys"
    nsys_path = "/opt/nvidia/nsight-systems/2020.3.2/bin/nsys"

    # You can either capture a step range or a time range. Capturing a step range is generally a better workflow, but it requires integrating profiling_utils.configure into your train program.
    do_capture_step_range = True

    if do_capture_step_range:
        # "Step" here refers to however you defined a train step in your train program. See habitat-sim profiling_utils.configure.
        # Prefer capturing a range of steps that are representative of your entire train job, in terms of the time spent in various parts of your program. Early train steps may suffer from poor agent behavior, too-short episodes, etc. If necessary, capture and inspect a very long-duration profile to determine when your training "settles".
        # DDPPO PointNav empirical test from Aug 2020: settled around 150-190, after which it was stable out to ~1400 steps
        capture_start_step = 3  # 190

        # If you're focusing on optimizing the train loop body (work that happens consistently every update), you don't need a large number here. However, beware overlooking infrequent events like env resets, scene loads, checkpointing, and eval.
        # Beware large profile storage requirement
        # DDPPO PointNav empirical test from Aug 2020:
        #   qdrep: 3.3 MB per 100 steps
        #   sqlite: 12 MB per 100 steps
        #   these figures are for a single task (see also capture_all_tasks below)
        num_steps_to_capture = 6  # 100

    else:
        nsys_capture_delay_seconds = 10
        nsys_capture_duration_seconds = 20

    do_slurm = False

    if do_slurm:

        # path can be absolute or relative to the working directory (where you run the profiling shell script, which is probably the habitat-lab root)
        # todo: create if necessary
        profile_output_folder = "profiles"

        # You must use ${SLURM_NODEID} and ${SLURM_LOCALID} if using capture_all_tasks. Use $1 for slurm job id (optional).
        profile_output_filename_base = (
            "profile_job$1_node${SLURM_NODEID}_local${SLURM_LOCALID}"
        )
    else:
        profile_output_folder = "/home/eundersander/projects/profiling/temp"
        profile_output_filename_base = "my_profile"

    if do_slurm:
        # A job duration to provide to slurm
        # Provide a reasonable upper bound here. It's not important to provide a tight bound. A too-short duration will cause your slurm job to terminate before profiles are saved. A much-too-large duration may result in a longer wait time before slurm starts your job.
        # DDPPO PointNav empirical test from Aug 2020:
        #   startup time is 2 minutes and 100 steps takes 12 minutes
        if do_capture_step_range:
            slurm_job_termination_minutes = 10 + int(
                num_steps_to_capture * 15 / 100
            )
        else:
            slurm_job_termination_minutes = (
                nsys_capture_delay_seconds + nsys_capture_duration_seconds
            ) * 60 + 5

        # If capture_all_tasks==True, we capture profiles for all tasks. Beware large profile storage requirement in this case. If False, only one task runs with profiling. The other tasks run without profiling. In theory, all tasks behave similarly and so a single task's profile is representative of all tasks. In my DDPPO PointNav empirical test from Aug 2020, this was true.
        capture_all_tasks = False

    capture_cuda = False  # beware large profile storage requirement

    # Beware, support is poor on FAIR cluster and Colab machines due to older Nvidia drivers. For best OpenGL profiling, profile your desktop linux machine using the Nsight Systems GUI, not the nsys command-line tool.
    capture_opengl = False

    # nsys produces a .qdrep multithreaded trace file which can be viewed in the Nsight GUI. Optionally, it can export a .sqlite database file for use with habitat's compare_profiles.py.
    export_sqlite = False

    if do_capture_step_range:
        program_with_extra_args_str = (
            program_str
            + " PROFILING.CAPTURE_START_STEP "
            + str(capture_start_step)
            + " PROFILING.NUM_STEPS_TO_CAPTURE "
            + str(num_steps_to_capture)
        )
    else:
        program_with_extra_args_str = program_str

    if do_capture_step_range:
        capture_range_args = '--capture-range=nvtx -p "habitat_capture_range" --stop-on-range-end=true'
    else:
        capture_range_args = (
            "--delay="
            + str(nsys_capture_delay_seconds)
            + " --duration="
            + str(nsys_capture_duration_seconds)
        )

    task_capture_str = (
        """export HABITAT_PROFILING=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
"""
        + nsys_path
        + " profile --sample=none --trace-fork-before-exec=true --force-overwrite=true --trace=nvtx"
        + (",cuda" if capture_cuda else "")
        + (",opengl" if capture_opengl else "")
        + " "
        + capture_range_args
        + ' --output="'
        + profile_output_folder
        + "/"
        + profile_output_filename_base
        + '" '
        + ("--export=sqlite" if export_sqlite else "")
        + " "
        + program_with_extra_args_str
    )

    if do_slurm:
        if capture_all_tasks:
            slurm_task_str = (
                """#!/bin/sh
"""
                + task_capture_str
                + """
"""
            )
        else:
            slurm_task_str = (
                """#!/bin/sh
if [ ${SLURM_NODEID} == "0" ] && [ ${SLURM_LOCALID} == "0" ]
then
"""
                + task_capture_str
                + """
else
"""
                + program_str
                + """
fi"""
            )

        slurm_submit_str = (
            """#!/bin/bash
#SBATCH --job-name=capture_profile
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 8
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnfair
#SBATCH --time="""
            + str(slurm_job_termination_minutes)
            + """:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
set -x
srun capture_profile_slurm_task.sh %j
"""
        )

    profile_output_filepath = (
        profile_output_folder + "/" + profile_output_filename_base + ".qdrep"
    )
    if not do_slurm and os.path.exists(profile_output_filepath):
        print(
            "warning: {} already exists and will be overwritten.".format(
                profile_output_filepath
            )
        )

    if not os.path.exists(profile_output_folder):
        os.makedirs(profile_output_folder)
        print("created " + profile_output_folder)

    if do_slurm:
        with open("slurm_task.sh", "w") as f:
            f.write(slurm_task_str)
        print("wrote slurm_task.sh")

        with open("capture_profile_slurm.sh", "w") as f:
            f.write(slurm_submit_str)
        print("wrote capture_profile_slurm.sh")

        print(
            "\nTo start capture, do:\nchmod +x slurm_task.sh\nchmod +x capture_profile_slurm.sh\nsbatch capture_profile_slurm.sh"
        )

    else:
        with open("capture_profile.sh", "w") as f:
            f.write(task_capture_str)
        print("wrote capture_profile.sh")

        print(
            "\nTo start capture, do:\nchmod +x capture_profile.sh\n./capture_profile.sh"
        )
