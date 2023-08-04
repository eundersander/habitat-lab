#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys
from enum import Enum

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import argparse
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Set


import magnum as mn
import numpy as np
from controllers import ControllerHelper
from hitl_tutorial import Tutorial, generate_tutorial
from magnum.platform.glfw import Application
from serialize_utils import (
    NullRecorder,
    StepRecorder,
    save_as_gzip,
    save_as_json_gzip,
    save_as_pickle_gzip,
)

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.gui.gui_application import GuiAppDriver, GuiApplication
from habitat.gui.replay_gui_app_renderer import ReplayGuiAppRenderer
from habitat_baselines.config.default import get_config as get_baselines_config

from server.server import launch_server_process, terminate_server_process
from server.interprocess_record import send_keyframe_to_networking_thread, get_queued_client_states
from cube_test import CubeTest
from remote_gui_input import RemoteGuiInput
import json
from app_states.app_state_rearrange import AppStateRearrange

do_network_server = False
do_cube_test = False
use_simplified_hssd_objects = True
use_simplified_ycb_objects = True
use_simplified_robot_meshes = True
remove_extra_config_in_model_filepaths = True  # removes cases of "configs/../"
use_collision_proxies_for_hssd_objects = False
use_glb_black_list = False
glb_black_list = [
    "decomposed/fcbd5b4248b9d224b000d36ec0a923993d8d69d7/fcbd5b4248b9d224b000d36ec0a923993d8d69d7_part_11.glb",
    "decomposed/3ae4103c3669dd551e1ab0982374940bf70d355a/3ae4103c3669dd551e1ab0982374940bf70d355a_part_3.glb",
    "6/61da788a7fd67fe1a1bf270e16f458caadb0225e.glb",
    "b/bc71f73e9f0f3ab3fee5f8a00085d8f6d51e29be.glb",
    "f/fa86d57a2be6ef6d18ecf7a03230c0ac5f369c51.glb",
    "d/de016d31ad24004bf75eeb8c0107e966c626133f.glb",
    "c/c3e45e226f51e328829dcf0afc2b3baf5de0ac0c.glb",
    "1/14eec36e26211ebb7a14a24a85800ff206c9f4ca.glb",
    "7/79208602e24336c463c0e52f5ed94794756de9df.glb",
    "7/76eb8317661751dd4897b39db29df5db517e2d5c.glb",
    "0/0d1b069fb10cfdf67080e59b1bf0e73fe84151dd.glb",
    "b/b6269374895fd9fdc7c86da11ad46a03e8b87682.glb",
    "9/906d71ff11f2bdcea5d37916ed1fd0c77259579d.glb",
    "e/ea4826625a4ad17e2bd2f4013acdb1c26b568999.glb",
    "5/538f11be1a17f9f645ca8e83fd5555baa3abdf66.glb",
    "a/a1ef5b93eaf029da57b49fb31ed5e05c01003749.glb",
    # "004_sugar_box",  # I don't know why this is complaining so much
]    

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = (
    "data/humanoids/humanoid_data/walking_motion_processed_smplx.pkl"
)

DEFAULT_CFG = "experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml"


def requires_habitat_sim_with_bullet(callable_):
    @wraps(callable_)
    def wrapper(*args, **kwds):
        assert (
            habitat_sim.built_with_bullet
        ), f"Habitat-sim is built without bullet, but {callable_.__name__} requires Habitat-sim with bullet."
        return callable_(*args, **kwds)

    return wrapper


def get_pretty_object_name_from_handle(obj_handle_str):
    handle_lower = obj_handle_str.lower()
    filtered_str = "".join(filter(lambda c: c.isalpha(), handle_lower))
    return filtered_str


class SandboxState(Enum):
    CONTROLLING_AGENT = 1
    TUTORIAL = 2


@requires_habitat_sim_with_bullet
class SandboxDriver(GuiAppDriver):
    def __init__(self, args, config, gui_input):
        self._dataset_config = config.habitat.dataset
        self._play_episodes_filter_str = args.episodes_filter
        self._num_recorded_episodes = 0
        self._args = args

        with habitat.config.read_write(config):
            # needed so we can provide keyframes to GuiApplication
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
            config.habitat.simulator.concur_render = False

        dataset = self._make_dataset(config=config)
        self.env = habitat.Env(config=config, dataset=dataset)

        if args.gui_controlled_agent_index is not None:
            sim_config = config.habitat.simulator
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            oracle_nav_sensor_key = f"{gui_agent_key}_has_finished_oracle_nav"
            if oracle_nav_sensor_key in self.env.task.sensor_suite.sensors:
                del self.env.task.sensor_suite.sensors[oracle_nav_sensor_key]

        self._save_filepath_base = args.save_filepath_base
        self._save_episode_record = args.save_episode_record
        self._step_recorder = (
            StepRecorder() if self._save_episode_record else NullRecorder()
        )
        self._episode_recorder_dict = {}

        self._save_gfx_replay_keyframes: bool = args.save_gfx_replay_keyframes
        self._recording_keyframes: List[str] = []

        self.ctrl_helper = ControllerHelper(
            self.env, config, args, gui_input, self._step_recorder
        )

        self._debug_line_render = None
        self._debug_images = args.debug_images

        is_free_camera_mode = self.ctrl_helper.get_gui_controlled_agent_index() is None
        if is_free_camera_mode and args.first_person_mode:
            raise RuntimeError(
                "--first-person-mode must be used with --gui-controlled-agent-index"
            )

        def local_end_episode(do_reset=False):
            self._end_episode(do_reset)

        self._app_state_rearrange = AppStateRearrange(
            args, 
            config,
            self.env,
            self.get_sim(), 
            gui_input,
            self.ctrl_helper.get_gui_agent_controller(), 
            lambda: self._compute_action_and_step_env(),
            local_end_episode,
            lambda: self._get_recent_metrics()
        )            

        self._num_iter_episodes: int = len(self.env.episode_iterator.episodes)  # type: ignore
        self._num_episodes_done: int = 0
        self._reset_environment()

        self._remote_gui_input = None
        if do_network_server:
            launch_server_process()
            self._remote_gui_input = RemoteGuiInput()

        self._cube_test = None
        if do_cube_test:
            self._cube_test = CubeTest(self.get_sim())

        # temp; todo: set up AppState base class
        self._app_state = self._app_state_rearrange

    def _make_dataset(self, config):
        from habitat.datasets import make_dataset

        dataset_config = config.habitat.dataset
        dataset = make_dataset(
            id_dataset=dataset_config.type, config=dataset_config
        )

        if self._play_episodes_filter_str is not None:
            max_num_digits: int = len(str(len(dataset.episodes)))

            def get_play_episodes_ids(play_episodes_filter_str):
                play_episodes_ids: Set[str] = set()
                for ep_filter_str in play_episodes_filter_str.split(" "):
                    if ":" in ep_filter_str:
                        range_params = map(int, ep_filter_str.split(":"))
                        play_episodes_ids.update(
                            episode_id.zfill(max_num_digits)
                            for episode_id in map(str, range(*range_params))
                        )
                    else:
                        episode_id = ep_filter_str
                        play_episodes_ids.add(episode_id.zfill(max_num_digits))

                return play_episodes_ids

            play_episodes_ids_set: Set[str] = get_play_episodes_ids(
                self._play_episodes_filter_str
            )
            dataset.episodes = [
                ep
                for ep in dataset.episodes
                if ep.episode_id.zfill(max_num_digits) in play_episodes_ids_set
            ]

        return dataset

    def _get_recent_metrics(self):
        assert self._metrics
        return self._metrics

    def _env_step(self, action):
        self._obs = self.env.step(action)
        self._metrics = self.env.get_metrics()

    def _next_episode_exists(self):
        return self._num_episodes_done < self._num_iter_episodes - 1

    def _compute_action_and_step_env(self):
        # # step env if episode is active
        # # otherwise pause simulation (don't do anything)
        # if not self._env_episode_active():
        #     return

        action = self.ctrl_helper.update(self._obs)
        self._env_step(action)

        if self._save_episode_record:
            self._record_action(action)
            self._record_task_state()
            self._record_metrics(self._metrics)
            self._step_recorder.finish_step()

    def _find_episode_save_filepath_base(self):
        retval = (
            self._save_filepath_base + "." + str(self._num_recorded_episodes)
        )
        return retval

    def _save_episode_recorder_dict(self):
        if not len(self._step_recorder._steps):
            return

        filepath_base = self._find_episode_save_filepath_base()

        json_filepath = filepath_base + ".json.gz"
        save_as_json_gzip(self._episode_recorder_dict, json_filepath)

        pkl_filepath = filepath_base + ".pkl.gz"
        save_as_pickle_gzip(self._episode_recorder_dict, pkl_filepath)

    def _reset_episode_recorder(self):
        assert self._step_recorder
        ep_dict: Any = dict()
        ep_dict["start_time"] = datetime.now()
        ep_dict["dataset"] = self._dataset_config
        ep_dict["scene_id"] = self.env.current_episode.scene_id
        ep_dict["episode_id"] = self.env.current_episode.episode_id

        ep_dict["target_obj_ids"] = self._target_obj_ids
        ep_dict[
            "goal_positions"
        ] = (
            self._goal_positions
        )  # [list[goal_pos] for goal_pos in self._goal_positions]

        self._step_recorder.reset()
        ep_dict["steps"] = self._step_recorder._steps

        self._episode_recorder_dict = ep_dict

    def _reset_environment(self):
        self._obs = self.env.reset()
        self._metrics = self.env.get_metrics()
        self.ctrl_helper.on_environment_reset()
        # self._held_target_obj_idx = None
        # self._num_remaining_objects = None  # resting, not at goal location yet
        # self._num_busy_objects = None  # currently held by non-gui agents

        # sim = self.get_sim()
        # temp_ids, goal_positions_np = sim.get_targets()
        # self._target_obj_ids = [
        #     sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        # ]
        # self._goal_positions = [mn.Vector3(pos) for pos in goal_positions_np]
        self._app_state_rearrange.on_environment_reset()

        self._sandbox_state = (
            SandboxState.TUTORIAL
            if args.show_tutorial
            else SandboxState.CONTROLLING_AGENT
        )
        self._tutorial: Tutorial = (
            generate_tutorial(
                self.get_sim(),
                self.ctrl_helper.get_gui_controlled_agent_index(),
                self._create_camera_lookat(),
            )
            if args.show_tutorial
            else None
        )

        # reset recorded keyframes and episode recorder data:
        # do not clead self._recording_keyframes as for now,
        # save a gfx-replay file per session not per episode
        # self._recording_keyframes.clear()
        if self._save_episode_record:
            self._reset_episode_recorder()

    def _check_save_episode_data(self, session_ended):
        assert self._save_filepath_base
        saved_keyframes, saved_episode_data = False, False
        if self._save_gfx_replay_keyframes and session_ended:
            self._save_recorded_keyframes_to_file()
            saved_keyframes = True
        if self._save_episode_record:
            self._save_episode_recorder_dict()
            saved_episode_data = True

        if saved_keyframes or saved_episode_data:
            self._num_recorded_episodes += 1

    def _end_episode(self, do_reset=False):
        self._check_save_episode_data(session_ended=do_reset == False)
        self._num_episodes_done += 1

        if do_reset and self._next_episode_exists():
            self._reset_environment()

    # todo: find a way to construct the object with this
    def set_debug_line_render(self, debug_line_render):
        self._debug_line_render = debug_line_render
        self._debug_line_render.set_line_width(3)
        if self._cube_test:
            self._cube_test._debug_line_render = self._debug_line_render
        if self._remote_gui_input:
            self._remote_gui_input._debug_line_render = self._debug_line_render
        self._app_state_rearrange._debug_line_render = debug_line_render

    def set_text_drawer(self, text_drawer):
        # self._text_drawer = text_drawer
        self._app_state_rearrange._text_drawer = text_drawer

    # trying to get around mypy complaints about missing sim attributes
    def get_sim(self) -> Any:
        return self.env.task._sim

    def _save_recorded_keyframes_to_file(self):
        if not self._recording_keyframes:
            return

        # Consolidate recorded keyframes into a single json string
        # self._recording_keyframes format:
        #     List['{"keyframe":{...}', '{"keyframe":{...}',...]
        # Output format:
        #     '{"keyframes":[{...},{...},...]}'
        json_keyframes = ",".join(
            keyframe[12:-1] for keyframe in self._recording_keyframes
        )
        json_content = '{{"keyframes":[{}]}}'.format(json_keyframes)

        # Save keyframes to file
        filepath = self._save_filepath_base + ".gfx_replay.json.gz"
        save_as_gzip(json_content.encode("utf-8"), filepath)


    def _record_action(self, action):
        action_args = action["action_args"]

        # These are large arrays and they massively bloat the record file size, so
        # let's exclude them.
        keys_to_clear = [
            "human_joints_trans",
            "agent_0_human_joints_trans",
            "agent_1_human_joints_trans",
        ]
        for key in keys_to_clear:
            if key in action_args:
                action_args[key] = None

        self._step_recorder.record("action", action)

    def _record_metrics(self, metrics):
        # We don't want to include this.
        if "gfx_replay_keyframes_string" in metrics:
            del metrics["gfx_replay_keyframes_string"]

        self._step_recorder.record("metrics", metrics)


    def sim_update(self, dt):
        # todo: pipe end_play somewhere
        post_sim_update_dict: Dict[str, Any] = {}

        if self._remote_gui_input:
            self._remote_gui_input.update()

        # if self.gui_input.get_key_down(GuiInput.KeyNS.ESC):
        #     self._end_episode()
        #     post_sim_update_dict["application_exit"] = True

        # if self.gui_input.get_key_down(GuiInput.KeyNS.M):
        #     self._end_episode(do_reset=True)

        # # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        # viz_anim_speed = 2.0
        # self._viz_anim_fraction = (
        #     self._viz_anim_fraction + dt * viz_anim_speed
        # ) % 1.0

        # if self._env_episode_active():
        #     self._update_task()
        #     self._update_grasping_and_set_act_hints()

        # # Navmesh visualization only works in the debug third-person view
        # # (--debug-third-person-width), not the main sandbox viewport. Navmesh
        # # visualization is only implemented for simulator-rendering, not replay-
        # # rendering.
        # if self.gui_input.get_key_down(GuiInput.KeyNS.N):
        #     self.env._sim.navmesh_visualization = (  # type: ignore
        #         not self.env._sim.navmesh_visualization  # type: ignore
        #     )

        # if self._sandbox_state == SandboxState.CONTROLLING_AGENT:
        #     self._sim_update_controlling_agent(dt)
        # else:
        #     self._sim_update_tutorial(dt)

        if self._cube_test:
            self._cube_test.pre_step()

        # # self.cam_transform is set to new value after
        # # self._sim_update_controlling_agent(dt) or self._sim_update_tutorial(dt)
        # post_sim_update_dict["cam_transform"] = self.cam_transform

        # if self._update_cursor_style():
        #     post_sim_update_dict["application_cursor"] = self._cursor_style
        self._app_state.sim_update(dt, post_sim_update_dict)

        keyframes = (
            self.get_sim().gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        )

        if self._save_gfx_replay_keyframes:
            for keyframe in keyframes:
                self._recording_keyframes.append(keyframe)

        if self._args.hide_humanoid_in_gui:
            # Hack to hide skinned humanoids in the GUI viewport. Specifically, this
            # hides all render instances with a filepath starting with
            # "data/humanoids/humanoid_data", by replacing with an invalid filepath.
            # Gfx-replay playback logic will print a warning to the terminal and then
            # not attempt to render the instance. This is a temp hack until
            # skinning is supported in gfx-replay.
            for i in range(len(keyframes)):
                keyframes[i] = keyframes[i].replace(
                    '"creation":{"filepath":"data/humanoids/humanoid_data',
                    '"creation":{"filepath":"invalid_filepath',
                )

        for i in range(len(keyframes)):
            if use_simplified_hssd_objects:
                keyframes[i] = keyframes[i].replace(
                    'data/fpss/fphab/objects',
                    'data/hitl/simplified/fpss/fphab/objects',
                )
            if use_simplified_ycb_objects:
                keyframes[i] = keyframes[i].replace(
                    'data/objects/ycb/',
                    'data/hitl/simplified/objects/ycb/',
                )
            if use_simplified_robot_meshes:
                keyframes[i] = keyframes[i].replace(
                    'data/robots/hab_spot_arm/',
                    'data/hitl/simplified/robots/hab_spot_arm/',
                )
            if remove_extra_config_in_model_filepaths:
                keyframes[i] = keyframes[i].replace(
                    'configs/../',
                    '/',
                )
            if use_glb_black_list:
                for black_item in glb_black_list:
                    keyframes[i] = keyframes[i].replace(
                        black_item,
                        black_item + ".invalid_filepath",
                    )

        # elif use_collision_proxies_for_hssd_objects:
        #     import re
        #     for i in range(len(keyframes)):
        #         pattern = r'(/objects/[^.]+)\.glb'
        #         replacement = r'\1.collider.glb'
        #         keyframes[i] = re.sub(pattern, replacement, keyframes[i])



        post_sim_update_dict["keyframes"] = keyframes

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # reference code for visualizing a camera sensor in the app GUI
        assert set(self._debug_images).issubset(set(self._obs.keys())), (
            f"Camera sensors ids: {list(set(self._debug_images).difference(set(self._obs.keys())))} "
            f"not in available sensors ids: {list(self._obs.keys())}"
        )
        debug_images = (
            depth_to_rgb(self._obs[k]) if "depth" in k else self._obs[k]
            for k in self._debug_images
        )
        post_sim_update_dict["debug_images"] = [
            np.flipud(image) for image in debug_images
        ]

        self._app_state._update_help_text()

        if self._cube_test:
            self._cube_test.post_step()

        if self._remote_gui_input:
            self._remote_gui_input.on_frame_end()

        if do_network_server:
            for keyframe_json in keyframes:
                single_item_array = json.loads(keyframe_json)
                assert len(single_item_array) == 1
                keyframe_obj = single_item_array
                send_keyframe_to_networking_thread(keyframe_obj)

        return post_sim_update_dict





def _parse_debug_third_person(args, framebuffer_size):
    viewport_multiplier = mn.Vector2(
        framebuffer_size.x / args.width, framebuffer_size.y / args.height
    )

    do_show = args.debug_third_person_width != 0

    width = args.debug_third_person_width
    # default to square aspect ratio
    height = (
        args.debug_third_person_height
        if args.debug_third_person_height != 0
        else width
    )

    width = int(width * viewport_multiplier.x)
    height = int(height * viewport_multiplier.y)

    return do_show, width, height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sps",
        type=int,
        default=30,
        help="Target rate to step the environment (steps per second); actual SPS may be lower depending on your hardware",
    )
    parser.add_argument(
        "--width",
        default=1280,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=720,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--gui-controlled-agent-index",
        type=int,
        default=None,
        help=(
            "GUI-controlled agent index (must be >= 0 and < number of agents). "
            "Defaults to None, indicating that all the agents are policy-controlled. "
            "If none of the agents is GUI-controlled, the camera is switched to 'free camera' mode "
            "that lets the user observe the scene (instead of controlling one of the agents)"
        ),
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--cfg-opts",
        nargs="*",
        default=list(),
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--debug-images",
        nargs="*",
        default=list(),
        help=(
            "Visualize camera sensors (corresponding to `--debug-images` keys) in the app GUI."
            "For example, to visualize agent1's head depth sensor set: --debug-images agent_1_head_depth"
        ),
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--use-batch-renderer",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--debug-third-person-width",
        default=0,
        type=int,
        help="If specified, enable the debug third-person camera (habitat.simulator.debug_render) with specified viewport width",
    )
    parser.add_argument(
        "--debug-third-person-height",
        default=0,
        type=int,
        help="If specified, use the specified viewport height for the debug third-person camera",
    )
    parser.add_argument(
        "--max-look-up-angle",
        default=15,
        type=float,
        help="Look up angle limit.",
    )
    parser.add_argument(
        "--min-look-down-angle",
        default=-60,
        type=float,
        help="Look down angle limit.",
    )
    parser.add_argument(
        "--first-person-mode",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--can-grasp-place-threshold",
        default=1.2,
        type=float,
        help="Object grasp/place proximity threshold",
    )
    parser.add_argument(
        "--episodes-filter",
        default=None,
        type=str,
        help=(
            "Episodes filter in the form '0:10 12 14:20:2', "
            "where single integer number (`12` in this case) represents an episode id, "
            "colon separated integers (`0:10' and `14:20:2`) represent start:stop:step ids range."
        ),
    )
    # temp argument:
    # allowed to switch between oracle baseline nav
    # and random base vel action
    parser.add_argument(
        "--sample-random-baseline-base-vel",
        action="store_true",
        default=False,
        help="Sample random BaselinesController base vel",
    )
    parser.add_argument(
        "--show-tutorial",
        action="store_true",
        default=False,
        help="Shows an intro sequence that helps familiarize the user to the scene and task in a HITL context.",
    )
    parser.add_argument(
        "--hide-humanoid-in-gui",
        action="store_true",
        default=False,
        help="Hide the humanoid in the GUI viewport. Note it will still be rendered into observations fed to policies. This option is a workaround for broken skinned humanoid rendering in the GUI viewport.",
    )
    parser.add_argument(
        "--save-gfx-replay-keyframes",
        action="store_true",
        default=False,
        help="Save the gfx-replay keyframes to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-episode-record",
        action="store_true",
        default=False,
        help="Save recorded episode data to file. Use --save-filepath-base to specify the filepath base.",
    )
    parser.add_argument(
        "--save-filepath-base",
        default=None,
        type=str,
        help="Filepath base used for saving various session data files. Include a full path including basename, but not an extension.",
    )
    args = parser.parse_args()
    if (
        args.save_gfx_replay_keyframes or args.save_episode_record
    ) and not args.save_filepath_base:
        raise ValueError(
            "--save-gfx-replay-keyframes and/or --save-episode-record flags are enabled, "
            "but --save-filepath-base argument is not set. Specify filepath base for the session episode data to be saved."
        )

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(args, framebuffer_size)

    config = get_baselines_config(args.cfg, args.cfg_opts)
    # config = habitat.get_config(args.cfg, args.cfg_opts)
    with habitat.config.read_write(config):
        habitat_config = config.habitat
        env_config = habitat_config.environment
        sim_config = habitat_config.simulator
        task_config = habitat_config.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        # task_config.actions[
        #     "agent_1_oracle_nav_action"
        # ] = OracleNavActionConfig(agent_index=1)

        agent_config = get_agent_config(sim_config=sim_config)

        if show_debug_third_person:
            sim_config.debug_render = True
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=debug_third_person_height,
                        width=debug_third_person_width,
                    )
                }
            )
            agent_key = "" if len(sim_config.agents) == 1 else "agent_0_"
            args.debug_images.append(f"{agent_key}third_rgb")

        # Code below is ported from interactive_play.py. I'm not sure what it is for.
        if True:
            if "composite_success" in task_config.measurements:
                task_config.measurements.composite_success.must_call_stop = (
                    False
                )
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

        if args.gui_controlled_agent_index is not None:
            # make sure gui_controlled_agent_index is valid
            if not (
                args.gui_controlled_agent_index >= 0
                and args.gui_controlled_agent_index < len(sim_config.agents)
            ):
                print(
                    f"--gui-controlled-agent-index argument value ({args.gui_controlled_agent_index}) "
                    f"must be >= 0 and < number of agents ({len(sim_config.agents)})"
                )
                exit()

            # avoid camera sensors for GUI-controlled agents
            gui_controlled_agent_config = get_agent_config(
                sim_config, agent_id=args.gui_controlled_agent_index
            )
            gui_controlled_agent_config.sim_sensors.clear()

            # make sure chosen articulated_agent_type is supported
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            if (
                sim_config.agents[gui_agent_key].articulated_agent_type
                != "KinematicHumanoid"
            ):
                print(
                    f"Selected agent for GUI control is of type {sim_config.agents[gui_agent_key].articulated_agent_type}, "
                    "but only KinematicHumanoid is supported at the moment."
                )
                exit()

            # use humanoidjoint_action for GUI-controlled KinematicHumanoid
            # for example, humanoid oracle-planner-based policy uses following actions:
            # base_velocity, rearrange_stop, pddl_apply_action, oracle_nav_action
            task_actions = task_config.actions
            gui_agent_actions = [
                action_key
                for action_key in task_actions.keys()
                if action_key.startswith(gui_agent_key)
            ]
            for action_key in gui_agent_actions:
                task_actions.pop(action_key)

            action_prefix = (
                f"{gui_agent_key}_" if len(sim_config.agents) > 1 else ""
            )
            task_actions[
                f"{action_prefix}humanoidjoint_action"
            ] = HumanoidJointActionConfig(
                agent_index=args.gui_controlled_agent_index
            )

    driver = SandboxDriver(args, config, gui_app_wrapper.get_sim_input())

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and args.gui_controlled_agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    viewport_rect = None
    if show_debug_third_person:
        # adjust main viewport to leave room for the debug third-person camera on the right
        assert framebuffer_size.x > debug_third_person_width
        viewport_rect = mn.Range2Di(
            mn.Vector2i(0, 0),
            mn.Vector2i(
                framebuffer_size.x - debug_third_person_width,
                framebuffer_size.y,
            ),
        )

    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size,
        viewport_rect,
        args.use_batch_renderer,
    )
    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    # sloppy: provide replay app renderer's debug_line_render to our driver
    driver.set_debug_line_render(
        app_renderer._replay_renderer.debug_line_render(0)
    )
    # sloppy: provide app renderer's text_drawer to our driver
    driver.set_text_drawer(app_renderer._text_drawer)

    gui_app_wrapper.exec()

    if do_network_server:
        terminate_server_process()
