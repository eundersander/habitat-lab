#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Manually control the articulated agent to interact with the environment. Run as
```
python examples/interative_play.py
```

To Run you need PyGame installed (to install run `pip install pygame==2.0.1`).

By default this controls with velocity control (which makes controlling the
agent hard). To use IK control instead add the `--add-ik` command line argument.

Controls:
- For velocity control
    - 1-7 to increase the motor target for the articulated agent arm joints
    - Q-U to decrease the motor target for the articulated agent arm joints
- For IK control
    - W,S,A,D to move side to side
    - E,Q to move up and down
- I,J,K,L to move the articulated agent base around
- PERIOD to print the current world coordinates of the articulated agent base.
- Z to toggle the camera to free movement mode. When in free camera mode:
    - W,S,A,D,Q,E to translate the camera
    - I,J,K,L,U,O to rotate the camera
    - B to reset the camera position
- X to change the articulated agent that is being controlled (if there are multiple articulated agents).

Change the task with `--cfg benchmark/rearrange/close_cab.yaml` (choose any task under the `habitat-lab/habitat/config/task/rearrange/` folder).

Change the grip type:
- Suction gripper `task.actions.arm_action.grip_controller "SuctionGraspAction"`

To record a video: `--save-obs` This will save the video to file under `data/vids/` specified by `--save-obs-fname` (by default `vid.mp4`).
"""

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import argparse
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from typing import Any

import gym.spaces as spaces
import magnum as mn
import numpy as np
import torch

import habitat
import habitat.gym.gym_wrapper as gym_wrapper
import habitat.tasks.rearrange.rearrange_task
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    GfxReplayMeasureMeasurementConfig,
    OracleNavActionConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.core.logging import logger
from habitat.tasks.rearrange.actions.actions import ArmEEAction
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.common import flatten_dict
from habitat.utils.gui_app_wrapper import (
    GuiAppWrapper,
    ImageDrawer,
    InputWrapper,
    RenderWrapper,
    SimWrapper,
)
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.articulated_agent_controllers import (
    HumanoidRearrangeController,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.utils.common import get_action_space_info
from habitat_sim.utils import viz_utils as vut

try:
    import pygame
except ImportError:
    pygame = None

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"

use_pygame = False  # select OS/window backend: Magnum or pygame
if not use_pygame:
    use_replay_batch_renderer = False  # choose classic or batch renderer

    import habitat_sim
    from habitat_sim import ReplayRenderer, ReplayRendererConfiguration

DEFAULT_CFG = "benchmark/rearrange/play.yaml"
DEFAULT_RENDER_STEPS_LIMIT = 60
SAVE_VIDEO_DIR = "./data/vids"
SAVE_ACTIONS_DIR = "./data/interactive_play_replays"


class Controller(ABC):
    def __init__(self, agent_idx, is_multi_agent):
        self._agent_idx = agent_idx
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass

    def on_environment_reset(self):
        pass


def clean_dict(d, remove_prefix):
    ret_d = {}
    for k, v in d.spaces.items():
        if k.startswith(remove_prefix):
            new_k = k[len(remove_prefix) :]
            if isinstance(v, spaces.Dict):
                ret_d[new_k] = clean_dict(v, remove_prefix)
            else:
                ret_d[new_k] = v
        elif not k.startswith("agent"):
            ret_d[k] = v
    return spaces.Dict(ret_d)


class BaselinesController(Controller):
    def __init__(self, agent_idx, is_multi_agent, cfg_path, env):
        super().__init__(agent_idx, is_multi_agent)

        config = get_baselines_config(
            cfg_path,
            [
                "habitat_baselines/rl/policy=hl_fixed",
                "habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills",
                "habitat_baselines.num_environments=1",
            ],
        )
        policy_cls = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.name
        )
        self._env_ac = env.action_space
        env_obs = env.observation_space
        self._agent_k = f"agent_{agent_idx}_"
        if is_multi_agent:
            self._env_ac = clean_dict(self._env_ac, self._agent_k)
            env_obs = clean_dict(env_obs, self._agent_k)

        self._gym_ac_space = gym_wrapper.create_action_space(self._env_ac)
        gym_obs_space = gym_wrapper.smash_observation_space(
            env_obs, list(env_obs.keys())
        )
        self._actor_critic = policy_cls.from_config(
            config,
            gym_obs_space,
            self._gym_ac_space,
            orig_action_space=self._env_ac,
        )
        self._action_shape, _ = get_action_space_info(self._gym_ac_space)
        self._step_i = 0

    def act(self, obs, env):
        masks = torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.bool,
        )
        if self._step_i == 0:
            masks = ~masks
        self._step_i += 1
        hxs = torch.ones(
            (
                1,
                1,
            ),
            dtype=torch.float32,
        )
        prev_ac = torch.ones(
            (
                1,
                self._action_shape[0],
            ),
            dtype=torch.float32,
        )
        obs = flatten_dict(obs)
        obs = TensorDict(
            {
                k[len(self._agent_k) :]: torch.tensor(v).unsqueeze(0)
                for k, v in obs.items()
                if k.startswith(self._agent_k)
            }
        )
        with torch.no_grad():
            action_data = self._actor_critic.act(obs, hxs, prev_ac, masks)
        action = gym_wrapper.continuous_vector_action_to_hab_dict(
            self._env_ac, self._gym_ac_space, action_data.env_actions[0]
        )

        # temp do random base actions
        action["action_args"]["base_vel"] = torch.rand_like(
            action["action_args"]["base_vel"]
        )

        def change_ac_name(k):
            if "pddl" in k:
                return k
            else:
                return self._agent_k + k

        action["action"] = [change_ac_name(k) for k in action["action"]]
        action["action_args"] = {
            change_ac_name(k): v.cpu().numpy()
            for k, v in action["action_args"].items()
        }
        return action, False, False, action_data.rnn_hidden_states


class HumanController(Controller):
    def __init__(self, agent_idx, is_multi_agent, gui_input):
        super().__init__(agent_idx, is_multi_agent)
        self._gui_input = gui_input

    def act(self, obs, env):
        if self._is_multi_agent:
            agent_k = f"agent_{self._agent_idx}_"
        else:
            agent_k = ""
        arm_k = f"{agent_k}arm_action"
        grip_k = f"{agent_k}grip_action"
        base_k = f"{agent_k}base_vel"
        arm_name = f"{agent_k}arm_action"
        base_name = f"{agent_k}base_velocity"
        ac_spaces = env.action_space.spaces

        if arm_name in ac_spaces:
            arm_action_space = ac_spaces[arm_name][arm_k]
            arm_ctrlr = env.task.actions[arm_name].arm_ctrlr
            arm_action = np.zeros(arm_action_space.shape[0])
            grasp = 0
        else:
            arm_ctrlr = None
            arm_action = None
            grasp = None

        base_action: Any = None
        if base_name in ac_spaces:
            base_action_space = ac_spaces[base_name][base_k]
            base_action = np.zeros(base_action_space.shape[0])
        else:
            base_action = None

        KeyNS = InputWrapper.KeyNS
        gui_input = self._gui_input

        should_end = False
        should_reset = False

        if gui_input.get_key_down(KeyNS.ESC):
            should_end = True
        elif gui_input.get_key_down(KeyNS.M):
            should_reset = True
        elif gui_input.get_key_down(KeyNS.N):
            env._sim.navmesh_visualization = not env._sim.navmesh_visualization

        if base_action is not None:
            # Base control
            base_action = [0, 0]
            if gui_input.get_key(KeyNS.J):
                # Left
                base_action[1] += 1
            if gui_input.get_key(KeyNS.L):
                # Right
                base_action[1] -= 1
            if gui_input.get_key(KeyNS.K):
                # Back
                base_action[0] -= 1
            if gui_input.get_key(KeyNS.I):
                # Forward
                base_action[0] += 1

        if isinstance(arm_ctrlr, ArmEEAction):
            EE_FACTOR = 0.5
            # End effector control
            if gui_input.get_key_down(KeyNS.D):
                arm_action[1] -= EE_FACTOR
            elif gui_input.get_key_down(KeyNS.A):
                arm_action[1] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.W):
                arm_action[0] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.S):
                arm_action[0] -= EE_FACTOR
            elif gui_input.get_key_down(KeyNS.Q):
                arm_action[2] += EE_FACTOR
            elif gui_input.get_key_down(KeyNS.E):
                arm_action[2] -= EE_FACTOR
        else:
            # Velocity control. A different key for each joint
            if gui_input.get_key_down(KeyNS.Q):
                arm_action[0] = 1.0
            elif gui_input.get_key_down(KeyNS.ONE):
                arm_action[0] = -1.0

            elif gui_input.get_key_down(KeyNS.W):
                arm_action[1] = 1.0
            elif gui_input.get_key_down(KeyNS.TWO):
                arm_action[1] = -1.0

            elif gui_input.get_key_down(KeyNS.E):
                arm_action[2] = 1.0
            elif gui_input.get_key_down(KeyNS.THREE):
                arm_action[2] = -1.0

            elif gui_input.get_key_down(KeyNS.R):
                arm_action[3] = 1.0
            elif gui_input.get_key_down(KeyNS.FOUR):
                arm_action[3] = -1.0

            elif gui_input.get_key_down(KeyNS.T):
                arm_action[4] = 1.0
            elif gui_input.get_key_down(KeyNS.FIVE):
                arm_action[4] = -1.0

            elif gui_input.get_key_down(KeyNS.Y):
                arm_action[5] = 1.0
            elif gui_input.get_key_down(KeyNS.SIX):
                arm_action[5] = -1.0

            elif gui_input.get_key_down(KeyNS.U):
                arm_action[6] = 1.0
            elif gui_input.get_key_down(KeyNS.SEVEN):
                arm_action[6] = -1.0

        if gui_input.get_key_down(KeyNS.P):
            logger.info("[play.py]: Unsnapping")
            # Unsnap
            grasp = -1
        elif gui_input.get_key_down(KeyNS.O):
            # Snap
            logger.info("[play.py]: Snapping")
            grasp = 1

        if gui_input.get_key_down(KeyNS.PERIOD):
            # Print the current position of the robot, useful for debugging.
            pos = [
                float("%.3f" % x) for x in env._sim.robot.sim_obj.translation
            ]
            rot = env._sim.robot.sim_obj.rotation
            ee_pos = env._sim.robot.ee_transform.translation
            logger.info(
                f"Robot state: pos = {pos}, rotation = {rot}, ee_pos = {ee_pos}"
            )
        elif gui_input.get_key_down(KeyNS.COMMA):
            # Print the current arm state of the robot, useful for debugging.
            joint_state = [
                float("%.3f" % x) for x in env._sim.robot.arm_joint_pos
            ]

            logger.info(f"Robot arm joint state: {joint_state}")

        action_names = []
        action_args = {}
        if base_action is not None:
            action_names.append(base_name)
            action_args.update(
                {
                    base_k: base_action,
                }
            )
        if arm_action is not None:
            action_names.append(arm_name)
            action_args.update(
                {
                    arm_k: arm_action,
                    grip_k: grasp,
                }
            )
        if len(action_names) == 0:
            raise ValueError("No active actions for human controller.")

        return (
            {"action": action_names, "action_args": action_args},
            should_reset,
            should_end,
            {},
        )


class GuiHumanoidController(Controller):
    def __init__(
        self, agent_idx, is_multi_agent, gui_input, env, walk_pose_path
    ):
        self.agent_idx = agent_idx
        super().__init__(self.agent_idx, is_multi_agent)
        self.humanoid_controller = HumanoidRearrangeController(walk_pose_path)
        self.env = env
        self._gui_input = gui_input
        self._walk_dir = None

    def get_articulated_agent(self):
        return self.env._sim.agents_mgr[self.agent_idx].articulated_agent

    def on_environment_reset(self):
        super().on_environment_reset()
        base_pos = self.get_articulated_agent().base_pos
        self.humanoid_controller.reset(base_pos)

    def get_random_joint_action(self):
        # Add random noise to human arms but keep global transform
        (
            joint_trans,
            root_trans,
        ) = self.get_articulated_agent().get_joint_transform()
        # Divide joint_trans by 4 since joint_trans has flattened quaternions
        # and the dimension of each quaternion is 4
        num_joints = len(joint_trans) // 4
        root_trans = np.array(root_trans)
        index_arms_start = 10
        joint_trans_quat = [
            mn.Quaternion(
                mn.Vector3(joint_trans[(4 * index) : (4 * index + 3)]),
                joint_trans[4 * index + 3],
            )
            for index in range(num_joints)
        ]
        rotated_joints_quat = []
        for index, joint_quat in enumerate(joint_trans_quat):
            random_vec = np.random.rand(3)
            # We allow for maximum 10 angles per step
            random_angle = np.random.rand() * 10
            rotation_quat = mn.Quaternion.rotation(
                mn.Rad(random_angle), mn.Vector3(random_vec).normalized()
            )
            if index > index_arms_start:
                joint_quat *= rotation_quat
            rotated_joints_quat.append(joint_quat)
        joint_trans = np.concatenate(
            [
                np.array(list(quat.vector) + [quat.scalar])
                for quat in rotated_joints_quat
            ]
        )
        humanoidjoint_action = np.concatenate(
            [joint_trans.reshape(-1), root_trans.transpose().reshape(-1)]
        )
        return humanoidjoint_action

    def act(self, obs, env):
        if self._is_multi_agent:
            agent_k = f"agent_{self._agent_idx}_"
        else:
            agent_k = ""
        humanoidjoint_name = f"{agent_k}humanoidjoint_action"
        ac_spaces = env.action_space.spaces

        do_humanoidjoint_action = humanoidjoint_name in ac_spaces

        KeyNS = InputWrapper.KeyNS
        gui_input = self._gui_input

        should_end = False
        should_reset = False

        if gui_input.get_key_down(KeyNS.ESC):
            should_end = True
        elif gui_input.get_key_down(KeyNS.M):
            should_reset = True
        elif gui_input.get_key_down(KeyNS.N):
            # todo: move outside this controller
            env._sim.navmesh_visualization = not env._sim.navmesh_visualization

        if do_humanoidjoint_action:
            humancontroller_base_user_input = [0, 0]
            # temp keyboard controls to test humanoid controller
            if gui_input.get_key(KeyNS.I):
                # move in world-space x+ direction ("east")
                humancontroller_base_user_input[0] += 1
            if gui_input.get_key(KeyNS.K):
                # move in world-space x- direction ("west")
                humancontroller_base_user_input[0] -= 1

            if self._walk_dir:
                humancontroller_base_user_input[0] += self._walk_dir.x
                humancontroller_base_user_input[1] += self._walk_dir.z

        action_names = []
        action_args = {}
        if do_humanoidjoint_action:
            if True:
                relative_pos = mn.Vector3(
                    humancontroller_base_user_input[0],
                    0,
                    humancontroller_base_user_input[1],
                )
                pose, root_trans = self.humanoid_controller.get_walk_pose(
                    relative_pos, distance_multiplier=1.0
                )
                humanoidjoint_action = self.humanoid_controller.vectorize_pose(
                    pose, root_trans
                )
            else:
                pass
                # reference code
                # humanoidjoint_action = self.get_random_joint_action()
            action_names.append(humanoidjoint_name)
            action_args.update(
                {
                    "human_joints_trans": humanoidjoint_action,
                }
            )

        return (
            {"action": action_names, "action_args": action_args},
            should_reset,
            should_end,
            {},
        )


class ControllerHelper:
    def __init__(self, env, args, gui_input):
        self.n_robots = len(env._sim.agents_mgr)
        is_multi_agent = self.n_robots > 1

        self.env = env

        gui_controller: Controller = None
        if args.use_humanoid_controller:
            gui_controller = GuiHumanoidController(
                0, is_multi_agent, gui_input, env, args.walk_pose_path
            )
        else:
            gui_controller = HumanController(0, is_multi_agent, gui_input)

        self.controllers = []
        self.n_robots = self.n_robots
        self.all_hxs = [None for _ in range(self.n_robots)]
        self.active_controllers = [0, 1]
        self.controllers = [
            gui_controller,
            BaselinesController(
                1,
                is_multi_agent,
                "habitat-baselines/habitat_baselines/config/rearrange/rl_hierarchical.yaml",
                env,
            ),
        ]

    def update(self, obs):
        all_names = []
        all_args = {}
        end_play = False
        reset_ep = False
        for i in self.active_controllers:
            (
                ctrl_action,
                ctrl_reset_ep,
                ctrl_end_play,
                self.all_hxs[i],
            ) = self.controllers[i].act(obs, self.env)
            end_play = end_play or ctrl_end_play
            reset_ep = reset_ep or ctrl_reset_ep
            all_names.extend(ctrl_action["action"])
            all_args.update(ctrl_action["action_args"])
        action = {"action": tuple(all_names), "action_args": all_args}
        return action, end_play, reset_ep

    def on_environment_reset(self):
        for i in self.active_controllers:
            self.controllers[i].on_environment_reset()


def play_env(env, args, config):
    render_steps_limit = None
    if args.no_render:
        render_steps_limit = DEFAULT_RENDER_STEPS_LIMIT

    obs = env.reset()

    if not args.no_render:
        draw_obs = observations_to_image(obs, {})
        pygame.init()
        screen = pygame.display.set_mode(
            [draw_obs.shape[1], draw_obs.shape[0]]
        )

    update_idx = 0
    target_fps = 60.0
    prev_time = time.time()
    all_obs = []
    total_reward = 0

    gfx_measure = env.task.measurements.measures.get(
        GfxReplayMeasure.cls_uuid, None
    )

    gui_input = None  # todo
    ctrl_helper = ControllerHelper(env, args, gui_input)

    ctrl_helper.on_environment_reset()

    def reset_helper():
        nonlocal total_reward
        env.reset()
        ctrl_helper.on_environment_reset()
        total_reward = 0

    while True:
        if render_steps_limit is not None and update_idx > render_steps_limit:
            break

        KeyNS = InputWrapper.KeyNS
        gui_input = None  # todo

        if not args.no_render and gui_input.get_key_down(KeyNS.x):
            ctrl_helper.active_controllers[0] = (
                ctrl_helper.active_controllers[0] + 1
            ) % ctrl_helper.n_robots
            logger.info(
                f"Controlled agent changed. Controlling agent {ctrl_helper.active_controllers[0]}."
            )

        end_play = False
        reset_ep = False
        if args.no_render:
            action = {"action": "empty", "action_args": {}}
        else:
            action, end_play, reset_ep = ctrl_helper.update(obs)

        obs = env.step(action)

        if not args.no_render and gui_input.get_key_down(KeyNS.c):
            pddl_action = env.task.actions["PDDL_APPLY_ACTION"]
            logger.info("Actions:")
            actions = pddl_action._action_ordering
            for i, action in enumerate(actions):
                logger.info(f"{i}: {action}")
            entities = pddl_action._entities_list
            logger.info("Entities")
            for i, entity in enumerate(entities):
                logger.info(f"{i}: {entity}")
            action_sel = input("Enter Action Selection: ")
            entity_sel = input("Enter Entity Selection: ")
            action_sel = int(action_sel)
            entity_sel = [int(x) + 1 for x in entity_sel.split(",")]
            ac = np.zeros(pddl_action.action_space["pddl_action"].shape[0])
            ac_start = pddl_action.get_pddl_action_start(action_sel)
            ac[ac_start : ac_start + len(entity_sel)] = entity_sel

            env.step(
                {
                    "action": "PDDL_APPLY_ACTION",
                    "action_args": {"pddl_action": ac},
                }
            )

        if not args.no_render and gui_input.get_key_down(KeyNS.g):
            pred_list = env.task.sensor_suite.sensors[
                "all_predicates"
            ]._predicates_list
            pred_values = obs["all_predicates"]
            logger.info("\nPredicate Truth Values:")
            for i, (pred, pred_value) in enumerate(
                zip(pred_list, pred_values)
            ):
                logger.info(f"{i}: {pred.compact_str} = {pred_value}")

        if reset_ep:
            # Clear the saved keyframes.
            if gfx_measure is not None:
                gfx_measure.get_metric(force_get=True)
            reset_helper()
        if end_play:
            break

        update_idx += 1

        info = env.get_metrics()
        reward_key = [k for k in info if "reward" in k]
        if len(reward_key) > 0:
            reward = info[reward_key[0]]
        else:
            reward = 0.0

        total_reward += reward
        info["Total Reward"] = total_reward

        use_ob = observations_to_image(obs, info)
        if not args.skip_render_text:
            use_ob = overlay_frame(use_ob, info)

        draw_ob = use_ob[:]

        if not args.no_render:
            draw_ob = np.transpose(draw_ob, (1, 0, 2))
            draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
            screen.blit(draw_obuse_ob, (0, 0))
            pygame.display.update()
        if args.save_obs:
            all_obs.append(draw_ob)  # type: ignore[assignment]

        if not args.no_render:
            pygame.event.pump()
        if env.episode_over:
            reset_helper()

        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0 / target_fps - diff, 0)
        time.sleep(delay)
        prev_time = curr_time

    if args.save_obs:
        all_obs = np.array(all_obs)  # type: ignore[assignment]
        all_obs = np.transpose(all_obs, (0, 2, 1, 3))  # type: ignore[assignment]
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        vut.make_video(
            np.expand_dims(all_obs, 1),
            0,
            "color",
            osp.join(SAVE_VIDEO_DIR, args.save_obs_fname),
        )
    if gfx_measure is not None and args.gfx:
        gfx_str = gfx_measure.get_metric(force_get=True)
        write_gfx_replay(
            gfx_str, config.habitat.task, env.current_episode.episode_id
        )

    if not args.no_render:
        pygame.quit()


def has_pygame():
    return pygame is not None


class PlaySimWrapper(SimWrapper):
    def __init__(self, args, config, gui_input):
        # todo: remove the debug/third-person sensor
        with habitat.config.read_write(config):
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
        self.env = habitat.Env(config=config)
        self.obs = self.env.reset()

        self.ctrl_helper = ControllerHelper(self.env, args, gui_input)

        self.ctrl_helper.on_environment_reset()

        self.cam_zoom_dist = 1.0
        self.gui_input = gui_input

        self._debug_line_render = None  # will be set later via a hack in gui_app_wrapper

    def do_raycast_and_get_walk_dir(self):
        dir = None
        ray = self.gui_input.mouse_ray

        target_y = 0.15

        if not ray or ray.direction.y >= 0 or ray.origin.y <= target_y:
            return dir

        dist_to_target_y = -ray.origin.y / ray.direction.y

        target = ray.origin + ray.direction * dist_to_target_y

        # raycast_results = self.env._sim.cast_ray(ray=ray)
        # if raycast_results.has_hits():
        #     hit_info = raycast_results.hits[0]
        #     self._debug_line_render.draw_circle(hit_info.point + mn.Vector3(0, 0.05, 0), 0.03, mn.Color3(0, 1, 0))

        agent_idx = 0
        art_obj = self.env._sim.agents_mgr[agent_idx].articulated_agent.sim_obj
        robot_root = art_obj.transformation

        pathfinder = self.env._sim.pathfinder
        snapped_pos = pathfinder.snap_point(target)
        snapped_start_pos = robot_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        if found_path:
            path_color = mn.Color3(0, 0, 1)
            # skip rendering first point. It is at the object root, at the wrong height
            for path_i in range(0, len(path.points) - 1):
                a = mn.Vector3(path.points[path_i])
                b = mn.Vector3(path.points[path_i + 1])

                self._debug_line_render.draw_transformed_line(a, b, path_color)
                # env.sim.viz_ids[f"next_loc_{path_i}"] = env.sim.visualize_position(
                #     path.points[path_i], env.sim.viz_ids[f"next_loc_{path_i}"]
                # )

            end_pos = mn.Vector3(path.points[-1])
            self._debug_line_render.draw_circle(end_pos, 0.16, path_color)

            if self.gui_input.get_key(InputWrapper.KeyNS.B):
                if len(path.points) >= 2:
                    dir = mn.Vector3(path.points[1]) - mn.Vector3(
                        path.points[0]
                    )

        color = mn.Color3(0, 0.5, 0) if found_path else mn.Color3(0.5, 0, 0)
        self._debug_line_render.draw_circle(target, 0.08, color)

        return dir

    def sim_update(self, dt):
        # todo: pipe end_play somewhere

        walk_dir = self.do_raycast_and_get_walk_dir()
        # hack
        self.ctrl_helper.controllers[0]._walk_dir = walk_dir

        action, end_play, reset_ep = self.ctrl_helper.update(self.obs)

        self.obs = self.env.step(action)

        if reset_ep:
            self.obs = self.env.reset()
            self.ctrl_helper.on_environment_reset()

        post_sim_update_dict = {}

        if self.gui_input.mouse_scroll_offset != 0:
            zoom_sensitivity = 0.07
            if self.gui_input.mouse_scroll_offset < 0:
                self.cam_zoom_dist *= (
                    1.0
                    + -self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            else:
                self.cam_zoom_dist /= (
                    1.0 + self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            max_zoom_dist = 50.0
            min_zoom_dist = 0.1
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist, min_zoom_dist, max_zoom_dist
            )

        agent_idx = 0
        art_obj = self.env._sim.agents_mgr[agent_idx].articulated_agent.sim_obj
        robot_root = art_obj.transformation
        lookat = robot_root.translation + mn.Vector3(0, 1, 0)
        cam_transform = mn.Matrix4.look_at(
            lookat + mn.Vector3(0.5, 1, 0.5).normalized() * self.cam_zoom_dist,
            lookat,
            mn.Vector3(0, 1, 0),
        )
        post_sim_update_dict["cam_transform"] = cam_transform

        post_sim_update_dict[
            "keyframes"
        ] = self.env._sim.gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()

        # post_sim_update_dict["debug_images"] = [
        #     observations_to_image(self.obs, {})]
        # convert depth to RGB

        def flip_vertical(obs):
            converted_obs = np.empty_like(obs)
            for row in range(obs.shape[0]):
                converted_obs[row, :] = obs[obs.shape[0] - row - 1, :]
            return converted_obs

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # post_sim_update_dict["debug_images"] = [
        #     flip_vertical(depth_to_rgb(self.obs["agent_1_robot_head_depth"]))
        # ]

        return post_sim_update_dict


class ReplayRenderWrapper(RenderWrapper):
    def __init__(self, width, height):
        self.viewport_size = mn.Vector2i(width, height)
        # arbitrary uuid
        self._sensor_uuid = "rgb_camera"

        cfg = ReplayRendererConfiguration()
        cfg.num_environments = 1
        cfg.standalone = False  # Context is owned by the GLFW window
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.uuid = self._sensor_uuid
        camera_sensor_spec.resolution = [
            height,
            width,
        ]
        camera_sensor_spec.position = np.array([0, 0, 0])
        camera_sensor_spec.orientation = np.array([0, 0, 0])

        cfg.sensor_specifications = [camera_sensor_spec]
        cfg.gpu_device_id = 0  # todo
        cfg.force_separate_semantic_scene_graph = False
        cfg.leave_context_with_background_renderer = False
        self._replay_renderer = (
            ReplayRenderer.create_batch_replay_renderer(cfg)
            if use_replay_batch_renderer
            else ReplayRenderer.create_classic_replay_renderer(cfg)
        )

        self._image_drawer = ImageDrawer(max_width=1024, max_height=1024)
        self._debug_images = []
        self._need_render = True

    def post_sim_update(self, post_sim_update_dict):
        keyframes = post_sim_update_dict["keyframes"]
        self.cam_transform = post_sim_update_dict["cam_transform"]

        env_index = 0
        for keyframe in keyframes:
            self._replay_renderer.set_environment_keyframe(env_index, keyframe)

        if "debug_images" in post_sim_update_dict:
            self._debug_images = post_sim_update_dict["debug_images"]

        if len(keyframes):
            self._need_render = True

    def unproject(self, viewport_pos):
        return self._replay_renderer.unproject(0, viewport_pos)

    def render_update(self, dt):
        if not self._need_render:
            return

        # self._replay_renderer.debug_line_render(0).draw_circle(mn.Vector3(0, 1, 0), 0.25, mn.Color3(1, 0, 1))

        transform = self.cam_transform
        env_index = 0
        self._replay_renderer.set_sensor_transform(
            env_index, self._sensor_uuid, transform
        )

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )
        mn.gl.default_framebuffer.bind()

        self._replay_renderer.render(mn.gl.default_framebuffer)

        # arrange debug images on right side of frame, tiled down from the top
        dest_y = 0
        for image in self._debug_images:
            assert isinstance(image, (np.ndarray, torch.Tensor))
            self._image_drawer.draw(
                image, self.viewport_size[0] - image.shape[0], dest_y
            )
            dest_y += image[1]

        self._need_render = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--save-obs", action="store_true", default=False)
    parser.add_argument("--save-obs-fname", type=str, default="play.mp4")
    parser.add_argument("--play-cam-res", type=int, default=512)
    parser.add_argument(
        "--skip-render-text", action="store_true", default=False
    )
    parser.add_argument(
        "--same-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--skip-task",
        action="store_true",
        default=False,
        help="If true, then do not add the render camera for better visualization",
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control.",
    )

    parser.add_argument(
        "--control-humanoid",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--use-humanoid-controller",
        action="store_true",
        default=False,
        help="Control humanoid agent.",
    )

    parser.add_argument(
        "--gfx",
        action="store_true",
        default=False,
        help="Save a GFX replay file.",
    )
    parser.add_argument("--load-actions", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )

    args = parser.parse_args()
    if use_pygame and not has_pygame() and not args.no_render:
        raise ImportError(
            "Need to install PyGame (run `pip install pygame==2.0.1`)"
        )

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        task_config.actions[
            "agent_1_oracle_nav_action"
        ] = OracleNavActionConfig(agent_index=1)

        if not args.same_task:
            sim_config.debug_render = True
            agent_config = get_agent_config(sim_config=sim_config)
            if use_pygame:
                agent_config.sim_sensors.update(
                    {
                        "third_rgb_sensor": ThirdRGBSensorConfig(
                            height=args.play_cam_res, width=args.play_cam_res
                        )
                    }
                )
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

        if args.gfx:
            sim_config.habitat_sim_v0.enable_gfx_replay_save = True
            task_config.measurements.update(
                {"gfx_replay_measure": GfxReplayMeasureMeasurementConfig()}
            )

        if args.never_end:
            env_config.max_episode_steps = 0

        if args.control_humanoid:
            args.disable_inverse_kinematics = True

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

    if use_pygame:
        with habitat.Env(config=config) as env:
            play_env(env, args, config)
    else:
        width = 1000  # args.play_cam_res
        height = 640  # args.play_cam_res
        gui_app_wrapper = GuiAppWrapper(width, height)
        sim_wrapper = PlaySimWrapper(
            args, config, gui_app_wrapper.get_sim_input()
        )
        # note this must be created after GuiAppWrapper due to OpenGL stuff
        render_wrapper = ReplayRenderWrapper(width, height)
        gui_app_wrapper.set_sim_and_render_wrappers(
            sim_wrapper, render_wrapper
        )
        gui_app_wrapper.exec()
