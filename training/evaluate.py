#! /usr/bin/env python
import os
import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import rail_walker_gym
import rail_walker_interface
from checkpoint_util import initialize_project_log, load_latest_checkpoint, save_rollout
from task_config_util import apply_task_configs
import time
import rail_walker_gym.envs.register_mujoco
from jaxrl5.agents.sac.sac_learner_wdynamics import SACLearnerWithDynamics

FLAGS = flags.FLAGS

# ==================== Training Flags ====================
flags.DEFINE_string('env_name', 'Go1SanityMujoco-v0', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# ==================== Eval Flags ====================
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')

# ==================== Log / Save Flags ====================
flags.DEFINE_string('exp_name', 'Go1SanityMujoco-v0', 'Experiment name.')
flags.DEFINE_integer('log_interval', 1, 'Logging interval.')
flags.DEFINE_string("save_dir", "./saved", "Directory to save the model checkpoint and replay buffer.")

flags.DEFINE_boolean('save_eval_videos', False, 'Save videos during evaluation.')
flags.DEFINE_integer("eval_video_length_limit", 0, "Limit the length of evaluation videos.")
flags.DEFINE_integer("eval_video_interval", 1, "Interval to save videos during evaluation.")
flags.DEFINE_boolean('save_eval_rollouts', False, 'Save rollouts during evaluation.')

# ========================================================

# ==================== Joystick Task Flags ====================
config_flags.DEFINE_config_file(
    'task_config',
    'task_configs/default.py',
    'File path to the task/control config parameters.',
    lock_config=False)
config_flags.DEFINE_config_file(
    'reset_agent_config',
    'configs/reset_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)
# ========================================================

def main(_):
    # ==================== Setup WandB ====================
    wandb.init(project="go1-eval")
    wandb.run.name = FLAGS.exp_name
    wandb.config.update(FLAGS)
    
    # ==================== Setup Environment ====================
    eval_env : gym.Env = gym.make(FLAGS.env_name)

    _, eval_env = apply_task_configs(eval_env, FLAGS.env_name, 0, FLAGS.task_config, FLAGS.reset_agent_config, True)

    eval_env.seed(FLAGS.seed)

    if FLAGS.save_eval_rollouts:
        eval_env = rail_walker_gym.envs.wrappers.RolloutCollector(eval_env) # wrap environment to automatically collect rollout

    if FLAGS.save_eval_videos:
        eval_env = rail_walker_gym.envs.wrappers.WanDBVideoWrapper(eval_env, record_every_n_steps=FLAGS.eval_video_interval, log_name="evaluation/video", video_length_limit=FLAGS.eval_video_length_limit) # wrap environment to automatically save video to wandb
        eval_env.enableWandbVideo = True
    
    observation, info = eval_env.reset(return_info=True)
    done = False
    # ==================== Setup Checkpointing ====================
    initialize_project_log(FLAGS.save_dir)

    # ==================== Setup Learning Agent and Replay Buffer ====================
    agent_kwargs = dict(FLAGS.config)
    model_cls = agent_kwargs.pop('model_cls')

    agent = globals()[model_cls].create(
        FLAGS.seed, 
        eval_env.observation_space,
        eval_env.action_space,
        **agent_kwargs
    )

    agent_loaded_checkpoint_step, agent = load_latest_checkpoint(FLAGS.save_dir, agent, 0)
    if agent_loaded_checkpoint_step > 0:
        print(f"===================== Loaded checkpoint at step {agent_loaded_checkpoint_step} =====================")
    else:
        print(f"===================== No checkpoint found! =====================")
        print("Check directory:", FLAGS.save_dir)
        exit(0)
    # ==================== Start Eval ====================
    accumulated_info_dict = {}
    episode_counter = 0

    try:
        for i in tqdm.trange(0, 500000, initial=0, smoothing=0.1):
            
            # Force steps in environment WanDBVideoWrapper to be the same as our training steps
            # Notice: we need to make sure that the very outside wrapper of env is WandbVideoWrapper, otherwise setting this will not work
            if FLAGS.save_eval_videos:
                eval_env.set_wandb_step(i)

            action = agent.eval_actions(observation)
            
            # Step the environment
            next_observation, _, done, info = eval_env.step(action)
            done = done or ("TimeLimit.truncated" in info and info['TimeLimit.truncated'])

            observation = next_observation

            # Accumulate info to log
            for key in info.keys():
                if key in ['TimeLimit.truncated', 'TimeLimit.joystick_target_change', 'episode']:
                    continue
                value = info[key]
                if key not in accumulated_info_dict:
                    accumulated_info_dict[key] = [value]
                else:
                    accumulated_info_dict[key].append(value)

            if i % FLAGS.log_interval == 0:
                for k in accumulated_info_dict.keys():
                    v = accumulated_info_dict[k]
                    if v is None or len(v) <= 0:
                        continue
                    if k in ['fall_count','traversible_finished_lap_count']:
                        to_log = v[-1]
                    else:
                        to_log = np.mean(v)
                    wandb.log({'evaluation/' + str(k): to_log}, step=i)
                accumulated_info_dict = {}

            if done:
                for k, v in info['episode'].items():
                    decode = {'r': 'return', 'l': 'length', 't': 'time'}
                    wandb.log({f'evaluation/episode_{decode[k]}': v}, step=i)

                observation, info = eval_env.reset(return_info=True)
                done = False
                if FLAGS.save_eval_rollouts:
                    # Since we clear the collected rollouts after saving, we don't need to worry about saving episode rollout
                    save_rollout(FLAGS.save_dir, i, False, eval_env.collected_rollouts)
                    eval_env.collected_rollouts.clear()
                episode_counter += 1
            
            if episode_counter >= FLAGS.eval_episodes:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt, please wait for clean up...")
        import traceback
        traceback.print_exc()
    finally:
        print("======================== Cleaning up ========================")
        # ==================== Save Final Checkpoint ====================
        if FLAGS.save_eval_rollouts and len(eval_env.collected_rollouts) > 0:
            save_rollout(FLAGS.save_dir, i, False, eval_env.collected_rollouts)
        if FLAGS.save_eval_videos:
            eval_env._terminate_record()
        eval_env.close()
        print("======================== Done ========================")

if __name__ == '__main__':
    app.run(main)
