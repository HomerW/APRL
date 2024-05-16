MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python evaluate.py \
  --env_name=Go1SanityMujoco-Empty-SepRew-v0 \
  --exp_name=walk \
  --eval_episodes=100 \
  --eval_video_length_limit=1000 \
  --save_eval_rollouts \
  --save_dir=/home/homer/research/APRL/training/saved_sim_exp/Go1SanityMujoco-Empty-SepRew-v0_s0_maxgr100.00_ac30000_0.35to0.6_linearly \
  --config=configs/droq_config.py \
  --config.critic_layer_norm=True \
  --config.exterior_linear_c=0.0 \
  --config.target_entropy=-12 \
  --task_config.action_interpolation=True \
  --task_config.enable_reset_policy=False \
  --task_config.Kp=20 \
  --task_config.Kd=1.0 \
  --task_config.limit_episode_length=300 \
  --task_config.action_range=0.35 \
  --task_config.frame_stack=0 \
  --task_config.action_history=1 \
  --task_config.rew_target_velocity=1.5 \
  --task_config.rew_energy_penalty_weight=0.0 \
  --task_config.rew_qpos_penalty_weight=2.0 \
  --task_config.rew_smooth_torque_penalty_weight=0.005 \
  --task_config.rew_pitch_rate_penalty_factor=0.4 \
  --task_config.rew_roll_rate_penalty_factor=0.2 \
  --task_config.rew_joint_diagonal_penalty_weight=0.00 \
  --task_config.rew_joint_shoulder_penalty_weight=0.00 \
  --task_config.rew_joint_acc_penalty_weight=0.0 \
  --task_config.rew_joint_vel_penalty_weight=0.0 \
  --task_config.center_init_action=True \
  --task_config.rew_contact_reward_weight=0.0 \
  --task_config.filter_actions=8 \
  --task_config.rew_smooth_change_in_tdy_steps=1 \





