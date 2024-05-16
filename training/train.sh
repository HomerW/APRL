MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py \
  --env_name=Go1SanityMujoco-Empty-SepRew-v0 \
  --save_buffer=True \
  --load_buffer \
  --utd_ratio=20 \
  --start_training=1000 \
  --config=configs/droq_config.py \
  --config.critic_layer_norm=True \
  --config.exterior_linear_c=0.0 \
  --config.target_entropy=-12 \
  --save_eval_videos=True \
  --eval_interval=1 \
  --save_training_videos=True \
  --training_video_interval=300 \
  --training_video_length_limit=300 \
  --eval_episodes=10 \
  --max_steps=40000 \
  --log_interval=1000 \
  --save_interval=10000 \
  --seed=0 \
  --project_name=APRL_sim_reproduce \
  --tqdm=True \
  --save_dir=saved_sim_exp \
  --task_config.action_interpolation=True \
  --task_config.enable_reset_policy=False \
  --task_config.Kp=20 \
  --task_config.Kd=1.0 \
  --task_config.limit_episode_length=0 \
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
  --action_curriculum_steps=30000 \
  --action_curriculum_start=0.35 \
  --action_curriculum_end=0.6 \
  --action_curriculum_linear=True \
  --action_curriculum_exploration_eps=0.15 \
  --task_config.filter_actions=8 \
  --reset_curriculum=True \
  --reset_criterion=dynamics_error \
  --task_config.rew_smooth_change_in_tdy_steps=1 \
  --threshold=1.5





