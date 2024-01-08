from easydict import EasyDict

main_config = dict(
    exp_name='data_az_ctree/gomoku_alphazero_bot-mode_rand0.0_ns50_upc50_seed0_240108_165919',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=1,
            retry_type='reset',
            auto_reset=True,
            step_timeout=None,
            reset_timeout=None,
            retry_waiting_time=0.1,
            cfg_type='BaseEnvManagerDict',
            type='base',
            shared_memory=False,
        ),
        stop_value=2,
        n_evaluator_episode=1,
        env_name='Gomoku',
        board_size=15,
        battle_mode='play_with_bot_mode',
        battle_mode_in_simulation_env='self_play_mode',
        render_mode='image_savefile_mode',
        replay_path='./video',
        screen_scaling=9,
        channel_last=False,
        scale=True,
        agent_vs_human=False,
        bot_action_type='v1',
        prob_random_agent=0,
        prob_random_action_in_bot=0.0,
        alphazero_mcts_ctree=False,
        cfg_type='GomokuEnvDict',
        collector_env_num=8,
        evaluator_env_num=1,
        prob_expert_agent=0,
    ),
    policy=dict(
        model=dict(
            observation_shape=[3, 15, 15],
            num_res_blocks=1,
            num_channels=32,
            action_space_size=225,
        ),
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(
                    num_workers=0,
                ),
                log_policy=True,
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True,
                ),
                cfg_type='BaseLearnerDict',
            ),
        ),
        collect=dict(
            collector=dict(
                cfg_type='AlphaZeroCollectorDict',
                type='episode_alphazero',
                import_names=['lzero.worker.alphazero_collector'],
            ),
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                render={'render_freq': -1, 'mode': 'train_iter'},
                figure_path=None,
                cfg_type='InteractionSerialEvaluatorDict',
                type='alphazero',
                import_names=['lzero.worker.alphazero_evaluator'],
                stop_value=2,
                n_episode=1,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                type='advanced',
                replay_buffer_size=1000000,
                max_use=float('inf'),
                max_staleness=float('inf'),
                alpha=0.6,
                beta=0.4,
                anneal_step=100000,
                enable_track_used_data=False,
                deepcopy=False,
                thruput_controller=dict(
                    push_sample_rate_limit=dict(
                        max=float('inf'),
                        min=0,
                    ),
                    window_seconds=30,
                    sample_min_limit_ratio=1,
                ),
                monitor=dict(
                    sampled_data_attr=dict(
                        average_range=5,
                        print_freq=200,
                    ),
                    periodic_thruput=dict(
                        seconds=60,
                    ),
                ),
                cfg_type='AdvancedReplayBufferDict',
                save_episode=False,
            ),
        ),
        on_policy=False,
        cuda=True,
        multi_gpu=False,
        bp_update_sync=True,
        traj_len_inf=False,
        torch_compile=False,
        tensor_float_32=False,
        sampled_algo=False,
        gumbel_algo=False,
        update_per_collect=50,
        model_update_ratio=0.1,
        batch_size=256,
        optim_type='Adam',
        learning_rate=0.003,
        weight_decay=0.0001,
        momentum=0.9,
        grad_clip_value=0.5,
        value_weight=1.0,
        collector_env_num=8,
        evaluator_env_num=3,
        lr_piecewise_constant_decay=False,
        threshold_training_steps_for_final_lr=500000,
        manual_temperature_decay=False,
        threshold_training_steps_for_final_temperature=100000,
        fixed_temperature_value=0.25,
        mcts={'num_simulations': 50, 'max_moves': 512, 'root_dirichlet_alpha': 0.3, 'root_noise_weight': 0.25, 'pb_c_base': 19652, 'pb_c_init': 1.25},
        cfg_type='AlphaZeroPolicyDict',
        import_names=['lzero.policy.alphazero'],
        mcts_ctree=False,
        simulation_env_name='gomoku',
        simulation_env_config_type='play_with_bot',
        board_size=15,
        entropy_weight=0.0,
        n_episode=8,
        eval_freq=2000,
        device='cpu',
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='gomoku',
        import_names=['zoo.board_games.gomoku.envs.gomoku_env'],
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
        type='base',
    ),
    policy=dict(type='alphazero'),
)
create_config = EasyDict(create_config)
create_config = create_config
