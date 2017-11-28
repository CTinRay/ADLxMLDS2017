def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.
    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--max_timesteps', type=int, default=2e8)
    parser.add_argument('--gamma', type=float, default=0.99999)
    parser.add_argument('--exploration_steps', type=int, default=75000)
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prioritized_replay_eps', type=float, default=1e-4)
    parser.add_argument('--target_network_update_freq', type=int,
                        default=4000)
    parser.add_argument('--buffer_size', type=int, default=1e5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    return parser
