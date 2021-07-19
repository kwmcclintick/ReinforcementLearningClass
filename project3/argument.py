def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--buffer_size', type=int, default=10000, help='replay buffer memory size')
    parser.add_argument('--learn_start', type=int, default=5000, help='begin to learn after this many time steps')
    parser.add_argument('--update_target', type=int, default=5000, help='perform update target network after this many time steps')
    parser.add_argument('--learning_rate', type=float, default=1.5*1e-4, help='sgd learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--n_episodes', type=int, default=30000, help='number of training epsiodes')
    return parser
