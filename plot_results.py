import os
import os.path as osp
import plotly.graph_objects as go

if __name__ == '__main__':

    fig_dir = 'figs'
    result_dir = 'results'
    env_dirs = [osp.join(result_dir, o) for o in os.listdir(result_dir) if osp.isdir(osp.join(result_dir, o))]

    algorithms = ['ddpg', 'a2c', 'ppo']
    networks = [
        'mlp2_64',
        'mlp2_256',
        'mlp2_1024',
        'mlp3_64',
        'mlp3_256',
        'mlp3_1024',
        'cnn_mlp_64',
        'cnn_mlp_256',
        'cnn_mlp_1024'
    ]
    r_networks = [
        'lstm_64',
        'lstm_256'
    ]

