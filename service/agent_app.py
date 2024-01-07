# export PATH="/Users/puyuan/miniconda3/envs/arm64-py38/bin:$PATH"
# FLASK_APP=agent_app.py FLASK_ENV=development FLASK_DEBUG=1 flask run --port 5001
import time
import numpy as np
import torch
from flask import Flask, request, jsonify, make_response
# from flask_restplus import Api, Resource, fields
from flask_restx import Api, Resource, fields
from flask_cors import CORS

from threading import Thread
from sheep_model import SheepModel
from flask import Flask

app = Flask(__name__)
api = Api(
    app=app,
    version="0.0.1",
    title="gomoku_ui App",
    description="Play Sheep with Deep Reinforcement Learning, Powered by OpenDILab"
)

# CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

name_space = api.namespace('gomoku_ui', description='gomoku_ui APIs')
model = api.model(
    'gomoku_ui params', {
        'command': fields.String(required=False, description="Command Field", help="reset, step"),
        'argument': fields.Integer(required=False, description="Argument Field", help="reset->level, step->action"),
    }
)
MAX_ENV_NUM = 50
ENV_TIMEOUT_SECOND = 60
envs = {}
model = SheepModel(item_obs_size=80, item_num=30, global_obs_size=19)
ckpt = torch.load('ckpt_best.pth.tar', map_location='cpu')['model']
ckpt = {'item_encoder.encoder' + k.split('item_encoder')[-1] if 'item_encoder' in k else k: v for k, v in ckpt.items()}  # compatibility for v1 and v2 model
model.load_state_dict(ckpt)
import sys
sys.path.append("/Users/puyuan/code/LightZero/")
import pytest
from easydict import EasyDict
from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv

cfg = EasyDict(
  prob_random_agent=0,
  board_size=15,
  battle_mode='self_play_mode',
  channel_last=False,
  scale=False,
  agent_vs_human=False,
  bot_action_type='v1',  # {'v0', 'v1', 'alpha_beta_pruning'}
  prob_random_action_in_bot=0.,
  check_action_to_connect4_in_bot_v0=False,
  # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
  # If None, then the game will not be rendered.
  render_mode='state_realtime_mode',  # 'image_realtime_mode' # "state_realtime_mode",
  replay_path=None,
  screen_scaling=9,
  alphazero_mcts_ctree=False,
)
env = GomokuEnv(cfg)
obs = env.reset()
test_episodes = 1
# for i in range(test_episodes):
#   obs = env.reset()
#   # print('init board state: ', obs)
#   env.render()
#   while True:
#     # action = env.bot_action()
#     # action = env.random_action()
#     action = env.human_to_action()
#     print('action index of player 1 is:', action)
#     print('player 1: ' + env.action_to_string(action))
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       if reward > 0:
#         print('player 1 win')
#       else:
#         print('draw')
#       break
#
#     action = env.bot_action()
#     # action = env.random_action()
#     print('action index of player 2 is:', action)
#     print('player 2: ' + env.action_to_string(action))
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       if reward > 0:
#         print('player 2 win')
#       else:
#         print('draw')
#       break


def random_action(obs, env):
    action_mask = obs['action_mask']
    action = np.random.choice(len(action_mask), p=action_mask / action_mask.sum())
    return action


def env_monitor():
    while True:
        cur_time = time.time()
        pop_keys = []
        for k, v in envs.items():
            if cur_time - v['update_time'] >= ENV_TIMEOUT_SECOND:
                pop_keys.append(k)
        for k in pop_keys:
            envs.pop(k)
        time.sleep(1)


api.env_thread = Thread(target=env_monitor, daemon=True)
api.env_thread.start()


@name_space.route("/", methods=['POST'])
# @app.route('/your-endpoint', methods=['POST'])
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @api.expect(model)
    def post(self):
        try:
            print('position 1')
            t_start = time.time()
            data = request.json
            cmd, arg, uid = data['command'], data['argument'], data['uid']
            print(request.remote_addr)
            ip = request.remote_addr + uid
            print(cmd, arg, uid, ip)
            print('envs:', envs)

            # if ip not in envs:
            #     print('ip not in envs')
            #     if cmd == 'reset':
            #         if len(envs) >= MAX_ENV_NUM:
            #             response = jsonify(
            #                 {
            #                     "statusCode": 501,
            #                     "status": "No enough env resource, please wait a moment",
            #                 }
            #             )
            #             response.headers.add('Access-Control-Allow-Origin', '*')
            #             return response
            #         else:
            #             env = SheepEnv(1, agent=True, max_padding=True)
            #             env.seed(0)
            #             envs[ip] = {'env': env, 'update_time': time.time()}
            #     else:
            #         response = jsonify(
            #             {
            #                 "statusCode": 501,
            #                 "status": "No response for too long time, please reset the game",
            #             }
            #         )
            #         response.headers.add('Access-Control-Allow-Origin', '*')
            #         return response
            # else:
            #     env = envs[ip]['env']
            #     envs[ip]['update_time'] = time.time()

            if cmd == 'reset':
                obs = env.reset()
                bot_action = env.random_action()
                # action = model.compute_action(obs)
                print('reset bot action: {}'.format(bot_action))
                response = jsonify(
                    {
                        "statusCode": 200,
                        "status": "Execution action",
                        "result": {
                            'board': env.board.tolist(),  # 假设 env.board 是一个 NumPy 数组
                            'action': bot_action,
                            # 'done': done,
                            # 'info': info,
                        }
                    }
                )

            elif cmd == 'step':
                data = request.json
                action = data.get('action')  # 前端发送的动作  action: [i, j] 从0开始的，表示下在第i+1行，第j+1列
                action = action[0] * 15 + action[1]
                # 更新游戏环境
                observation, reward, done, info = env.step(action)
                # 如果游戏没有结束，获取 bot 的动作
                if not done:
                    # bot_action = env.random_action()
                    bot_action = env.bot_action()
                    # 更新环境状态
                    _, _, done, _ = env.step(bot_action)
                else:
                    bot_action = None
                # 准备响应数据
                observation, reward, done, info = None, None, None, None
                print('orig bot action: {}'.format(bot_action))
                bot_action = {'i':  int(bot_action // 15), 'j': int(bot_action % 15)}
                print('bot action: {}'.format(bot_action))
                response = {
                    "statusCode": 200,
                    "status": "Execution action",
                    "result": {
                        # 'board': env.board.tolist(),  # 假设 env.board 是一个 NumPy 数组
                        'board': None,  # 假设 env.board 是一个 NumPy 数组
                        # 'action': bot_action,  # bot action格式为 { i: x, j: y }
                        'action': bot_action,  # bot action格式为 { i: x, j: y }
                        'done': done,
                        'info': info
                    }
                }
            else:
                response = jsonify({
                    "statusCode": 500,
                    "status": "Invalid command: {}".format(cmd),
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
            print('backend process time: {}'.format(time.time() - t_start))
            print('current env number: {}'.format(len(envs)))
            # response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            import traceback
            print(repr(e))
            print(traceback.format_exc())
            response = jsonify({
                "statusCode": 500,
                "status": "Could not execute action",
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

if '__name__' == 'main':
  app.run()