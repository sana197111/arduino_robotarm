import numpy as np
import gym
import math
from gym import spaces

# 환경 조성
class ArmGym(gym.Env):
    # 강화 학습 환경을 초기화. action_space와 observation_space가 정의
    def __init__(self, red_positions, blue_positions):
        super(ArmGym, self).__init__()

        self.red_positions = red_positions
        self.blue_positions = blue_positions

        self.servo_ranges = [(0, 150), (0, 150), (0, 150), (0, 150)]

        low_action = [0, 0, 0, 0]
        high_action = [150, 150, 150, 150]
        low_state = [0, 0, 0, 0, 0, 0, 0, 0]
        high_state = [640, 480, 640, 480, 150, 150, 150, 150]

        self.action_space = spaces.Box(low=np.array(low_action), high=np.array(high_action))
        self.observation_space = spaces.Box(low=np.array(low_state), high=np.array(high_state))

    # 에이전트가 환경에 있는 상태를 기반으로 특정 행동을 수행하고 지정된 행동으로 상태를 업데이트하는데 사용. 그런 다음 상태, 보상 및 종료 플래그를 반환
    def step(self, action):
        # 서브모터 angle
        servo1, servo2, servo3, servo4 = action

        # red_positions, blue_positions의 마지막 값을 사용하거나 다른 전략을 사용할 수 있습니다.
        red_x, red_y = self.red_positions[-1]
        blue_x, blue_y = self.blue_positions[-1]

        # 상태 및 보상 설정
        reward = self.calculate_reward(red_x, red_y, blue_x, blue_y)
        state = [red_x, red_y, blue_x, blue_y, servo1, servo2, servo3, servo4]

        return state, reward, {}

    # 환경을 초기 상태로 되돌림
    def reset(self):
        red_x, red_y = self.red_positions[-1]
        blue_x, blue_y = self.blue_positions[-1]

        return [red_x, red_y, blue_x, blue_y, 90, 90, 90, 5]

    # 환경의 현재 상태를 시각화, 이 코드에는 구현되지 않았지만 시각화를 필요로 하는 경우 사용할 수 있음.
    def render(self, mode='human', close=False):
        pass

    def calculate_reward(self, red_x, red_y, blue_x, blue_y):
        distance = math.sqrt((red_x - blue_x)**2 + (red_y - blue_y)**2)
        reward = 0
        distance_threshold = 5  # 두 스티커 간 거리 임계값

        if distance < distance_threshold:  # 빨간 스티커와 파란 스티커 간 거리가 임계값보다 작으면
            reward += 0  # 더 이상 보상이 증가하지 않습니다.
        else:
            reward = -abs(distance - distance_threshold)  # 거리가 임계값보다 클 때 보상은 거리와 임계값의 차이에 페널티를 줍니다.

        reward += red_y + blue_y  # 빨간색 스티커와 파란색 스티커의 y 좌표가 둘 다 증가할 때마다 추가 보상을 받습니다.

        return reward