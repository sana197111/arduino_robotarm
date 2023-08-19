import numpy as np
import threading
import serial
import random
import time
import concurrent.futures

from capture_frame import find_sticker_and_save_coordinates
from ArmGym import ArmGym

lock = threading.Lock()

def update_positions():
    global red_positions, blue_positions
    while True:
        red_positions, blue_positions = find_sticker_and_save_coordinates()
        time.sleep(0.5)
    #     lock.acquire()
    #     try:
    #         red_positions, blue_positions = find_sticker_and_save_coordinates()
    #     finally:
    #         lock.release()
    #     time.sleep(0.5)

class QLAgent:
    def __init__(self, low_state, high_state, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_size = tuple([(high - low) - 1 for low, high in zip(low_state, high_state)])

        self.q_table = np.zeros(self.state_size + (action_size,))

    def state_to_index(self, state, low_state, high_state):
        # 각 상태변수에서 최솟값을 뺀 값들을 정수로 변환하여 인덱스로 사용합니다.
        index = [int((state[i] - low_state[i])) for i in range(len(state))]
        return tuple(index)
    
    def choose_action(self, state, low_state, high_state):
        # 상태를 Q-table 인덱스로 변환합니다.
        state_index = self.state_to_index(state, low_state, high_state)

        # 입실론(epsilon) 값이 무작위 난수보다 작거나 같으면 무작위 행동을 선택하고,
        # 그렇지 않으면 현재 상태 최대 Q-값을 갖는 행동을 선택합니다.
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_index])

    def learn(self, state, action, reward, next_state, done, low_state, high_state):
        # 상태와 다음 상태를 Q-table 인덱스로 변환합니다.
        state_index = self.state_to_index(state, low_state, high_state)
        next_state_index = self.state_to_index(next_state, low_state, high_state)
        
        # Q-table에서 현재 상태 및 행동에 대한 예측값을 가져옵니다.
        predict = self.q_table[state_index + (action,)]
        target = reward

        # 종료 상태가 아닌 경우, 세기(gamma)를 곱하고 최대 Q-values를 더합니다.
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_index])

        # Q-table을 업데이트합니다. 학습률(alpha)와 예측값과 목표값의 차이를 곱합니다.
        self.q_table[state_index + (action,)] += self.alpha * (target - predict)

        # 입실론의 값을 감쇠시킵니다.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ser = serial.Serial('COM3', 9600)

# # 아두이노로 각도 보내기
# def send_angles_to_arduino(angles):
#     ser.write(','.join(str(angle) for angle in angles) + '\n')

# 환경 생성
def q_learning(red_positions, blue_positions):
    env = ArmGym(red_positions_copy, blue_positions_copy)

    num_episodes = 1000
    num_steps = 50

    low_state = [0, 0, 0, 0, 0, 0, 0, 0]
    high_state = [640, 480, 640, 480, 150, 150, 150, 150]
    agent = QLAgent(low_state, high_state, env.action_space.shape[0])

    for episode in range(num_episodes):
        lock.acquire()
        try:
            red_positions_copy = list(red_positions)
            blue_positions_copy = list(blue_positions)
        finally:
            lock.release()
        
        # state 초기화
        state = env.reset()
        # total_reward 초기화
        total_reward = 0

        for step in range(num_steps):
            # 행동 선택, 다음 상태, 보상 및 종료 여부를 얻습니다.
            action = agent.choose_action(state, env.observation_space.low, env.observation_space.high)
            next_state, reward, done, _ = env.step(env.action_space.sample())

            # 에이전트가 학습합니다.
            agent.learn(state, action, reward, next_state, done, env.observation_space.low, env.observation_space.high)

            # 아두이노에 서보모터 각도 값을 전송합니다.
            # send_angles_to_arduino(action[:4])

            total_reward += reward
            state = next_state

            # 현재 상태, 행동 및 보상 출력
            print(f'Step: {step}, State: {state}, Action: {action}, Reward: {reward}')

        if (episode+1) % 100 == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# # 위치 정보를 업데이트하는 쓰레드를 생성합니다.
# position_thread = threading.Thread(target=update_positions)
# position_thread.start()

# # 강화학습을 실행하는 쓰레드를 생성합니다.
# q_learning_thread = threading.Thread(target=q_learning, args=(red_positions, blue_positions))
# q_learning_thread.start()

# def test():
#     while True:
#         print("1")
#         time.sleep(1)

# test_thread = threading.Thread(target=test)
# test_thread.start()


def main():
    # 위치 정보를 업데이트하는 함수와 강화학습을 실행하는 함수를 실행합니다.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        update_positions_future = executor.submit(update_positions)
        # q_learning_future = executor.submit(q_learning)
        # q_learning_future = executor.submit(test)

    # 두 개의 쓰레드가 완료될 때까지 기다립니다.
    for future in concurrent.futures.as_completed([update_positions_future, q_learning_future]):
        pass  # 쓰레드가 완료되면 아무 작업도 수행하지 않습니다.

if __name__ == '__main__':
    # main()
    update_positions()