"""
Worm Robot Simulation - RL Trainer
DQN 학습 루프 구현 (간단한 버전)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdevs.simulator import Simulator
from rl.replay_buffer import ReplayBuffer
from config import STATUS_RUNNING, STATUS_WIN, STATUS_FAIL


class DQNTrainer:
    """
    DQN 학습 루프를 관리하는 클래스
    
    주의: 현재는 에피소드 전체를 실행 후 보상 계산하는 간단한 버전
    """

    def __init__(
        self,
        agent,
        create_system_fn,
        num_episodes=1000,
        termination_time=100,
        batch_size=32,
        buffer_size=10000,
        log_interval=10,
        save_interval=100,
        model_path="models/dqn_worm_robot.pth"
    ):
        """
        Args:
            agent: DQN 에이전트
            create_system_fn: WormRobotSystem을 생성하는 함수
            num_episodes: 학습 에피소드 수
            termination_time: 시뮬레이션 최대 시간 (초)
            batch_size: 배치 크기
            buffer_size: Replay Buffer 크기
            log_interval: 로그 출력 간격
            save_interval: 모델 저장 간격
            model_path: 모델 저장 경로
        """
        self.agent = agent
        self.create_system_fn = create_system_fn
        self.num_episodes = num_episodes
        self.termination_time = termination_time
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.model_path = model_path
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # 통계
        self.stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_losses": [],
            "success_count": 0,
            "fail_count": 0
        }

    def train(self):
        """학습 루프 실행"""
        print("=" * 60)
        print("DQN 학습 시작")
        print("=" * 60)
        print(f"에피소드 수: {self.num_episodes}")
        print(f"시뮬레이션 시간: {self.termination_time}초")
        print(f"배치 크기: {self.batch_size}")
        print(f"초기 Epsilon: {self.agent.epsilon:.3f}")
        print("=" * 60)
        
        for episode in range(self.num_episodes):
            episode_reward, episode_steps, episode_status = self._run_episode()
            
            # 통계 업데이트
            self.stats["episode_rewards"].append(episode_reward)
            self.stats["episode_steps"].append(episode_steps)
            
            if episode_status == STATUS_WIN:
                self.stats["success_count"] += 1
            elif episode_status == STATUS_FAIL:
                self.stats["fail_count"] += 1
            
            # 학습 (배치가 충분히 쌓이면)
            if len(self.replay_buffer) >= self.batch_size:
                total_loss = 0.0
                # 여러 번 학습
                for _ in range(5):
                    # ReplayBuffer.sample()은 (states, actions, rewards, next_states, dones) 반환
                    # agent.train()은 [(s,a,r,s',d), ...] 형태 기대
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    batch = list(zip(states, actions, rewards, next_states, dones))
                    loss = self.agent.train(batch)
                    total_loss += loss
                self.stats["episode_losses"].append(total_loss / 5)
            else:
                self.stats["episode_losses"].append(0.0)
            
            # Epsilon 감소
            self.agent.update_epsilon()
            
            # 로그 출력
            if (episode + 1) % self.log_interval == 0:
                recent = self.log_interval
                avg_reward = sum(self.stats["episode_rewards"][-recent:]) / recent
                avg_steps = sum(self.stats["episode_steps"][-recent:]) / recent
                avg_loss = sum(self.stats["episode_losses"][-recent:]) / recent
                
                print(
                    f"Ep {episode + 1:4d}/{self.num_episodes} | "
                    f"Reward: {avg_reward:6.1f} | "
                    f"Steps: {avg_steps:4.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"Win: {self.stats['success_count']:3d} | "
                    f"Fail: {self.stats['fail_count']:3d}"
                )
            
            # 모델 저장
            if (episode + 1) % self.save_interval == 0:
                self._save_model()
        
        # 최종 모델 저장
        self._save_model()
        
        print("\n" + "=" * 60)
        print("학습 완료!")
        print(f"총 성공: {self.stats['success_count']}")
        print(f"총 실패: {self.stats['fail_count']}")
        print(f"성공률: {self.stats['success_count'] / self.num_episodes * 100:.1f}%")
        print("=" * 60)
        
        return self.stats

    def _run_episode(self):
        """
        단일 에피소드 실행
        
        간단한 버전: 에피소드 전체 실행 후 최종 상태 기반 보상
        향후 개선: 스텝별 경험 수집
        """
        # 새로운 시스템 생성 (랜덤 초기화)
        system = self.create_system_fn(rl_agent=self.agent)
        
        # 초기 상태 수집
        initial_states = {}
        for rid in range(system.controller.num_robots):
            if rid in system.environment.state.robot_positions:
                obs = system.environment._generate_observations()[rid]
                state = system.controller._observation_to_state(obs)
                initial_states[rid] = state
        
        # 시뮬레이터 설정 및 실행
        sim = Simulator(system)
        sim.setClassicDEVS()
        sim.setTerminationTime(self.termination_time)
        sim.simulate()
        
        # 최종 상태 및 보상 수집
        final_status = system.environment.state.status
        step_count = system.environment.state.step_count
        
        # 각 로봇의 최종 상태
        final_states = {}
        for rid in range(system.controller.num_robots):
            if rid in system.environment.state.robot_positions:
                obs = system.environment._generate_observations()[rid]
                state = system.controller._observation_to_state(obs)
                final_states[rid] = state
        
        # 보상 계산
        total_reward = 0.0
        for rid in initial_states.keys():
            # 간단한 보상 설계
            if final_status == STATUS_WIN:
                reward = 100.0
            elif final_status == STATUS_FAIL:
                reward = -50.0
            else:
                # 목표까지 거리 기반
                if rid in system.environment.state.robot_positions:
                    pos = system.environment.state.robot_positions[rid]
                    tail = pos["tail"]
                    head = pos["head"]
                    goal_head = system.environment.robot_goals.get(rid, (0, 0))
                    
                    tail_dist = abs(tail[0]) + abs(tail[1])
                    head_dist = abs(head[0] - goal_head[0]) + abs(head[1] - goal_head[1])
                    total_dist = tail_dist + head_dist
                    
                    # 거리가 가까울수록 높은 보상
                    reward = -total_dist * 2.0
                else:
                    reward = -50.0
            
            total_reward += reward
            
            # 경험 저장 (초기 상태 → 최종 상태)
            if rid in initial_states and rid in final_states:
                # 행동은 간략화 (실제로는 각 로봇의 행동 기록 필요)
                action = 0  # 임시
                done = (final_status != STATUS_RUNNING)
                
                self.replay_buffer.add(
                    initial_states[rid],
                    action,
                    reward,
                    final_states[rid],
                    float(done)
                )
        
        return total_reward, step_count, final_status

    def _save_model(self):
        """모델 저장"""
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "models", exist_ok=True)
        self.agent.save(self.model_path)

    def evaluate(self, num_episodes=10):
        """학습된 에이전트 평가"""
        print("\n" + "=" * 60)
        print(f"평가 시작 ({num_episodes} 에피소드)")
        print("=" * 60)
        
        success_count = 0
        total_rewards = []
        total_steps = []
        
        # 원래 epsilon 저장
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # 평가 시에는 탐험 안 함
        
        for episode in range(num_episodes):
            reward, steps, status = self._run_episode()
            total_rewards.append(reward)
            total_steps.append(steps)
            
            if status == STATUS_WIN:
                success_count += 1
        
        # Epsilon 복원
        self.agent.epsilon = original_epsilon
        
        avg_reward = sum(total_rewards) / num_episodes
        avg_steps = sum(total_steps) / num_episodes
        success_rate = success_count / num_episodes * 100
        
        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 스텝: {avg_steps:.1f}")
        print(f"성공률: {success_rate:.1f}%")
        print("=" * 60)
        
        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps
        }
