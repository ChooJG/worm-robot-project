"""
초간단 Phase 1 학습 V2
- Prioritized Experience Replay 활성화
- 장애물 충돌 페널티 극대화
- 더 나은 탐험 전략
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.replay_buffer import PrioritizedReplayBuffer  # 변경!
from rl.demonstrations_extended import get_extended_demonstrations
from system import WormRobotSystem
from config import STATUS_WIN, STATUS_FAIL, STATUS_RUNNING
import numpy as np

def create_system_fn(rl_agent=None):
    """로봇 1개, 장애물 1개 시스템"""
    return WormRobotSystem(
        rl_agent=rl_agent, 
        num_robots=1,
        obstacles=[(0, 1)]  # 중앙 근처에 장애물 1개
    )

class ImprovedDQNTrainer:
    """
    개선된 DQN Trainer
    - Prioritized Experience Replay
    - 장애물 충돌 시 큰 페널티
    """
    
    def __init__(self, agent, create_system_fn, num_episodes=20000, 
                 termination_time=80, batch_size=128, buffer_size=100000,
                 log_interval=100, save_interval=1000, model_path="outputs/simple_phase1_v2.pth"):
        self.agent = agent
        self.create_system_fn = create_system_fn
        self.num_episodes = num_episodes
        self.termination_time = termination_time
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.model_path = model_path
        
        # Prioritized Replay Buffer!
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        
        # 통계
        self.stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_losses": [],
            "success_count": 0,
            "fail_count": 0
        }
    
    def train(self):
        """학습 루프"""
        print("학습 시작...")
        
        for episode in range(self.num_episodes):
            episode_reward, episode_steps, status = self._run_episode()
            
            # 통계 기록
            self.stats["episode_rewards"].append(episode_reward)
            self.stats["episode_steps"].append(episode_steps)
            
            if status == STATUS_WIN:
                self.stats["success_count"] += 1
            elif status == STATUS_FAIL:
                self.stats["fail_count"] += 1
            
            # 학습 (버퍼에 충분한 경험이 있을 때)
            if len(self.replay_buffer.buffer) >= self.batch_size:
                # sample()은 튜플 반환: (states, actions, rewards, next_states, dones)
                # agent.train()은 리스트 기대: [(s, a, r, ns, d), ...]
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                batch = list(zip(states, actions, rewards, next_states, dones))
                loss = self.agent.train(batch)
                self.stats["episode_losses"].append(loss)
            
            # Epsilon 감소
            self.agent.update_epsilon()
            
            # 로그 출력
            if (episode + 1) % self.log_interval == 0:
                recent = min(self.log_interval, len(self.stats["episode_rewards"]))
                avg_reward = sum(self.stats["episode_rewards"][-recent:]) / recent
                avg_steps = sum(self.stats["episode_steps"][-recent:]) / recent
                avg_loss = sum(self.stats["episode_losses"][-recent:]) / recent if self.stats["episode_losses"] else 0
                
                print(
                    f"Ep {episode + 1:5d}/{self.num_episodes} | "
                    f"Reward: {avg_reward:7.1f} | "
                    f"Steps: {avg_steps:4.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"Win: {self.stats['success_count']:4d} | "
                    f"Fail: {self.stats['fail_count']:4d}"
                )
            
            # 모델 저장
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(self.model_path)
        
        # 최종 저장
        self.agent.save(self.model_path)
        
        print("\n" + "=" * 60)
        print("학습 완료!")
        print(f"총 성공: {self.stats['success_count']}")
        print(f"총 실패: {self.stats['fail_count']}")
        print(f"성공률: {self.stats['success_count'] / self.num_episodes * 100:.1f}%")
        print("=" * 60)
        
        return self.stats
    
    def _run_episode(self):
        """에피소드 실행 (Step-by-Step)"""
        system = self.create_system_fn(rl_agent=self.agent)
        num_robots = len(system.robots)
        
        episode_reward = 0.0
        step_count = 0
        max_steps = self.termination_time
        
        # 에피소드 루프
        while not system.is_done() and step_count < max_steps:
            # 현재 상태
            current_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    current_states[rid] = state
            
            # 행동 선택
            actions = {}
            for rid in current_states.keys():
                action = self.agent.get_action(current_states[rid])
                actions[rid] = action
            
            # 스텝 실행
            observations, rewards, done, status = system.step(actions)
            
            # 다음 상태
            next_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    next_states[rid] = state
            
            # 경험 저장 (장애물 충돌 시 큰 페널티!)
            step_reward = 0.0
            for rid in current_states.keys():
                if rid in next_states and rid in rewards:
                    robot_reward = rewards[rid]
                    
                    # 실패 판정
                    if done and status == STATUS_FAIL:
                        # 장애물 충돌 페널티 극대화!!!
                        robot_reward -= 500.0  # 50.0 → 500.0
                    elif done and status == STATUS_WIN:
                        # 성공 보너스
                        robot_reward += 300.0
                    
                    step_reward += robot_reward
                    
                    # Prioritized Replay Buffer에 추가
                    self.replay_buffer.add(
                        current_states[rid],
                        actions[rid],
                        robot_reward,
                        next_states[rid],
                        float(done)
                    )
            
            episode_reward += step_reward
            step_count += 1
            
            if done:
                break
        
        avg_reward = episode_reward / num_robots if num_robots > 0 else 0.0
        final_status = system.get_status()
        
        return avg_reward, step_count, final_status
    
    def evaluate(self, num_episodes=50):
        """평가"""
        print("\n" + "=" * 60)
        print(f"평가 시작 ({num_episodes} 에피소드)")
        print("=" * 60)
        
        success_count = 0
        fail_count = 0
        total_rewards = []
        total_steps = []
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # 평가 시에는 탐험 없음
        
        for _ in range(num_episodes):
            reward, steps, status = self._run_episode()
            total_rewards.append(reward)
            total_steps.append(steps)
            
            if status == STATUS_WIN:
                success_count += 1
            elif status == STATUS_FAIL:
                fail_count += 1
        
        self.agent.epsilon = original_epsilon
        
        avg_reward = sum(total_rewards) / num_episodes
        avg_steps = sum(total_steps) / num_episodes
        win_rate = success_count / num_episodes
        
        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 스텝: {avg_steps:.1f}")
        print(f"승률: {win_rate * 100:.1f}%")
        print("=" * 60)
        
        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "win_rate": win_rate
        }

def main():
    print("\n" + "=" * 70)
    print("🎯 초간단 Phase 1 학습 V2 (개선판)")
    print("=" * 70)
    print("개선 사항:")
    print("  1. Prioritized Experience Replay 활성화")
    print("  2. 장애물 충돌 페널티 10배 증가 (-50 → -500)")
    print("  3. 성공 보상 강화")
    print("=" * 70)
    
    # DQN 에이전트
    agent = DQNAgent(
        state_dim=13,
        action_dim=4,  # 3 → 4 (STAY 추가)
        learning_rate=0.0005,     # 0.001 → 0.0005 (더 안정적)
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,           # 0.05 → 0.1 (더 많은 탐험)
        epsilon_decay=0.9995,      # 0.9997 → 0.9995 (조금 빠른 감소)
        use_target_net=True,
        device="cpu"
    )
    
    # 트레이너
    trainer = ImprovedDQNTrainer(
        agent=agent,
        create_system_fn=create_system_fn,
        num_episodes=25000,        # 20000 → 25000
        termination_time=80,
        batch_size=128,
        buffer_size=100000,
        log_interval=100,
        save_interval=1000,
        model_path="outputs/simple_phase1_v2.pth"
    )
    
    # Happy Path 추가
    print(f"\n📖 Happy Path 추가 중...")
    demos = get_extended_demonstrations(num_robots=1, num_random=500)
    trainer.replay_buffer.add_demonstrations(demos)
    print(f"   ✅ 총 {len(demos)}개의 성공 경험 추가!")
    
    print(f"\n🚀 학습 시작!\n")
    
    try:
        stats = trainer.train()
        
        print(f"\n📊 최종 평가")
        eval_stats = trainer.evaluate(num_episodes=50)
        
        print(f"\n" + "=" * 70)
        print("✅ 학습 완료!")
        print("=" * 70)
        print(f"최종 승률:   {eval_stats['win_rate']*100:.1f}%")
        print(f"평균 보상:   {eval_stats['avg_reward']:.1f}")
        print(f"평균 스텝:   {eval_stats['avg_steps']:.1f}")
        print("=" * 70)
        
        if eval_stats['win_rate'] >= 0.3:
            print("\n🎉 성공! Phase 1을 충분히 학습했습니다!")
        elif eval_stats['win_rate'] >= 0.1:
            print("\n⚠️ 부분 성공. 더 학습이 필요합니다.")
        else:
            print("\n❌ 학습 실패. 설정을 재검토해야 합니다.")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 학습 중단됨!")
        agent.save(trainer.model_path)
    
    print(f"\n저장된 모델: {trainer.model_path}\n")

if __name__ == "__main__":
    main()

