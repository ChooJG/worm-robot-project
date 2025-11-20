"""
Worm Robot Simulation - DQN 학습 실행
"""

import sys
import os

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.trainer import DQNTrainer
from system import WormRobotSystem


def create_system(rl_agent=None):
    """
    WormRobotSystem 생성 함수
    
    Args:
        rl_agent: RL 에이전트 (None이면 휴리스틱)
    
    Returns:
        WormRobotSystem 인스턴스
    """
    # 시스템 생성 (내부에서 자동으로 랜덤 초기 위치 생성)
    system = WormRobotSystem(rl_agent=rl_agent)
    
    return system


def main():
    """메인 함수"""
    print("=" * 60)
    print("Worm Robot DQN 학습")
    print("=" * 60)
    
    # 하이퍼파라미터
    STATE_DIM = 13  # controller._observation_to_state에서 정의한 차원
    ACTION_DIM = 3  # 전진, 시계방향, 반시계방향
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        use_target_net=False,  # 간단하게: Target Network 사용 안 함
        device="cpu"  # GPU 사용 시 "cuda"
    )
    
    # 트레이너 생성
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system,
        num_episodes=100,         # 테스트용으로 100 에피소드
        termination_time=200,     # 시뮬레이션 최대 시간 (초)
        batch_size=32,
        buffer_size=10000,
        log_interval=10,
        save_interval=50,
        model_path="outputs/dqn_worm_robot.pth"
    )
    
    # 학습 실행
    try:
        stats = trainer.train()
        
        # 평가
        print("\n학습된 에이전트 평가...")
        eval_stats = trainer.evaluate(num_episodes=10)
        
    except KeyboardInterrupt:
        print("\n\n학습 중단됨!")
        trainer._save_model()
    
    print("\n완료!")


if __name__ == "__main__":
    main()

