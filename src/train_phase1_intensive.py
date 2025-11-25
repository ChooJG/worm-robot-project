"""
Phase 1 집중 학습 - 가장 간단한 케이스부터 확실히 해결
로봇 1개, 장애물 없음, 대폭 간소화
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.trainer import DQNTrainer
from rl.demonstrations_extended import get_extended_demonstrations
from system import WormRobotSystem

def create_system_fn(rl_agent=None):
    """1개 로봇 시스템 생성"""
    return WormRobotSystem(
        rl_agent=rl_agent, 
        num_robots=1,
        obstacles=None
    )

def main():
    print("\n" + "=" * 70)
    print("🎯 Phase 1 집중 학습 - 초간소화 버전")
    print("=" * 70)
    print("전략: 로봇 1개만, 대량 에피소드, 강력한 보상")
    print("=" * 70)
    
    # 하이퍼파라미터 (극단적 단순화)
    STATE_DIM = 13
    ACTION_DIM = 4  # 3 → 4 (STAY 추가)
    
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,      # 더 빠른 학습
        gamma=0.95,               # 단기 보상에 집중
        epsilon_start=1.0,
        epsilon_end=0.05,         # 더 많은 활용
        epsilon_decay=0.9998,     # 천천히 감소
        use_target_net=True,      # 안정화를 위해 활성화!
        target_update_freq=100,   # 자주 업데이트
        device="cpu"
    )
    
    # 트레이너 생성
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system_fn,
        num_episodes=30000,       # 10배 증가!
        termination_time=80,      # 적당히 줄임
        batch_size=128,           # 더 큰 배치
        buffer_size=100000,       # 대용량 버퍼
        log_interval=100,
        save_interval=1000,
        model_path="outputs/phase1_intensive.pth",
        use_tensorboard=True
    )
    
    # Happy Path 대량 추가
    print(f"\n📖 Happy Path 대량 추가 중...")
    demos = get_extended_demonstrations(num_robots=1, num_random=500)  # 5배 증가!
    trainer.replay_buffer.add_demonstrations(demos)
    print(f"   Demo 비율: {trainer.replay_buffer.get_demo_ratio()*100:.1f}%")
    print(f"   총 {len(demos)}개의 성공 경험!")
    
    # 학습 시작
    print(f"\n🚀 Phase 1 집중 학습 시작!\n")
    try:
        stats = trainer.train()
        
        # 평가
        print(f"\n📊 평가 중...")
        eval_stats = trainer.evaluate(num_episodes=20)
        
        print(f"\n✅ 학습 완료!")
        print(f"   최종 승률: {eval_stats['win_rate']*100:.1f}%")
        print(f"   평균 보상: {eval_stats['avg_reward']:.1f}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 학습 중단됨!")
        trainer._save_model()
    
    print("\n" + "=" * 70)
    print("모델 저장: outputs/phase1_intensive.pth")
    print("=" * 70)

if __name__ == "__main__":
    main()

