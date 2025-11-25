"""
초간단 Phase 1 학습
- 로봇 1개
- 장애물 1개 (고정 위치)
- 랜덤 시작 위치 → 목적지 이동
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.trainer import DQNTrainer
from rl.demonstrations_extended import get_extended_demonstrations
from system import WormRobotSystem

def create_system_fn(rl_agent=None):
    """로봇 1개, 장애물 1개 시스템"""
    return WormRobotSystem(
        rl_agent=rl_agent, 
        num_robots=1,
        obstacles=[(0, 1)]  # 중앙 근처에 장애물 1개
    )

def main():
    print("\n" + "=" * 70)
    print("🎯 초간단 Phase 1 학습")
    print("=" * 70)
    print("목표: 로봇 1개가 장애물 1개를 피해 목적지에 도달")
    print("  - 로봇: 1개 (랜덤 시작 위치)")
    print("  - 장애물: 1개 (고정 위치: (0, 1))")
    print("  - 목적지: 랜덤 (매 에피소드마다 변경)")
    print("=" * 70)
    
    # 하이퍼파라미터 (간단한 문제에 최적화)
    STATE_DIM = 13
    ACTION_DIM = 4  # 3 → 4 (STAY 추가)
    
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,      # 적당한 학습률
        gamma=0.95,               # 단기 보상 중시
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9997,     # 느린 감소
        use_target_net=True,      # 안정화!
        device="cpu"
    )
    
    print("\n📐 DQN 설정:")
    print(f"  Learning Rate: 0.001")
    print(f"  Gamma: 0.95")
    print(f"  Epsilon: 1.0 → 0.05 (decay=0.9997)")
    print(f"  Target Network: 활성화")
    
    # 트레이너 생성
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system_fn,
        num_episodes=20000,       # 충분한 에피소드
        termination_time=80,      # 적당한 시간 제한
        batch_size=128,           # 큰 배치 크기
        buffer_size=100000,       # 대용량 버퍼
        log_interval=100,         # 100 에피소드마다 로그
        save_interval=1000,       # 1000 에피소드마다 저장
        model_path="outputs/simple_phase1.pth",
        use_tensorboard=True
    )
    
    print("\n📊 학습 설정:")
    print(f"  에피소드: 20,000개")
    print(f"  Termination Time: 80 스텝")
    print(f"  Batch Size: 128")
    print(f"  Replay Buffer: 100,000")
    
    # Happy Path 대량 추가
    print(f"\n📖 Happy Path (성공 경험) 추가 중...")
    demos = get_extended_demonstrations(num_robots=1, num_random=500)
    trainer.replay_buffer.add_demonstrations(demos)
    print(f"   ✅ Demo 비율: {trainer.replay_buffer.get_demo_ratio()*100:.1f}%")
    print(f"   ✅ 총 {len(demos)}개의 성공 경험 추가!")
    
    # 학습 시작
    print(f"\n" + "=" * 70)
    print("🚀 학습 시작!")
    print("=" * 70)
    print("💡 TensorBoard로 실시간 모니터링:")
    print("   tensorboard --logdir=runs")
    print("=" * 70 + "\n")
    
    try:
        stats = trainer.train()
        
        # 평가
        print(f"\n" + "=" * 70)
        print("📊 최종 평가")
        print("=" * 70)
        eval_stats = trainer.evaluate(num_episodes=50)  # 50회 평가
        
        print(f"\n" + "=" * 70)
        print("✅ 학습 완료!")
        print("=" * 70)
        print(f"최종 승률:   {eval_stats['win_rate']*100:.1f}%")
        print(f"평균 보상:   {eval_stats['avg_reward']:.1f}")
        print(f"평균 스텝:   {eval_stats['avg_steps']:.1f}")
        print("=" * 70)
        
        # 결과 판정
        if eval_stats['win_rate'] >= 0.3:  # 30% 이상
            print("\n🎉 성공! Phase 1을 충분히 학습했습니다!")
            print("   다음 단계로 진행할 수 있습니다.")
        elif eval_stats['win_rate'] >= 0.1:  # 10% 이상
            print("\n⚠️ 부분 성공. 더 학습이 필요합니다.")
            print("   에피소드 수를 늘려서 재학습을 권장합니다.")
        else:
            print("\n❌ 학습 실패. 설정을 재검토해야 합니다.")
            print("   보상 함수나 하이퍼파라미터를 조정하세요.")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 학습 중단됨!")
        trainer._save_model()
        print("모델이 저장되었습니다.")
    
    print(f"\n저장된 모델: outputs/simple_phase1.pth")
    print(f"평가 명령어: python3.11 evaluate.py --model outputs/simple_phase1.pth\n")

if __name__ == "__main__":
    main()

