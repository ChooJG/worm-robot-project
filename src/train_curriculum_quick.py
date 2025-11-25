"""
Worm Robot Simulation - Curriculum Learning (빠른 테스트 버전)
점진적 난이도 증가 학습 - 적은 에피소드로 빠르게 테스트

⚠️ 주의: 이 버전은 빠른 테스트용입니다. 
실제 학습은 train_curriculum.py를 사용하세요.
"""

import sys
import os
import config

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.trainer import DQNTrainer
from rl.demonstrations import get_all_demonstrations
from rl.demonstrations_extended import get_extended_demonstrations
from system import WormRobotSystem


def create_system_with_num_robots(num_robots, obstacles=None):
    """
    지정된 로봇 수로 시스템 생성 함수를 반환
    
    Args:
        num_robots: 로봇 수
        obstacles: 장애물 위치 리스트 [(x, y), ...] (선택)
    
    Returns:
        함수: create_system_fn
    """
    def create_system_fn(rl_agent=None):
        # WormRobotSystem에 num_robots와 obstacles 전달
        system = WormRobotSystem(
            rl_agent=rl_agent, 
            num_robots=num_robots,
            obstacles=obstacles
        )
        return system
    
    return create_system_fn


def train_phase(
    phase_num,
    num_robots,
    num_episodes,
    prev_model_path=None,
    use_demonstrations=True,
    termination_time=100,
    obstacles=None
):
    """
    단일 Phase 학습 실행 (빠른 버전)
    
    Args:
        phase_num: Phase 번호 (1, 1.5, 2, 2.5, 3, ...)
        num_robots: 이 Phase의 로봇 수
        num_episodes: 학습 에피소드 수
        prev_model_path: 이전 Phase 모델 경로 (파인튜닝용)
        use_demonstrations: Happy Path 사용 여부
        termination_time: 에피소드당 최대 시간
        obstacles: 장애물 위치 리스트 [(x, y), ...] (선택)
    
    Returns:
        str: 저장된 모델 경로
    """
    print("\n" + "=" * 70)
    print(f"📚 Curriculum Learning (Quick Test) - Phase {phase_num}")
    print("=" * 70)
    print(f"로봇 수: {num_robots}개")
    print(f"장애물 수: {len(obstacles) if obstacles else 0}개")
    if obstacles:
        print(f"장애물 위치: {obstacles}")
    print(f"에피소드: {num_episodes}개 (빠른 테스트)")
    print(f"이전 모델: {prev_model_path if prev_model_path else '없음 (처음부터)'}")
    print(f"Demonstrations: {'사용' if use_demonstrations else '미사용'}")
    print("=" * 70)
    
    # 하이퍼파라미터
    STATE_DIM = 13
    ACTION_DIM = 4  # 3 → 4 (STAY 추가)
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,     # 빠른 테스트: 더 빠른 감소
        use_target_net=False,
        device="cpu"
    )
    
    # 이전 Phase 모델 로드 (파인튜닝!)
    if prev_model_path and os.path.exists(prev_model_path):
        print(f"\n✅ 이전 Phase 모델 로드 중...")
        agent.load(prev_model_path)
        
        # Epsilon 조정: 새로운 상황 탐험 필요
        if phase_num == 2:
            agent.epsilon = 0.7  # Phase 2: 중간 탐험
        elif phase_num == 3:
            agent.epsilon = 0.5  # Phase 3: 적당한 탐험
        else:
            agent.epsilon = 0.8  # 기타: 높은 탐험
        
        print(f"   Epsilon 조정: {agent.epsilon:.2f} (새 상황 탐험)")
    
    # 트레이너 생성
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system_with_num_robots(num_robots, obstacles=obstacles),
        num_episodes=num_episodes,
        termination_time=termination_time,
        batch_size=32,
        buffer_size=10000,
        log_interval=100,        # 100 에피소드마다 로그
        save_interval=500,       # 500 에피소드마다 저장
        model_path=f"outputs/quick_phase{phase_num}_{num_robots}robots.pth",
        use_tensorboard=False    # 빠른 테스트: TensorBoard 비활성화
    )
    
    # Happy Path (Demonstrations) 추가 - 확장 버전!
    if use_demonstrations:
        print(f"\n📖 Happy Path (Extended Demonstrations) 추가 중...")
        # 로봇 수에 따라 더 많은 데모 생성
        num_random_demos = {1: 50, 2: 80, 4: 100}.get(num_robots, 50)
        demos = get_extended_demonstrations(num_robots=num_robots, num_random=num_random_demos)
        trainer.replay_buffer.add_demonstrations(demos)
        print(f"   현재 Demo 비율: {trainer.replay_buffer.get_demo_ratio()*100:.1f}%")
        print(f"   총 {len(demos)}개의 성공 경험 추가!")
    
    # 학습 실행
    print(f"\n🚀 Phase {phase_num} 학습 시작!\n")
    try:
        stats = trainer.train()
        
        # 간단한 평가
        print(f"\n📊 Phase {phase_num} 평가 중...")
        eval_stats = trainer.evaluate(num_episodes=10)
        
        print(f"\n✅ Phase {phase_num} 완료!")
        print(f"   최종 승률: {eval_stats['win_rate']*100:.1f}%")
        print(f"   평균 보상: {eval_stats['avg_reward']:.1f}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Phase {phase_num} 학습 중단됨!")
        trainer._save_model()
    
    # 모델 경로 반환
    return trainer.model_path


def main():
    """메인 함수: Curriculum Learning 빠른 테스트"""
    print("\n" + "=" * 70)
    print("🎓 Curriculum Learning - 빠른 테스트 버전")
    print("=" * 70)
    print("\n⚠️ 주의: 이 버전은 빠른 테스트용입니다.")
    print("   실제 학습은 train_curriculum.py를 사용하세요.")
    print("\n전략:")
    print("  Phase 1 (쉬움):    로봇 1개, 장애물 없음 → 500 에피소드")
    print("  Phase 1.5 (쉬움+): 로봇 1개, 장애물 2개 → 800 에피소드")
    print("  Phase 2 (중간):    로봇 2개, 장애물 없음 → 1,500 에피소드")
    print("  Phase 2.5 (중간+): 로봇 2개, 장애물 2개 → 2,000 에피소드")
    print("  Phase 3 (어려움):  로봇 4개, 장애물 없음 → 3,000 에피소드")
    print("\n예상 소요 시간: 약 1~1.5시간")
    print("=" * 70)
    
    # outputs 디렉토리 생성
    os.makedirs("outputs", exist_ok=True)
    
    # Phase 1: 로봇 1개, 장애물 없음 (기본 학습)
    phase1_model = train_phase(
        phase_num=1,
        num_robots=1,
        num_episodes=500,       # 빠른 테스트
        prev_model_path=None,
        use_demonstrations=True,
        termination_time=80,
        obstacles=None
    )
    
    # Phase 1.5: 로봇 1개, 장애물 있음 (장애물 회피)
    obstacles_phase15 = [(0, 1), (-1, -1)]  # 장애물 2개
    phase15_model = train_phase(
        phase_num=1.5,
        num_robots=1,
        num_episodes=800,       # 장애물 회피 연습
        prev_model_path=phase1_model,
        use_demonstrations=True,
        termination_time=80,
        obstacles=obstacles_phase15
    )
    
    # Phase 2: 로봇 2개, 장애물 없음 (로봇 간 협력)
    phase2_model = train_phase(
        phase_num=2,
        num_robots=2,
        num_episodes=1500,      # 2개 로봇 협력
        prev_model_path=phase15_model,
        use_demonstrations=True,
        termination_time=100,
        obstacles=None
    )
    
    # Phase 2.5: 로봇 2개, 장애물 있음 (복합 회피)
    obstacles_phase25 = [(0, 2), (-2, 0)]  # 장애물 2개
    phase25_model = train_phase(
        phase_num=2.5,
        num_robots=2,
        num_episodes=2000,      # 복합 회피 연습
        prev_model_path=phase2_model,
        use_demonstrations=True,
        termination_time=100,
        obstacles=obstacles_phase25
    )
    
    # Phase 3: 로봇 4개, 장애물 없음 (최종 목표)
    phase3_model = train_phase(
        phase_num=3,
        num_robots=4,
        num_episodes=3000,      # 멀티 로봇 협력
        prev_model_path=phase25_model,
        use_demonstrations=True,
        termination_time=120,
        obstacles=None
    )
    
    # 완료
    print("\n" + "=" * 70)
    print("🎉 빠른 테스트 완료!")
    print("=" * 70)
    print(f"\n저장된 모델:")
    print(f"  Phase 1   (1 robot, 장애물 없음):  {phase1_model}")
    print(f"  Phase 1.5 (1 robot, 장애물 2개):  {phase15_model}")
    print(f"  Phase 2   (2 robots, 장애물 없음): {phase2_model}")
    print(f"  Phase 2.5 (2 robots, 장애물 2개): {phase25_model}")
    print(f"  Phase 3   (4 robots, 장애물 없음): {phase3_model}")
    print("\n⚠️ 이 모델들은 테스트용입니다. 성능이 낮을 수 있습니다.")
    print("\n💡 실제 학습을 위해:")
    print("   python3.11 train_curriculum.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

