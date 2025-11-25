"""
Worm Robot Simulation - Curriculum Learning
점진적 난이도 증가 학습

해결하는 문제:
1. Sparse Reward: 로봇 4개 동시 성공이 너무 어려움
2. 해결책: 1개 → 2개 → 4개 순서로 점진적 학습

각 Phase:
- Phase 1: 로봇 1개 → 기본 행동 학습 (목표 찾아가기)
- Phase 2: 로봇 2개 → 충돌 회피 학습
- Phase 3: 로봇 4개 → 멀티 로봇 협력
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
    단일 Phase 학습 실행
    
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
    print(f"📚 Curriculum Learning - Phase {phase_num}")
    print("=" * 70)
    print(f"로봇 수: {num_robots}개")
    print(f"장애물 수: {len(obstacles) if obstacles else 0}개")
    if obstacles:
        print(f"장애물 위치: {obstacles}")
    print(f"에피소드: {num_episodes}개")
    print(f"이전 모델: {prev_model_path if prev_model_path else '없음 (처음부터)'}")
    print(f"Demonstrations: {'사용' if use_demonstrations else '미사용'}")
    print("=" * 70)
    
    # 하이퍼파라미터 (Step-by-Step 최적화)
    STATE_DIM = 13
    ACTION_DIM = 4  # 3 → 4 (STAY 추가)
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.0003,    # Step-by-step은 업데이트가 많으므로 낮춤
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,    # Step이 많아지므로 decay 빠르게
        use_target_net=False,
        device="cpu"
    )
    
    # 이전 Phase 모델 로드 (파인튜닝!)
    if prev_model_path and os.path.exists(prev_model_path):
        print(f"\n✅ 이전 Phase 모델 로드 중...")
        agent.load(prev_model_path)
        
        # Epsilon 조정: 새로운 상황 탐험 필요
        if phase_num == 2:
            agent.epsilon = 0.6  # Phase 2: 적절한 탐험
        elif phase_num == 2.5:
            agent.epsilon = 0.5  # Phase 2.5: 중간 탐험
        elif phase_num == 3:
            agent.epsilon = 0.4  # Phase 3: 제한적 탐험
        else:
            agent.epsilon = 0.7  # Phase 1.5: 높은 탐험
        
        print(f"   Epsilon 조정: {agent.epsilon:.2f} (새 상황 탐험)")
    
    # 트레이너 생성
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system_with_num_robots(num_robots, obstacles=obstacles),
        num_episodes=num_episodes,
        termination_time=termination_time,
        batch_size=64,            # Step-by-step은 경험이 많으므로 batch size 증가
        buffer_size=50000,        # Buffer 크기 대폭 증가 (step-by-step)
        log_interval=10,
        save_interval=50,
        model_path=f"outputs/curriculum_phase{phase_num}_{num_robots}robots.pth"
    )
    
    # Happy Path (Demonstrations) 추가 - 확장 버전!
    if use_demonstrations:
        print(f"\n📖 Happy Path (Extended Demonstrations) 추가 중...")
        # 로봇 수에 따라 더 많은 데모 생성
        num_random_demos = {1: 100, 2: 150, 4: 200}.get(num_robots, 100)
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
    """메인 함수: Curriculum Learning 실행"""
    print("\n" + "=" * 70)
    print("🎓 Curriculum Learning - Worm Robot 협력 학습")
    print("=" * 70)
    print("\n전략:")
    print("  Phase 1 (쉬움):    로봇 1개, 장애물 없음 → 기본 행동 학습")
    print("  Phase 1.5 (쉬움+): 로봇 1개, 장애물 2-3개 → 장애물 회피 학습")
    print("  Phase 2 (중간):    로봇 2개, 장애물 없음 → 로봇 간 협력")
    print("  Phase 2.5 (중간+): 로봇 2개, 장애물 2-3개 → 복합 회피")
    print("  Phase 3 (어려움):  로봇 4개, 장애물 없음 → 멀티 로봇 협력")
    print("\n각 Phase는 이전 Phase의 학습된 모델을 이어받아 파인튜닝합니다.")
    print("=" * 70)
    
    # outputs 디렉토리 생성
    os.makedirs("outputs", exist_ok=True)
    
    # Phase 1: 로봇 1개, 장애물 없음 (기본 학습)
    phase1_model = train_phase(
        phase_num=1,
        num_robots=1,
        num_episodes=3000,      # Step-by-step: 경험 수집이 많아서 에피소드 줄임
        prev_model_path=None,   # 처음부터
        use_demonstrations=True,
        termination_time=100,   # 스텝 수 (적당하게 설정)
        obstacles=None          # 장애물 없음
    )
    
    # Phase 1.5: 로봇 1개, 장애물 있음 (장애물 회피 학습)
    obstacles_phase15 = [(0, 1), (-1, -1), (1, 0)]  # 장애물 3개
    phase15_model = train_phase(
        phase_num=1.5,
        num_robots=1,
        num_episodes=4000,      # 장애물 회피 연습
        prev_model_path=phase1_model,  # ⬅️ Phase 1 모델 파인튜닝!
        use_demonstrations=True,
        termination_time=100,
        obstacles=obstacles_phase15
    )
    
    # Phase 2: 로봇 2개, 장애물 없음 (로봇 간 협력)
    phase2_model = train_phase(
        phase_num=2,
        num_robots=2,
        num_episodes=5000,      # 2개 로봇 협력
        prev_model_path=phase15_model,  # ⬅️ Phase 1.5 모델 파인튜닝!
        use_demonstrations=True,
        termination_time=120,
        obstacles=None
    )
    
    # Phase 2.5: 로봇 2개, 장애물 있음 (복합 회피)
    obstacles_phase25 = [(0, 2), (-2, 0), (1, -1)]  # 장애물 3개
    phase25_model = train_phase(
        phase_num=2.5,
        num_robots=2,
        num_episodes=6000,      # 복합 회피 연습
        prev_model_path=phase2_model,  # ⬅️ Phase 2 모델 파인튜닝!
        use_demonstrations=True,
        termination_time=120,
        obstacles=obstacles_phase25
    )
    
    # Phase 3: 로봇 4개, 장애물 없음 (최종 목표)
    phase3_model = train_phase(
        phase_num=3,
        num_robots=4,
        num_episodes=10000,     # 멀티 로봇 협력
        prev_model_path=phase25_model,  # ⬅️ Phase 2.5 모델 파인튜닝!
        use_demonstrations=True,
        termination_time=150    # 4개 로봇은 시간 더 필요
    )
    
    # 완료
    print("\n" + "=" * 70)
    print("🎉 Curriculum Learning 완료!")
    print("=" * 70)
    print(f"\n저장된 모델:")
    print(f"  Phase 1   (1 robot, 장애물 없음):  {phase1_model}")
    print(f"  Phase 1.5 (1 robot, 장애물 3개):  {phase15_model}")
    print(f"  Phase 2   (2 robots, 장애물 없음): {phase2_model}")
    print(f"  Phase 2.5 (2 robots, 장애물 3개): {phase25_model}")
    print(f"  Phase 3   (4 robots, 장애물 없음): {phase3_model}")
    print("\n최종 모델을 사용하여 시뮬레이션을 실행하세요:")
    print(f"  python3.11 evaluate.py --model {phase3_model}")
    print("=" * 70)


if __name__ == "__main__":
    main()

