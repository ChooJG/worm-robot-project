"""
Worm Robot Simulation - 학습된 모델 평가
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.trainer import DQNTrainer
from system import WormRobotSystem


def create_system(rl_agent=None, num_robots=1, obstacles=None):
    """
    시스템 생성 함수
    
    Args:
        rl_agent: RL 에이전트
        num_robots: 로봇 수
        obstacles: 장애물 리스트
    
    Returns:
        WormRobotSystem 인스턴스
    """
    return WormRobotSystem(rl_agent=rl_agent, num_robots=num_robots, obstacles=obstacles)


def evaluate_model(model_path, num_episodes=20, num_robots=1, obstacles=None, verbose=True):
    """
    학습된 모델 평가
    
    Args:
        model_path: 모델 파일 경로
        num_episodes: 평가 에피소드 수
        num_robots: 로봇 수
        obstacles: 장애물 리스트
        verbose: 상세 출력 여부
    """
    print("=" * 70)
    print("📊 학습된 모델 평가")
    print("=" * 70)
    print(f"모델: {model_path}")
    print(f"평가 에피소드: {num_episodes}개")
    print(f"로봇 수: {num_robots}개")
    if obstacles:
        print(f"장애물: {obstacles}")
    print("=" * 70)
    
    # 에이전트 생성 및 로드
    agent = DQNAgent(
        state_dim=13,
        action_dim=4,  # 3 → 4 (STAY 추가)
        learning_rate=0.0005,
        gamma=0.99,
        device="cpu"
    )
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    agent.load(model_path)
    agent.epsilon = 0.0  # 평가 모드 (탐험 안 함)
    
    # 트레이너로 평가 실행
    # 시스템 생성 함수 (num_robots, obstacles 포함)
    def create_system_fn(rl_agent=None):
        return create_system(rl_agent=rl_agent, num_robots=num_robots, obstacles=obstacles)
    
    trainer = DQNTrainer(
        agent=agent,
        create_system_fn=create_system_fn,
        num_episodes=1,  # 평가만 할 것이므로
        termination_time=200,
        batch_size=32,
        buffer_size=1000
    )
    
    # 평가
    print("\n평가 진행 중...\n")
    stats = trainer.evaluate(num_episodes=num_episodes, verbose=verbose)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📈 평가 결과")
    print("=" * 70)
    print(f"총 에피소드:     {stats['total_episodes']}개")
    print(f"성공:           {stats['wins']}회")
    print(f"실패:           {stats['fails']}회")
    print(f"승률:           {stats['win_rate']*100:.1f}%")
    print(f"평균 보상:       {stats['avg_reward']:.2f}")
    print(f"평균 스텝:       {stats['avg_steps']:.1f}")
    print("=" * 70)
    
    if stats['win_rate'] > 0:
        print("\n✅ 모델이 성공적으로 학습되었습니다!")
    elif stats['avg_reward'] > 0:
        print("\n⚠️ 아직 성공하지 못했지만, 학습이 진행 중입니다.")
    else:
        print("\n❌ 학습이 충분하지 않습니다. 더 많은 에피소드가 필요합니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="학습된 DQN 모델 평가")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/curriculum_phase3_4robots.pth",
        help="평가할 모델 파일 경로"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="평가 에피소드 수"
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=1,
        help="로봇 수"
    )
    parser.add_argument(
        "--obstacles",
        type=str,
        default=None,
        help="장애물 위치 (예: '(0,1),(-1,-1)')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    # 장애물 파싱
    obstacles = None
    if args.obstacles:
        try:
            obstacles = eval(f"[{args.obstacles}]")
        except:
            print(f"⚠️ 장애물 파싱 실패: {args.obstacles}")
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        num_robots=args.num_robots,
        obstacles=obstacles,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

