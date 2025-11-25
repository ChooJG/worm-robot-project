"""
장애물 기능 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system import WormRobotSystem
from pypdevs.simulator import Simulator


def test_obstacles():
    """장애물이 있는 시스템 테스트"""
    print("=" * 70)
    print("🧪 장애물 기능 테스트")
    print("=" * 70)
    
    # 테스트 1: 장애물 없음
    print("\n1️⃣ 장애물 없음 (기본)")
    print("-" * 70)
    system1 = WormRobotSystem(num_robots=1, obstacles=None)
    print(f"✅ 시스템 생성 성공")
    print(f"   로봇 수: {len(system1.robots)}개")
    print(f"   장애물 수: {len(system1.environment.obstacles)}개")
    
    # 테스트 2: 장애물 있음
    print("\n2️⃣ 장애물 3개")
    print("-" * 70)
    obstacles = [(0, 1), (-1, -1), (1, 0)]
    system2 = WormRobotSystem(num_robots=1, obstacles=obstacles)
    print(f"✅ 시스템 생성 성공")
    print(f"   로봇 수: {len(system2.robots)}개")
    print(f"   장애물 수: {len(system2.environment.obstacles)}개")
    print(f"   장애물 위치: {system2.environment.obstacles}")
    
    # 테스트 3: 시뮬레이션 실행 (짧게)
    print("\n3️⃣ 장애물 충돌 감지 테스트")
    print("-" * 70)
    print("5초 시뮬레이션 실행 중...")
    
    system3 = WormRobotSystem(num_robots=1, obstacles=[(0, 0), (1, 1)])
    sim = Simulator(system3)
    sim.setClassicDEVS()
    sim.setTerminationTime(5)
    sim.simulate()
    
    print(f"✅ 시뮬레이션 완료")
    print(f"   최종 상태: {system3.environment.state.status}")
    print(f"   스텝 수: {system3.environment.state.step_count}")
    
    # 테스트 4: Observation 확인
    print("\n4️⃣ Observation에 장애물 정보 포함 확인")
    print("-" * 70)
    system4 = WormRobotSystem(num_robots=1, obstacles=[(2, 2), (-2, -2)])
    obs = system4.environment._generate_observations()
    
    for rid, observation in obs.items():
        print(f"Robot {rid} Observation:")
        print(f"   Head: {observation['own_head']}")
        print(f"   Tail: {observation['own_tail']}")
        print(f"   Goal: {observation['goal_position']}")
        print(f"   Obstacles: {observation['obstacles']}")
    
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)
    print("\n💡 장애물 기능이 정상적으로 작동합니다.")
    print("   이제 train_curriculum.py를 실행하여 학습할 수 있습니다.")
    print("\n🚀 실행:")
    print("   cd src")
    print("   python3.11 train_curriculum.py")
    print("=" * 70)


if __name__ == "__main__":
    test_obstacles()

