"""
Demonstration 생성 및 테스트
Happy Path가 올바르게 생성되는지 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.demonstrations import (
    create_simple_demonstration_1robot,
    create_demonstration_with_rotation,
    create_demonstration_avoid_collision,
    get_all_demonstrations
)
from rl.replay_buffer import ReplayBuffer


def test_demonstrations():
    """데모 생성 테스트"""
    print("=" * 70)
    print("🎓 Demonstration 생성 테스트")
    print("=" * 70)
    
    # 1. 간단한 1 로봇 데모
    print("\n1️⃣ 간단한 1 로봇 데모")
    print("-" * 70)
    demo_1robot = create_simple_demonstration_1robot()
    print(f"생성된 경험 수: {len(demo_1robot)}개")
    for i, (state, action, reward, next_state, done) in enumerate(demo_1robot):
        print(f"  Step {i+1}:")
        print(f"    Action: {action} (0=전진, 1=시계, 2=반시계)")
        print(f"    Reward: {reward:.1f}")
        print(f"    Done: {done}")
        print(f"    State shape: {state.shape}")
    
    # 2. 회전 포함 데모
    print("\n2️⃣ 회전 포함 데모")
    print("-" * 70)
    demo_rotation = create_demonstration_with_rotation()
    print(f"생성된 경험 수: {len(demo_rotation)}개")
    for i, (state, action, reward, next_state, done) in enumerate(demo_rotation):
        print(f"  Step {i+1}:")
        print(f"    Action: {action}")
        print(f"    Reward: {reward:.1f}")
    
    # 3. 충돌 회피 데모
    print("\n3️⃣ 충돌 회피 데모")
    print("-" * 70)
    demo_collision = create_demonstration_avoid_collision()
    print(f"생성된 경험 수: {len(demo_collision)}개")
    for i, (state, action, reward, next_state, done) in enumerate(demo_collision):
        print(f"  Step {i+1}:")
        print(f"    Action: {action}")
        print(f"    Reward: {reward:.1f}")
        print(f"    Nearby robots: {state[11]*3:.0f}개")  # num_nearby
    
    # 4. 전체 데모 (로봇 수별)
    print("\n4️⃣ 전체 데모 (로봇 수별)")
    print("-" * 70)
    for num_robots in [1, 2, 4]:
        demos = get_all_demonstrations(num_robots=num_robots)
        print(f"  로봇 {num_robots}개: {len(demos)}개 경험")
    
    return demo_1robot, demo_rotation, demo_collision


def test_replay_buffer_with_demos():
    """Replay Buffer에 데모 추가 테스트"""
    print("\n" + "=" * 70)
    print("📦 Replay Buffer + Demonstrations 테스트")
    print("=" * 70)
    
    # Replay Buffer 생성
    buffer = ReplayBuffer(capacity=1000)
    
    print(f"\n초기 버퍼 크기: {len(buffer)}")
    print(f"초기 Demo 비율: {buffer.get_demo_ratio()*100:.1f}%")
    
    # Demonstration 추가
    demos = get_all_demonstrations(num_robots=2)
    buffer.add_demonstrations(demos)
    
    print(f"\nDemo 추가 후 버퍼 크기: {len(buffer)}")
    print(f"Demo 비율: {buffer.get_demo_ratio()*100:.1f}%")
    
    # 일반 경험 추가
    import numpy as np
    for i in range(10):
        state = np.random.randn(13).astype(np.float32)
        action = i % 3
        reward = -10.0  # 낮은 보상 (실패)
        next_state = np.random.randn(13).astype(np.float32)
        done = False
        buffer.add(state, action, reward, next_state, done)
    
    print(f"\n일반 경험 추가 후 버퍼 크기: {len(buffer)}")
    print(f"Demo 비율: {buffer.get_demo_ratio()*100:.1f}%")
    
    # 샘플링 테스트
    if len(buffer) >= 8:
        print("\n샘플링 테스트 (8개)")
        states, actions, rewards, next_states, dones = buffer.sample(8)
        print(f"  Rewards: {[f'{r:.1f}' for r in rewards]}")
        print(f"  Actions: {actions}")
        print(f"  높은 보상(demo) 비율: {sum(1 for r in rewards if r >= 50) / len(rewards) * 100:.1f}%")


def main():
    """메인 함수"""
    print("\n🧪 Demonstration 시스템 테스트\n")
    
    # 1. Demonstration 생성 테스트
    demo_1, demo_2, demo_3 = test_demonstrations()
    
    # 2. Replay Buffer 테스트
    test_replay_buffer_with_demos()
    
    # 요약
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)
    print("\n💡 Demonstration은 Replay Buffer에 추가되어")
    print("   학습 초기부터 '성공 경험'을 제공합니다.")
    print("\n📚 Curriculum Learning과 함께 사용하면:")
    print("   - Phase 1: 로봇 1개 + Demos → 빠른 학습")
    print("   - Phase 2: 로봇 2개 + Phase 1 모델 + Demos")
    print("   - Phase 3: 로봇 4개 + Phase 2 모델 + Demos")
    print("\n🚀 실행:")
    print("   cd src")
    print("   python3.11 train_curriculum.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

