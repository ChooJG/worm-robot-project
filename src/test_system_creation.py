"""
WormRobotSystem 생성 테스트
Curriculum Learning을 위한 동적 로봇 수 변경 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system import WormRobotSystem
import config


def test_system_creation():
    """다양한 로봇 수로 시스템 생성 테스트"""
    print("=" * 70)
    print("🧪 WormRobotSystem 동적 생성 테스트")
    print("=" * 70)
    
    # 테스트 1: 기본 (config.NUM_ROBOTS 사용)
    print(f"\n1️⃣ 기본 생성 (config.NUM_ROBOTS={config.NUM_ROBOTS})")
    print("-" * 70)
    try:
        system1 = WormRobotSystem()
        print(f"✅ 시스템 생성 성공")
        print(f"   로봇 수: {len(system1.robots)}개")
        print(f"   Environment 로봇 수: {system1.environment.num_robots}")
        print(f"   Controller 로봇 수: {system1.controller.num_robots}")
        assert len(system1.robots) == system1.environment.num_robots
        assert len(system1.robots) == system1.controller.num_robots
        print(f"✅ 로봇 수 일치 확인")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return False
    
    # 테스트 2: 로봇 1개
    print(f"\n2️⃣ 로봇 1개 생성 (Phase 1)")
    print("-" * 70)
    try:
        system2 = WormRobotSystem(num_robots=1)
        print(f"✅ 시스템 생성 성공")
        print(f"   로봇 수: {len(system2.robots)}개")
        print(f"   Environment 로봇 수: {system2.environment.num_robots}")
        print(f"   Controller 로봇 수: {system2.controller.num_robots}")
        assert len(system2.robots) == 1
        assert system2.environment.num_robots == 1
        assert system2.controller.num_robots == 1
        print(f"✅ 로봇 수 일치 확인")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return False
    
    # 테스트 3: 로봇 2개
    print(f"\n3️⃣ 로봇 2개 생성 (Phase 2)")
    print("-" * 70)
    try:
        system3 = WormRobotSystem(num_robots=2)
        print(f"✅ 시스템 생성 성공")
        print(f"   로봇 수: {len(system3.robots)}개")
        print(f"   Environment 로봇 수: {system3.environment.num_robots}")
        print(f"   Controller 로봇 수: {system3.controller.num_robots}")
        assert len(system3.robots) == 2
        assert system3.environment.num_robots == 2
        assert system3.controller.num_robots == 2
        print(f"✅ 로봇 수 일치 확인")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 테스트 4: 로봇 4개
    print(f"\n4️⃣ 로봇 4개 생성 (Phase 3)")
    print("-" * 70)
    try:
        system4 = WormRobotSystem(num_robots=4)
        print(f"✅ 시스템 생성 성공")
        print(f"   로봇 수: {len(system4.robots)}개")
        print(f"   Environment 로봇 수: {system4.environment.num_robots}")
        print(f"   Controller 로봇 수: {system4.controller.num_robots}")
        assert len(system4.robots) == 4
        assert system4.environment.num_robots == 4
        assert system4.controller.num_robots == 4
        print(f"✅ 로봇 수 일치 확인")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return False
    
    # 테스트 5: 연속 생성 (Phase 1 → 2 → 3 시뮬레이션)
    print(f"\n5️⃣ 연속 생성 테스트 (1 → 2 → 4)")
    print("-" * 70)
    try:
        for num in [1, 2, 4]:
            system = WormRobotSystem(num_robots=num)
            assert len(system.robots) == num
            print(f"✅ 로봇 {num}개 시스템 생성 성공")
        print(f"✅ 연속 생성 성공")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return False
    
    return True


def main():
    """메인 함수"""
    print("\n🧪 WormRobotSystem 동적 생성 테스트\n")
    
    success = test_system_creation()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ 모든 테스트 통과!")
        print("=" * 70)
        print("\n💡 Curriculum Learning 준비 완료:")
        print("   - 로봇 1개 시스템 생성 가능")
        print("   - 로봇 2개 시스템 생성 가능")
        print("   - 로봇 4개 시스템 생성 가능")
        print("   - 연속 생성 가능 (Phase 전환 지원)")
        print("\n🚀 이제 train_curriculum.py를 실행할 수 있습니다!")
    else:
        print("❌ 테스트 실패!")
        print("=" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()

