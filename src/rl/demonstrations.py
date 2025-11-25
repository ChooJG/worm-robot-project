"""
Worm Robot Simulation - Demonstrations (Happy Paths)
미리 정의된 성공 경로 생성기

Sparse Reward 문제 해결:
- 성공 경험을 미리 Replay Buffer에 추가
- Q-network가 "어떻게 하면 성공하는지" 학습할 단서 제공
"""

import numpy as np
from config import ACTION_MOVE, ACTION_ROTATE_CW, ACTION_ROTATE_CCW


def create_simple_demonstration_1robot():
    """
    로봇 1개가 목표에 도달하는 간단한 경로
    
    시나리오:
    - 초기: tail=(-2, 0), head=(-1, 0), direction=0(동), goal=(1, 0)
    - 목표: tail=(0, 0), head=(1, 0)
    
    Returns:
        list: [(state, action, reward, next_state, done), ...]
    """
    demos = []
    
    # Step 1: 초기 상태 → 전진
    state = np.array([
        -1/3, 0/3,           # head (-1, 0)
        -2/3, 0/3,           # tail (-2, 0)
        0/3,                 # direction (동)
        2/6, 0/6,           # vector to goal head: (1-(-1), 0-0) = (2, 0)
        2/6, 0/6,           # vector to goal tail: (0-(-2), 0-0) = (2, 0)
        1/3, 0/3,           # goal position (1, 0)
        0/3,                # num nearby robots
        10/10               # closest distance (no robots)
    ], dtype=np.float32)
    
    next_state = np.array([
        0/3, 0/3,           # head (0, 0) - 전진
        -1/3, 0/3,          # tail (-1, 0) - 따라옴
        0/3,                # direction (동)
        1/6, 0/6,           # vector to goal head: (1, 0)
        1/6, 0/6,           # vector to goal tail: (1, 0)
        1/3, 0/3,           # goal position (1, 0)
        0/3, 10/10
    ], dtype=np.float32)
    
    action = 0  # ACTION_MOVE
    reward = 10.0  # 목표에 가까워짐
    done = False
    
    demos.append((state, action, reward, next_state, done))
    
    # Step 2: 중간 상태 → 전진
    state = next_state
    next_state = np.array([
        1/3, 0/3,           # head (1, 0) - 목표 도착!
        0/3, 0/3,           # tail (0, 0) - 중앙 도착!
        0/3,                # direction (동)
        0/6, 0/6,           # vector to goal head: (0, 0) - 도착!
        0/6, 0/6,           # vector to goal tail: (0, 0) - 도착!
        1/3, 0/3,           # goal position (1, 0)
        0/3, 10/10
    ], dtype=np.float32)
    
    reward = 300.0  # 대성공!
    done = True
    
    demos.append((state, action, reward, next_state, done))
    
    return demos


def create_demonstration_with_rotation():
    """
    회전이 포함된 데모 경로
    
    시나리오:
    - 초기: tail=(1, 1), head=(1, 2), direction=3(북), goal=(-1, 0)
    - 잘못된 방향 → 회전 필요
    """
    demos = []
    
    # Step 1: 잘못된 방향 → 회전
    state = np.array([
        1/3, 2/3,           # head (1, 2)
        1/3, 1/3,           # tail (1, 1)
        3/3,                # direction (북)
        -2/6, -2/6,         # vector to goal head
        -1/6, -1/6,         # vector to goal tail
        -1/3, 0/3,          # goal position (-1, 0)
        0/3, 10/10
    ], dtype=np.float32)
    
    next_state = np.array([
        2/3, 2/3,           # head (2, 2) - 시계방향 회전 (동쪽)
        1/3, 2/3,           # tail (1, 2) - 따라옴
        0/3,                # direction (동)
        -3/6, -2/6,         # vector updated
        -1/6, -2/6,
        -1/3, 0/3,
        0/3, 10/10
    ], dtype=np.float32)
    
    action = 1  # ACTION_ROTATE_CW
    reward = 3.0  # 회전 보상
    done = False
    
    demos.append((state, action, reward, next_state, done))
    
    return demos


def create_demonstration_avoid_collision():
    """
    충돌 회피 데모
    
    시나리오:
    - 로봇 2개가 근처에 있을 때 충돌 회피
    """
    demos = []
    
    # 다른 로봇 감지 → 회전으로 회피
    state = np.array([
        0/3, 1/3,           # head (0, 1)
        0/3, 0/3,           # tail (0, 0) - 이미 중앙!
        3/3,                # direction (북)
        1/6, 0/6,           # vector to goal head (1, 1)
        0/6, 0/6,           # vector to goal tail (0, 0) - 도착
        1/3, 1/3,           # goal position (1, 1)
        1/3,                # num nearby robots: 1개 감지!
        1/10                # closest distance: 1
    ], dtype=np.float32)
    
    next_state = np.array([
        1/3, 1/3,           # head (1, 1) - 회전 후 전진
        0/3, 1/3,           # tail (0, 1)
        0/3,                # direction (동)
        0/6, 0/6,           # 목표 도달!
        -1/6, -1/6,
        1/3, 1/3,
        0/3, 10/10          # 로봇 멀어짐
    ], dtype=np.float32)
    
    action = 1  # ACTION_ROTATE_CW (회피)
    reward = 15.0  # 충돌 회피 + 목표 접근
    done = False
    
    demos.append((state, action, reward, next_state, done))
    
    return demos


def get_all_demonstrations(num_robots=1):
    """
    로봇 수에 따라 적절한 데모 반환
    
    Args:
        num_robots: 로봇 수
    
    Returns:
        list: 모든 demonstration 경험들
    """
    demos = []
    
    # 기본: 전진하여 목표 도달
    demos.extend(create_simple_demonstration_1robot())
    
    # 추가: 회전 사용
    demos.extend(create_demonstration_with_rotation())
    
    # 로봇 2개 이상: 충돌 회피 데모 추가
    if num_robots >= 2:
        demos.extend(create_demonstration_avoid_collision())
    
    return demos


def create_adaptive_demonstrations(current_position, goal_position):
    """
    현재 위치와 목표를 기반으로 동적 데모 생성
    
    Args:
        current_position: {"head": (x, y), "tail": (x, y), "direction": int}
        goal_position: (goal_x, goal_y)
    
    Returns:
        list: 생성된 demonstration
    """
    # TODO: A* 알고리즘 등으로 최적 경로 계산
    # 현재는 간단한 휴리스틱 경로 생성
    demos = []
    
    head = current_position["head"]
    tail = current_position["tail"]
    direction = current_position["direction"]
    
    # 목표까지 맨해튼 거리
    dist_head = abs(goal_position[0] - head[0]) + abs(goal_position[1] - head[1])
    dist_tail = abs(tail[0]) + abs(tail[1])
    
    # 가까우면 전진 추천
    if dist_head + dist_tail < 4:
        state = _position_to_state(head, tail, direction, goal_position)
        # 전진 후 상태 (간단히 추정)
        next_head = (head[0] + (1 if direction == 0 else -1 if direction == 2 else 0),
                     head[1] + (-1 if direction == 1 else 1 if direction == 3 else 0))
        next_state = _position_to_state(next_head, head, direction, goal_position)
        
        demos.append((state, 0, 20.0, next_state, False))  # ACTION_MOVE
    
    return demos


def _position_to_state(head, tail, direction, goal):
    """위치 정보를 state vector로 변환 (헬퍼 함수)"""
    return np.array([
        head[0]/3, head[1]/3,
        tail[0]/3, tail[1]/3,
        direction/3,
        (goal[0]-head[0])/6, (goal[1]-head[1])/6,
        (0-tail[0])/6, (0-tail[1])/6,
        goal[0]/3, goal[1]/3,
        0/3, 10/10
    ], dtype=np.float32)


if __name__ == "__main__":
    # 테스트
    print("=== Demonstration Examples ===\n")
    
    demos_1robot = get_all_demonstrations(num_robots=1)
    print(f"로봇 1개 Demonstrations: {len(demos_1robot)}개")
    for i, (s, a, r, ns, d) in enumerate(demos_1robot):
        print(f"  Demo {i+1}: action={a}, reward={r:.1f}, done={d}")
    
    demos_2robots = get_all_demonstrations(num_robots=2)
    print(f"\n로봇 2개 Demonstrations: {len(demos_2robots)}개")
    for i, (s, a, r, ns, d) in enumerate(demos_2robots):
        print(f"  Demo {i+1}: action={a}, reward={r:.1f}, done={d}")

