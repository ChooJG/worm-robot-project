"""
Worm Robot Simulation - Extended Demonstrations
대량의 Happy Path 생성으로 Sparse Reward 문제 해결

전략:
1. 다양한 시작 위치 → 목표 경로 생성
2. A* 알고리즘으로 최적 경로 계산
3. 각 경로를 스텝별 demonstration으로 변환
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque
from config import DIRECTIONS, ACTION_MOVE, ACTION_ROTATE_CW, ACTION_ROTATE_CCW


def manhattan_distance(pos1, pos2):
    """맨해튼 거리 계산"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_valid_position(pos):
    """격자 범위 내 위치인지 확인"""
    return -3 <= pos[0] <= 3 and -3 <= pos[1] <= 3


def get_next_position(head, direction):
    """다음 위치 계산"""
    dx, dy = DIRECTIONS[direction]
    return (head[0] + dx, head[1] + dy)


def find_simple_path(start_head, start_tail, start_dir, goal_head):
    """
    간단한 경로 찾기 (BFS 기반)
    
    Returns:
        list: [(head, tail, direction, action), ...] 또는 None
    """
    # BFS
    queue = deque()
    queue.append((start_head, start_tail, start_dir, []))
    visited = set()
    visited.add((start_head, start_tail, start_dir))
    
    max_steps = 20  # 최대 스텝 제한
    
    while queue and len(queue) < 1000:  # 무한 루프 방지
        head, tail, direction, path = queue.popleft()
        
        # 목표 도달
        if head == goal_head and tail == (0, 0):
            return path
        
        # 최대 스텝 초과
        if len(path) >= max_steps:
            continue
        
        # 가능한 행동들
        actions = [
            (ACTION_MOVE, 0),           # 전진
            (ACTION_ROTATE_CW, 1),      # 시계방향 (direction + 1)
            (ACTION_ROTATE_CCW, -1),    # 반시계방향 (direction - 1)
        ]
        
        for action, dir_change in actions:
            if action == ACTION_MOVE:
                # 전진
                new_head = get_next_position(head, direction)
                new_tail = head
                new_dir = direction
            else:
                # 회전
                new_dir = (direction + dir_change) % 4
                new_head = get_next_position(tail, new_dir)
                new_tail = tail
            
            # 유효성 검사
            if not is_valid_position(new_head) or not is_valid_position(new_tail):
                continue
            
            # 방문 체크
            state = (new_head, new_tail, new_dir)
            if state in visited:
                continue
            
            visited.add(state)
            new_path = path + [(head, tail, direction, action, new_head, new_tail, new_dir)]
            queue.append((new_head, new_tail, new_dir, new_path))
    
    return None


def path_to_demonstrations(path, goal_head):
    """
    경로를 demonstration으로 변환
    
    Args:
        path: [(head, tail, direction, action, new_head, new_tail, new_dir), ...]
        goal_head: 목표 위치
    
    Returns:
        list: [(state, action, reward, next_state, done), ...]
    """
    demos = []
    
    for i, (head, tail, direction, action, new_head, new_tail, new_dir) in enumerate(path):
        # 현재 상태
        state = _position_to_state(head, tail, direction, goal_head)
        
        # 다음 상태
        next_state = _position_to_state(new_head, new_tail, new_dir, goal_head)
        
        # 행동 인덱스
        if action == ACTION_MOVE:
            action_idx = 0
        elif action == ACTION_ROTATE_CW:
            action_idx = 1
        else:
            action_idx = 2
        
        # 보상 계산
        curr_dist = manhattan_distance(head, goal_head) + manhattan_distance(tail, (0, 0))
        next_dist = manhattan_distance(new_head, goal_head) + manhattan_distance(new_tail, (0, 0))
        
        # 거리 감소 보상
        reward = (curr_dist - next_dist) * 10.0
        
        # 목표 접근 보너스
        if next_dist < curr_dist:
            reward += 5.0
        
        # 완료 보너스
        done = (i == len(path) - 1)
        if done:
            if new_head == goal_head and new_tail == (0, 0):
                reward += 200.0  # 성공!
        
        demos.append((state, action_idx, reward, next_state, done))
    
    return demos


def _position_to_state(head, tail, direction, goal):
    """위치 정보를 state vector로 변환"""
    return np.array([
        head[0]/3, head[1]/3,
        tail[0]/3, tail[1]/3,
        direction/3,
        (goal[0]-head[0])/6, (goal[1]-head[1])/6,
        (0-tail[0])/6, (0-tail[1])/6,
        goal[0]/3, goal[1]/3,
        0/3, 10/10
    ], dtype=np.float32)


def generate_random_demonstrations(num_demos=50, num_robots=1):
    """
    랜덤 시작 위치에서 다양한 demonstration 생성
    
    Args:
        num_demos: 생성할 demonstration 수
        num_robots: 로봇 수 (목표 위치 결정용)
    
    Returns:
        list: 모든 demonstration 경험들
    """
    from config import GOAL_POSITIONS
    import random
    
    all_demos = []
    goals = GOAL_POSITIONS[:num_robots]
    
    successful_paths = 0
    attempts = 0
    max_attempts = num_demos * 5  # 충분한 시도
    
    print(f"\n🎯 {num_demos}개의 Happy Path 생성 중...")
    
    while successful_paths < num_demos and attempts < max_attempts:
        attempts += 1
        
        # 랜덤 시작 위치
        start_head = (random.randint(-2, 2), random.randint(-2, 2))
        start_tail = (random.randint(-2, 2), random.randint(-2, 2))
        start_dir = random.randint(0, 3)
        
        # 유효성 검사
        if start_head == start_tail:
            continue
        
        if manhattan_distance(start_head, start_tail) != 1:
            continue
        
        # 목표 선택
        goal_head = random.choice(goals)
        
        # 경로 찾기
        path = find_simple_path(start_head, start_tail, start_dir, goal_head)
        
        if path:
            # Demonstration 생성
            demos = path_to_demonstrations(path, goal_head)
            all_demos.extend(demos)
            successful_paths += 1
            
            if successful_paths % 10 == 0:
                print(f"   생성 완료: {successful_paths}/{num_demos}")
    
    print(f"✅ 총 {len(all_demos)}개의 경험 생성 (경로 {successful_paths}개)")
    print(f"   평균 경로 길이: {len(all_demos)/max(successful_paths, 1):.1f} 스텝")
    
    return all_demos


def generate_demonstrations_grid(num_robots=1):
    """
    격자 기반으로 체계적인 demonstration 생성
    
    주요 시나리오 커버:
    - 각 사분면에서 시작
    - 가까운 거리 / 먼 거리
    - 직선 경로 / 우회 경로
    """
    from config import GOAL_POSITIONS
    
    all_demos = []
    goals = GOAL_POSITIONS[:num_robots]
    
    print(f"\n🎯 격자 기반 Happy Path 생성 중...")
    
    # 체계적인 시작 위치들
    start_positions = [
        # 가까운 위치들
        ((-1, -1), (-1, 0), 0),  # 중앙 근처
        ((-1, 1), (-1, 0), 0),
        ((1, -1), (1, 0), 2),
        ((1, 1), (1, 0), 2),
        
        # 중간 거리
        ((-2, -2), (-2, -1), 0),
        ((-2, 2), (-2, 1), 0),
        ((2, -2), (2, -1), 2),
        ((2, 2), (2, 1), 2),
        
        # 먼 거리
        ((-3, -3), (-3, -2), 0),
        ((-3, 3), (-3, 2), 0),
        ((3, -3), (3, -2), 2),
        ((3, 3), (3, 2), 2),
        
        # 다양한 방향
        ((0, -2), (0, -3), 1),
        ((0, 2), (0, 3), 3),
        ((-2, 0), (-3, 0), 2),
        ((2, 0), (3, 0), 0),
    ]
    
    successful_paths = 0
    
    for start_head, start_tail, start_dir in start_positions:
        # 유효성 검사
        if not is_valid_position(start_head) or not is_valid_position(start_tail):
            continue
        
        for goal_head in goals:
            # 경로 찾기
            path = find_simple_path(start_head, start_tail, start_dir, goal_head)
            
            if path:
                demos = path_to_demonstrations(path, goal_head)
                all_demos.extend(demos)
                successful_paths += 1
    
    print(f"✅ 총 {len(all_demos)}개의 경험 생성 (경로 {successful_paths}개)")
    print(f"   평균 경로 길이: {len(all_demos)/max(successful_paths, 1):.1f} 스텝")
    
    return all_demos


def get_extended_demonstrations(num_robots=1, num_random=30):
    """
    확장된 demonstration 세트 반환
    
    Args:
        num_robots: 로봇 수
        num_random: 추가로 생성할 랜덤 경로 수
    
    Returns:
        list: 모든 demonstration 경험들
    """
    all_demos = []
    
    # 1. 격자 기반 체계적 경로
    grid_demos = generate_demonstrations_grid(num_robots=num_robots)
    all_demos.extend(grid_demos)
    
    # 2. 랜덤 경로 추가
    random_demos = generate_random_demonstrations(num_demos=num_random, num_robots=num_robots)
    all_demos.extend(random_demos)
    
    print(f"\n📊 최종 통계:")
    print(f"   총 경험 수: {len(all_demos)}개")
    print(f"   예상 경로 수: {len(all_demos)//10}~{len(all_demos)//5}개")
    
    return all_demos


if __name__ == "__main__":
    # 테스트
    print("=" * 70)
    print("🧪 Extended Demonstrations 생성 테스트")
    print("=" * 70)
    
    # 로봇 1개
    demos_1robot = get_extended_demonstrations(num_robots=1, num_random=20)
    print(f"\n로봇 1개: {len(demos_1robot)}개 경험")
    
    # 로봇 2개
    demos_2robots = get_extended_demonstrations(num_robots=2, num_random=30)
    print(f"\n로봇 2개: {len(demos_2robots)}개 경험")
    
    # 보상 분포 확인
    rewards = [demo[2] for demo in demos_1robot]
    print(f"\n보상 통계:")
    print(f"  평균: {np.mean(rewards):.1f}")
    print(f"  최대: {np.max(rewards):.1f}")
    print(f"  최소: {np.min(rewards):.1f}")
    
    print("\n✅ 테스트 완료!")

