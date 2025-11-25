"""
Worm Robot Simulation - Environment Model
환경 DEVS 모델 정의
"""

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import DIR_NAMES, STATUS_RUNNING, STATUS_WIN, STATUS_FAIL
from utils import in_bounds, get_sensor_area


# ========================================
# Environment 상태 클래스
# ========================================

class EnvironmentState:
    """환경의 내부 상태를 표현하는 클래스"""

    def __init__(self):
        self.robot_positions = {}  # {robot_id: {"head": (x,y), "tail": (x,y), "direction": int}}
        self.status = STATUS_RUNNING
        self.step_count = 0
        self.phase = "INIT"        # 상태: INIT, IDLE, PROCESSING
        self.pending_updates = []  # 대기 중인 로봇 업데이트
        self.rewards = {}          # {robot_id: reward} - 각 로봇의 보상
        self.prev_distances = {}   # {robot_id: distance} - 이전 스텝의 목표까지 거리

    def __str__(self):
        return (
            f"Environment["
            f"상태:{self.phase},"
            f"게임상태:{self.status},"
            f"스텝:{self.step_count},"
            f"로봇수:{len(self.robot_positions)}]"
        )


# ========================================
# Environment 모델 (Atomic DEVS)
# ========================================

class Environment(AtomicDEVS):
    """로봇들이 활동하는 환경의 DEVS 모델"""

    def __init__(self, num_robots=1, initial_positions=None, robot_goals=None, obstacles=None, moving_obstacles=None):
        """
        Args:
            num_robots: 로봇 수
            initial_positions: 초기 로봇 위치 리스트
            robot_goals: 각 로봇의 목적지 딕셔너리 {robot_id: goal_position}
            obstacles: 정적 장애물 위치 리스트 [(x, y), ...] (선택)
            moving_obstacles: 움직이는 장애물 리스트 [MovingObstacle, ...] (선택)
        """
        AtomicDEVS.__init__(self, "Environment")
        self.num_robots = num_robots
        self.state = EnvironmentState()
        self.initial_positions = initial_positions or []
        self.robot_goals = robot_goals or {}  # 각 로봇의 목적지
        self.obstacles = obstacles or []  # 정적 장애물 위치
        self.moving_obstacles = moving_obstacles or []  # 움직이는 장애물

        # 로봇 초기 위치 설정
        for pos_data in self.initial_positions:
            self.state.robot_positions[pos_data["id"]] = {
                "head": pos_data["head"],
                "tail": pos_data["tail"],
                "direction": pos_data["dir"]
            }

        # 입력 포트 (로봇들로부터)
        self.robot_done_in = [self.addInPort(f"robot{i}_done_in") for i in range(num_robots)]

        # 출력 포트 (컨트롤러로)
        self.obs_out = self.addOutPort("obs_out")          # 관찰 데이터
        self.status_out = self.addOutPort("status_out")    # 게임 상태

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "INIT":
            return 0  # 즉시 초기화
        elif self.state.phase == "IDLE":
            if self.state.status != STATUS_RUNNING:
                return INFINITY  # 게임 종료
            return INFINITY  # 로봇 행동 대기
        elif self.state.phase == "PROCESSING":
            return 0  # 즉시 처리
        return INFINITY

    def intTransition(self):
        """내부 전이 함수"""
        if self.state.phase == "INIT":
            # 초기 관찰 데이터를 보낸 후 IDLE로 전환
            self.state.phase = "IDLE"
            return self.state

        elif self.state.phase == "PROCESSING":
            # 로봇 위치 업데이트 및 승패 판정
            self._update_environment()
            self.state.pending_updates = []
            self.state.phase = "IDLE"
            return self.state

        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 로봇 행동 완료 신호 수신"""
        if self.state.phase == "IDLE":
            # 로봇들의 행동 완료 신호 수집
            for i, port in enumerate(self.robot_done_in):
                update = inputs.get(port)
                if update:
                    self.state.pending_updates.append(update)

            # 모든 로봇이 완료했으면 처리 시작
            if len(self.state.pending_updates) > 0:
                self.state.phase = "PROCESSING"

        return self.state

    def outputFnc(self):
        """출력 함수 - 관찰 데이터 및 상태 전송"""
        # INIT이나 PROCESSING 상태에서만 출력 (IDLE에서는 출력 안함)
        if self.state.phase in ["INIT", "PROCESSING"]:
            obs = self._generate_observations()
            return {
                self.obs_out: obs,
                self.status_out: {
                    "status": self.state.status,
                    "step": self.state.step_count
                }
            }
        return {}

    def _update_environment(self):
        """환경 업데이트: 로봇 위치 갱신 및 승패 판정"""
        # 1. 움직이는 장애물 업데이트
        for moving_obs in self.moving_obstacles:
            moving_obs.step()
        
        # 2. 로봇 위치 갱신
        for update in self.state.pending_updates:
            rid = update["robot_id"]
            self.state.robot_positions[rid] = {
                "head": update["head"],
                "tail": update["tail"],
                "direction": update["direction"]
            }

        self.state.step_count += 1

        # 보상 계산 (승패 판정 전에 수행)
        self._calculate_rewards()

        # 승패 판정
        if self._check_fail():
            self.state.status = STATUS_FAIL
        elif self._check_win():
            self.state.status = STATUS_WIN

    def _check_fail(self):
        """실패 조건 확인: 격자 이탈, 로봇 충돌, 또는 장애물 충돌"""
        positions = self.state.robot_positions

        # 격자 범위 이탈 확인
        for rid, pos_data in positions.items():
            if not in_bounds(pos_data["head"]) or not in_bounds(pos_data["tail"]):
                print(f"[실패] Robot {rid}가 격자를 벗어났습니다!")
                return True

        # 정적 장애물 충돌 확인
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]
            
            if head in self.obstacles:
                print(f"[실패] Robot {rid}의 앞발이 {head} 위치의 정적 장애물과 충돌!")
                return True
            
            if tail in self.obstacles:
                print(f"[실패] Robot {rid}의 뒷발이 {tail} 위치의 정적 장애물과 충돌!")
                return True
        
        # 움직이는 장애물 충돌 확인
        moving_obs_positions = [obs.get_position() for obs in self.moving_obstacles]
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]
            
            if head in moving_obs_positions:
                print(f"[실패] Robot {rid}의 앞발이 {head} 위치의 움직이는 장애물과 충돌!")
                return True
            
            if tail in moving_obs_positions:
                print(f"[실패] Robot {rid}의 뒷발이 {tail} 위치의 움직이는 장애물과 충돌!")
                return True

        # 로봇 간 충돌 확인
        occupied = {}
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]

            # 앞발 충돌 체크
            if head in occupied:
                print(f"[실패] {head} 위치에서 Robot {rid}와 Robot {occupied[head]}가 충돌!")
                return True
            occupied[head] = rid

            # 뒷발 충돌 체크 (단, (0,0)은 예외)
            if tail != (0, 0):
                if tail in occupied:
                    print(f"[실패] {tail} 위치에서 Robot {rid}와 Robot {occupied[tail]}가 충돌!")
                    return True
                occupied[tail] = rid

        return False

    def _check_win(self):
        """승리 조건 확인: 모든 로봇이 자신의 목적지에 도달"""
        if len(self.state.robot_positions) < self.num_robots:
            return False

        # 각 로봇이 자신의 목적지에 도달했는지 확인
        for rid, pos_data in self.state.robot_positions.items():
            # 뒷발이 중앙 (0,0)에 있는지 확인
            if pos_data["tail"] != (0, 0):
                return False
            
            # 앞발이 목적지에 있는지 확인
            goal_position = self.robot_goals.get(rid)
            if goal_position is None or pos_data["head"] != goal_position:
                return False

        print(f"[승리] 모든 로봇이 자신의 목적지에 성공적으로 도착했습니다!")
        return True

    def _generate_observations(self):
        """각 로봇의 센서 관찰 데이터 생성"""
        observations = {}

        for rid, pos_data in self.state.robot_positions.items():
            head = pos_data["head"]
            sensor_area = get_sensor_area(head)

            # 센서 범위 내 다른 로봇 감지
            detected = []
            for other_id, other_pos in self.state.robot_positions.items():
                if other_id == rid:
                    continue
                if other_pos["head"] in sensor_area or other_pos["tail"] in sensor_area:
                    detected.append({
                        "robot_id": other_id,
                        "head": other_pos["head"],
                        "tail": other_pos["tail"]
                    })

            # 움직이는 장애물 위치
            moving_obs_positions = [obs.get_position() for obs in self.moving_obstacles]
            
            observations[rid] = {
                "own_head": head,
                "own_tail": pos_data["tail"],
                "own_direction": pos_data["direction"],
                "detected_robots": detected,
                "goal_position": self.robot_goals.get(rid, (0, 0)),  # 목적지 좌표 전달
                "obstacles": self.obstacles,  # 정적 장애물 위치 전달
                "moving_obstacles": moving_obs_positions  # 움직이는 장애물 위치 전달
            }

        return observations
    
    def _calculate_rewards(self):
        """각 로봇의 보상 계산 (극대화 버전 - 명확한 신호)"""
        for rid, pos_data in self.state.robot_positions.items():
            reward = 0.0
            
            # 현재 위치
            tail = pos_data["tail"]
            head = pos_data["head"]
            goal_head = self.robot_goals.get(rid, (0, 0))
            
            # 목표까지 거리 (Manhattan distance)
            tail_dist = abs(tail[0]) + abs(tail[1])
            head_dist = abs(head[0] - goal_head[0]) + abs(head[1] - goal_head[1])
            total_dist = tail_dist + head_dist
            
            # 1. 기본 스텝 페널티 (10배 강화!)
            reward -= 0.5  # 0.05 → 0.5 (최소 스텝 유도)
            
            # STAY 행동 페널티 (로봇 1개일 때만 - 정적 장애물에서는 무의미)
            # 다중 로봇에서는 STAY가 충돌 회피 전략이 될 수 있으므로 페널티 X
            # 주의: 이 로직은 system.step()에서 처리하는게 더 적절
            # 여기서는 일단 주석 처리
            
            # 2. 거리 기반 보상 (더 강하게!)
            # 거리가 가까울수록 기하급수적으로 증가
            distance_reward = (12 - total_dist) / 12 * 20.0  # 5.0 → 20.0
            reward += distance_reward
            
            # 3. 거리 감소 보너스 (극대화!)
            if rid in self.state.prev_distances:
                prev_dist = self.state.prev_distances[rid]
                distance_change = prev_dist - total_dist
                
                if distance_change > 0:
                    # 가까워지면 엄청난 보너스! (20배 증폭)
                    reward += distance_change * 20.0  # 10.0 → 20.0
                elif distance_change < 0:
                    # 멀어지면 강한 페널티 (10배 증폭)
                    reward += distance_change * 10.0  # 5.0 → 10.0
            
            # 현재 거리 저장 (다음 스텝 비교용)
            self.state.prev_distances[rid] = total_dist
            
            # 4. 중간 목표 달성 시 엄청난 보너스!
            tail_at_center = (tail == (0, 0))
            head_at_goal = (head == goal_head)
            
            if tail_at_center and not head_at_goal:
                # 뒷발만 중앙 도달 - 큰 보너스!
                reward += 100.0
            elif head_at_goal and not tail_at_center:
                # 앞발만 목표 도달 - 큰 보너스!
                reward += 100.0
            elif tail_at_center and head_at_goal:
                # 완전 성공! - 초대형 보너스!!!
                reward += 500.0
                
                # 효율성 보너스 (빠르게 도달할수록 추가 보상)
                steps_used = self.state.step_count
                max_steps = 100  # 평균 예상 최대 스텝
                if steps_used < max_steps:
                    efficiency_bonus = (max_steps - steps_used) * 5.0
                    reward += efficiency_bonus
            
            # 5. 거리별 추가 보너스 (매우 가까울 때 더 큰 보상)
            if total_dist <= 2:
                reward += 50.0  # 거의 다 왔을 때 추가 보상
            elif total_dist <= 4:
                reward += 20.0
            elif total_dist <= 6:
                reward += 10.0
            
            # 6. 격자 경계 근처 경고 (이탈 방지)
            if not self._is_position_safe(head) or not self._is_position_safe(tail):
                reward -= 5.0  # 2.0 → 5.0 (더 강한 페널티)
            
            # 7. 장애물 근처 경고
            if self._is_near_obstacle(head) or self._is_near_obstacle(tail):
                reward -= 3.0  # 1.0 → 3.0
            
            self.state.rewards[rid] = reward
    
    def _is_position_safe(self, pos):
        """위치가 격자 경계에서 안전한지 확인 (1칸 여유)"""
        x, y = pos
        return -2 <= x <= 2 and -2 <= y <= 2
    
    def _is_near_obstacle(self, pos):
        """위치가 장애물 근처인지 확인 (1칸 이내)"""
        x, y = pos
        for ox, oy in self.obstacles:
            if abs(x - ox) <= 1 and abs(y - oy) <= 1:
                return True
        return False
    
    def get_rewards(self):
        """현재 보상 반환 (RL 학습용)"""
        return self.state.rewards.copy()
