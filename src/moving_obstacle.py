"""
움직이는 장애물 모듈

로봇이 STAY 행동을 학습하기 위한 규칙적으로 움직이는 장애물 구현
"""


class MovingObstacle:
    """
    규칙적으로 움직이는 장애물
    
    로봇이 STAY 행동으로 회피하는 것을 학습하기 위한 장애물
    """
    
    def __init__(self, pattern="horizontal", start_pos=(0, 0), move_range=(-2, 2), speed=1):
        """
        Args:
            pattern: 움직임 패턴 ("horizontal", "vertical")
            start_pos: 시작 위치 (x, y)
            move_range: 이동 범위 (min, max)
            speed: 이동 속도 (스텝당 칸 수)
        """
        self.pattern = pattern
        self.start_pos = start_pos
        self.move_range = move_range
        self.speed = speed
        
        # 현재 위치
        self.x, self.y = start_pos
        
        # 이동 방향 (1: 증가, -1: 감소)
        self.direction = 1
        
    def step(self):
        """한 스텝 진행: 장애물을 규칙에 따라 이동"""
        if self.pattern == "horizontal":
            # 수평 왕복 이동
            self.x += self.direction * self.speed
            
            # 범위 체크 및 방향 전환
            if self.x >= self.move_range[1]:
                self.x = self.move_range[1]
                self.direction = -1
            elif self.x <= self.move_range[0]:
                self.x = self.move_range[0]
                self.direction = 1
                
        elif self.pattern == "vertical":
            # 수직 왕복 이동
            self.y += self.direction * self.speed
            
            # 범위 체크 및 방향 전환
            if self.y >= self.move_range[1]:
                self.y = self.move_range[1]
                self.direction = -1
            elif self.y <= self.move_range[0]:
                self.y = self.move_range[0]
                self.direction = 1
    
    def get_position(self):
        """현재 위치 반환"""
        return (self.x, self.y)
    
    def reset(self):
        """초기 위치로 리셋"""
        self.x, self.y = self.start_pos
        self.direction = 1
    
    def __repr__(self):
        return f"MovingObstacle(pos={self.get_position()}, pattern={self.pattern}, dir={self.direction})"


def create_horizontal_obstacle(y=0, speed=1):
    """
    수평 왕복 장애물 생성 (헬퍼 함수)
    
    Args:
        y: y 좌표 (기본값: 0, 중앙 라인)
        speed: 이동 속도
    
    Returns:
        MovingObstacle 인스턴스
    """
    return MovingObstacle(
        pattern="horizontal",
        start_pos=(-2, y),  # 왼쪽 끝에서 시작
        move_range=(-2, 2),
        speed=speed
    )


def create_vertical_obstacle(x=0, speed=1):
    """
    수직 왕복 장애물 생성 (헬퍼 함수)
    
    Args:
        x: x 좌표 (기본값: 0, 중앙 라인)
        speed: 이동 속도
    
    Returns:
        MovingObstacle 인스턴스
    """
    return MovingObstacle(
        pattern="vertical",
        start_pos=(x, -2),  # 아래쪽 끝에서 시작
        move_range=(-2, 2),
        speed=speed
    )

