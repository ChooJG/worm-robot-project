"""
초간단 Curriculum Learning
로봇 1개 기준으로 난이도를 점진적으로 증가

Phase 0: 장애물 없음 (기본 이동)
Phase 1: 장애물 1개 (모서리)
Phase 2: 장애물 1개 (중앙 근처)
Phase 3: 장애물 3개
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.agent import DQNAgent
from rl.replay_buffer import PrioritizedReplayBuffer
from rl.demonstrations_extended import get_extended_demonstrations
from system import WormRobotSystem
from moving_obstacle import create_horizontal_obstacle
from config import STATUS_WIN, STATUS_FAIL, STATUS_RUNNING
import numpy as np

class SimpleCurriculumTrainer:
    """단순화된 Curriculum Learning 트레이너"""
    
    def __init__(self, agent, batch_size=128, log_interval=100):
        self.agent = agent
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
    
    def train_phase(self, phase_name, obstacles, num_episodes, termination_time=80, 
                    success_threshold=0.3, model_path=None, num_robots=1, moving_obstacles=None):
        """
        단일 Phase 학습
        
        Args:
            phase_name: Phase 이름 (예: "Phase 0")
            obstacles: 정적 장애물 리스트 (None or [(x, y), ...])
            num_episodes: 학습 에피소드 수
            termination_time: 최대 스텝
            success_threshold: 목표 성공률 (조기 종료용)
            model_path: 저장할 모델 경로
            num_robots: 로봇 수 (기본값 1)
            moving_obstacles: 움직이는 장애물 리스트 (None or [MovingObstacle, ...])
        """
        print("\n" + "=" * 70)
        print(f"📚 {phase_name} 학습 시작")
        print("=" * 70)
        print(f"로봇 수: {num_robots}개")
        print(f"정적 장애물: {obstacles if obstacles else '없음'}")
        print(f"움직이는 장애물: {len(moving_obstacles) if moving_obstacles else 0}개")
        print(f"에피소드: {num_episodes}")
        print(f"목표 성공률: {success_threshold * 100:.0f}%")
        print("=" * 70)
        
        # 시스템 생성 함수
        def create_system_fn(rl_agent=None):
            return WormRobotSystem(
                rl_agent=rl_agent, 
                num_robots=num_robots,
                obstacles=obstacles,
                moving_obstacles=moving_obstacles
            )
        
        # 중단된 학습 재개 확인
        resumed = False
        if model_path:
            # 임시 모델 경로 확인 (우선순위: interrupted > error)
            tmp_interrupted = model_path.replace('.pth', '_tmp_interrupted.pth')
            tmp_error = model_path.replace('.pth', '_tmp_error.pth')
            
            import os
            if os.path.exists(tmp_interrupted):
                print(f"\n🔄 중단된 학습 발견! 이어서 진행합니다...")
                print(f"   모델 로드: {tmp_interrupted}")
                self.agent.load(tmp_interrupted)
                resumed = True
                # 로드 후 임시 파일 삭제 (선택적)
                # os.remove(tmp_interrupted)
            elif os.path.exists(tmp_error):
                print(f"\n🔄 오류로 중단된 학습 발견! 이어서 진행합니다...")
                print(f"   모델 로드: {tmp_error}")
                self.agent.load(tmp_error)
                resumed = True
                # 로드 후 임시 파일 삭제 (선택적)
                # os.remove(tmp_error)
        
        if resumed:
            print(f"   ✅ 이전 학습 상태에서 재개합니다!")
            print(f"   현재 Epsilon: {self.agent.epsilon:.3f}")
        
        # 통계
        stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_losses": [],
            "success_count": 0,
            "fail_count": 0
        }
        
        # Happy Path 추가
        print(f"\n📖 Happy Path 추가 중...")
        
        # 움직이는 장애물이 있으면 Happy Path 사용 안 함 (시행착오로 학습)
        if moving_obstacles and len(moving_obstacles) > 0:
            print(f"   ⚠️ 움직이는 장애물 환경: Happy Path 생략 (시행착오 학습)")
            demos = []
        elif num_robots == 1:
            if obstacles is None or len(obstacles) == 0:
                # Phase 0: 로봇 1개, 장애물 없음 - 대량의 데모
                demos = get_extended_demonstrations(num_robots=1, num_random=3000)
            else:
                # Phase 1-3: 로봇 1개, 장애물 있음 - 적은 데모
                demos = get_extended_demonstrations(num_robots=1, num_random=500)
        else:
            # Phase 3.5, 4: 로봇 2개 - 충분한 데모 (중요!)
            if obstacles is None or len(obstacles) == 0:
                # 장애물 없음: 대량 데모
                demos = get_extended_demonstrations(num_robots=2, num_random=2000)
            else:
                # 장애물 있음: 중간 데모
                demos = get_extended_demonstrations(num_robots=2, num_random=1000)
        
        if demos:
            self.replay_buffer.add_demonstrations(demos)
            print(f"   ✅ 총 {len(demos)}개의 성공 경험 추가!")
        else:
            print(f"   ℹ️ Happy Path 없이 학습 시작")
        
        print(f"\n🚀 학습 시작!\n")
        
        best_success_rate = 0.0
        
        # 학습 루프
        try:
            for episode in range(num_episodes):
                episode_reward, episode_steps, status = self._run_episode(create_system_fn, termination_time)
                
                # 통계 기록
                stats["episode_rewards"].append(episode_reward)
                stats["episode_steps"].append(episode_steps)
                
                if status == STATUS_WIN:
                    stats["success_count"] += 1
                elif status == STATUS_FAIL:
                    stats["fail_count"] += 1
                
                # 학습
                if len(self.replay_buffer.buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    batch = list(zip(states, actions, rewards, next_states, dones))
                    loss = self.agent.train(batch)
                    stats["episode_losses"].append(loss)
                
                # Epsilon 감소
                self.agent.update_epsilon()
                
                # 로그 출력
                if (episode + 1) % self.log_interval == 0:
                    recent = min(self.log_interval, len(stats["episode_rewards"]))
                    avg_reward = sum(stats["episode_rewards"][-recent:]) / recent
                    avg_steps = sum(stats["episode_steps"][-recent:]) / recent
                    avg_loss = sum(stats["episode_losses"][-recent:]) / recent if stats["episode_losses"] else 0
                    
                    # 최근 성공률 계산
                    recent_episodes = stats["success_count"] + stats["fail_count"]
                    recent_success_rate = stats["success_count"] / recent_episodes if recent_episodes > 0 else 0
                    
                    print(
                        f"Ep {episode + 1:5d}/{num_episodes} | "
                        f"Reward: {avg_reward:7.1f} | "
                        f"Steps: {avg_steps:4.1f} | "
                        f"Loss: {avg_loss:.2f} | "
                        f"ε: {self.agent.epsilon:.3f} | "
                        f"Success: {recent_success_rate*100:4.1f}% | "
                        f"Win: {stats['success_count']:4d}"
                    )
                    
                    # 최고 성공률 갱신
                    if recent_success_rate > best_success_rate:
                        best_success_rate = recent_success_rate
                        if model_path:
                            self.agent.save(model_path)
                            print(f"   ✅ 새 최고 성공률! 모델 저장: {model_path}")
                
                # 조기 종료 조건 체크 (충분히 학습했으면)
                if episode > 1000 and episode % 500 == 0:
                    # 전체 성공률로 판단
                    total_episodes = stats["success_count"] + stats["fail_count"]
                    current_success_rate = stats["success_count"] / total_episodes if total_episodes > 0 else 0
                    
                    if current_success_rate >= success_threshold:
                        print(f"\n🎉 목표 달성! (전체 성공률: {current_success_rate*100:.1f}%)")
                        break
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 사용자가 학습을 중단했습니다!")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_interrupted.pth')
                self.agent.save(tmp_path)
                print(f"   💾 임시 모델 저장: {tmp_path}")
                print(f"   현재까지 진행: {episode + 1}/{num_episodes} 에피소드")
            raise  # 예외를 다시 발생시켜 상위에서 처리
        
        except Exception as e:
            print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_error.pth')
                self.agent.save(tmp_path)
                print(f"   💾 임시 모델 저장: {tmp_path}")
            raise  # 예외를 다시 발생시켜 상위에서 처리
        
        # 최종 저장
        if model_path:
            self.agent.save(model_path)
        
        # 최종 통계
        total_episodes = stats["success_count"] + stats["fail_count"]
        final_success_rate = stats["success_count"] / total_episodes if total_episodes > 0 else 0
        
        print("\n" + "=" * 70)
        print(f"✅ {phase_name} 완료!")
        print("=" * 70)
        print(f"총 성공: {stats['success_count']}")
        print(f"총 실패: {stats['fail_count']}")
        print(f"최종 성공률: {final_success_rate * 100:.1f}%")
        print(f"최고 성공률: {best_success_rate * 100:.1f}%")
        print("=" * 70)
        
        return stats, final_success_rate
    
    def _run_episode(self, create_system_fn, termination_time):
        """에피소드 실행"""
        system = create_system_fn(rl_agent=self.agent)
        num_robots = len(system.robots)
        
        episode_reward = 0.0
        step_count = 0
        
        while not system.is_done() and step_count < termination_time:
            # 현재 상태
            current_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    current_states[rid] = state
            
            # 행동 선택
            actions = {}
            for rid in current_states.keys():
                action = self.agent.get_action(current_states[rid])
                actions[rid] = action
            
            # 스텝 실행
            observations, rewards, done, status = system.step(actions)
            
            # 다음 상태
            next_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    next_states[rid] = state
            
            # 경험 저장
            step_reward = 0.0
            for rid in current_states.keys():
                if rid in next_states and rid in rewards:
                    robot_reward = rewards[rid]
                    
                    # 실패 시 큰 페널티
                    if done and status == STATUS_FAIL:
                        robot_reward -= 300.0
                    elif done and status == STATUS_WIN:
                        robot_reward += 300.0
                    
                    step_reward += robot_reward
                    
                    self.replay_buffer.add(
                        current_states[rid],
                        actions[rid],
                        robot_reward,
                        next_states[rid],
                        float(done)
                    )
            
            episode_reward += step_reward
            step_count += 1
            
            if done:
                break
        
        avg_reward = episode_reward / num_robots if num_robots > 0 else 0.0
        final_status = system.get_status()
        
        return avg_reward, step_count, final_status


def main():
    print("\n" + "=" * 70)
    print("🎓 개선된 Curriculum Learning (STAY 학습 포함)")
    print("=" * 70)
    print("전략:")
    print("  Phase 0-3:   로봇 1개 (정적 장애물 난이도 증가)")
    print("  Phase 3.25:  로봇 1개 + 움직이는 장애물 (STAY 학습!) ✨")
    print("  Phase 3.5:   로봇 2개 (장애물 없음 - 협력 학습)")
    print("  Phase 4:     로봇 2개 + 정적 장애물 (종합)")
    print("=" * 70)
    
    # DQN 에이전트 생성 (한 번만!)
    agent = DQNAgent(
        state_dim=13,
        action_dim=4,  # 3 → 4 (STAY 추가)
        learning_rate=0.0005,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        use_target_net=True,
        device="cpu"
    )
    
    # 트레이너 생성
    trainer = SimpleCurriculumTrainer(
        agent=agent,
        batch_size=128,
        log_interval=100
    )
    
    # Phase 0: 장애물 없음 (기본 이동 학습)
    try:
        phase0_stats, phase0_success = trainer.train_phase(
            phase_name="Phase 0: 장애물 없음",
            obstacles=None,
            num_episodes=20000,  # 15000 → 20000 증가
            termination_time=80,
            success_threshold=0.3,  # 50% → 30% 낮춤 (현실적)
            model_path="outputs/curriculum_simple_phase0.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    if phase0_success < 0.15:  # 20% → 15% 낮춤
        print("\n❌ Phase 0 실패! 기본 이동조차 학습하지 못했습니다.")
        print("   하이퍼파라미터를 재조정하거나 에피소드 수를 늘려야 합니다.")
        return
    
    # Epsilon 재조정 (새로운 상황 탐험)
    agent.epsilon = 0.5
    print(f"\n🔄 Phase 1을 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 1: 장애물 1개 (모서리)
    try:
        phase1_stats, phase1_success = trainer.train_phase(
            phase_name="Phase 1: 장애물 1개 (모서리)",
            obstacles=[(2, 2)],  # 모서리
            num_episodes=12000,  # 10000 → 12000 증가
            termination_time=80,
            success_threshold=0.2,  # 30% → 20% 낮춤
            model_path="outputs/curriculum_simple_phase1.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    if phase1_success < 0.08:  # 10% → 8% 낮춤
        print("\n⚠️ Phase 1 성공률 낮음. Phase 2는 건너뜁니다.")
        return
    
    # Epsilon 재조정
    agent.epsilon = 0.4
    print(f"\n🔄 Phase 2를 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 2: 장애물 1개 (중앙 근처)
    try:
        phase2_stats, phase2_success = trainer.train_phase(
            phase_name="Phase 2: 장애물 1개 (중앙 근처)",
            obstacles=[(0, 1)],  # 원래 어려웠던 위치
            num_episodes=15000,  # 10000 → 15000 증가
            termination_time=80,
            success_threshold=0.15,  # 20% → 15% 낮춤
            model_path="outputs/curriculum_simple_phase2.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    # Epsilon 재조정
    agent.epsilon = 0.3
    print(f"\n🔄 Phase 3을 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 3: 장애물 3개
    try:
        phase3_stats, phase3_success = trainer.train_phase(
            phase_name="Phase 3: 장애물 3개",
            obstacles=[(0, 1), (-1, -1), (1, 0)],
            num_episodes=30000,  # 20000 → 30000 증가 (더 충분히 학습)
            termination_time=80,
            success_threshold=0.15,  # 10% → 15% 상향 (충분히 달성 가능)
            model_path="outputs/curriculum_simple_phase3.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    # Epsilon 재조정
    agent.epsilon = 0.35
    print(f"\n🔄 Phase 3.25를 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 3.25: 로봇 1개 + 움직이는 장애물 (STAY 학습!)
    moving_obs = create_horizontal_obstacle(y=0, speed=1)  # 중앙 라인을 왕복
    try:
        phase325_stats, phase325_success = trainer.train_phase(
            phase_name="Phase 3.25: 움직이는 장애물 (STAY 학습!)",
            obstacles=None,  # 정적 장애물 없음
            moving_obstacles=[moving_obs],  # 움직이는 장애물 1개
            num_episodes=25000,  # 충분한 학습
            termination_time=100,  # 움직이는 장애물 대응 시간 필요
            success_threshold=0.15,  # 15% 이상
            model_path="outputs/curriculum_simple_phase3.25.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    if phase325_success < 0.08:  # 8% 미만이면
        print("\n⚠️ Phase 3.25 성공률 낮음. 그래도 Phase 3.5로 진행합니다.")
        # STAY를 배웠다면 괜찮으므로 계속 진행
    
    # Epsilon 재조정
    agent.epsilon = 0.4
    print(f"\n🔄 Phase 3.5를 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 3.5: 로봇 2개, 장애물 없음 (다중 로봇 협력 기초)
    try:
        phase35_stats, phase35_success = trainer.train_phase(
            phase_name="Phase 3.5: 로봇 2개 (장애물 없음)",
            obstacles=None,  # 장애물 없음 (협력 학습에 집중)
            num_episodes=30000,
            termination_time=100,  # 로봇 2개라 시간 더 필요
            success_threshold=0.15,  # 15% 이상
            model_path="outputs/curriculum_simple_phase3.5.pth",
            num_robots=2  # ← 로봇 2개!
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    if phase35_success < 0.05:  # 5% 미만이면
        print("\n⚠️ Phase 3.5 성공률 낮음. Phase 4는 건너뜁니다.")
        return
    
    # Epsilon 재조정
    agent.epsilon = 0.3
    print(f"\n🔄 Phase 4를 위해 Epsilon 재설정: {agent.epsilon}")
    
    # Phase 4: 로봇 2개, 장애물 1개 (STAY 학습!)
    try:
        phase4_stats, phase4_success = trainer.train_phase(
            phase_name="Phase 4: 로봇 2개 + 장애물 (STAY 활용)",
            obstacles=[(2, 2)],  # 모서리 1개 (쉬운 위치)
            num_episodes=30000,  # 25000 → 30000
            termination_time=120,  # 100 → 120 (더 충분한 시간)
            success_threshold=0.1,  # 10% 이상
            model_path="outputs/curriculum_simple_phase4.pth",
            num_robots=2  # ← 로봇 2개!
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("🎉 Curriculum Learning 완료!")
    print("=" * 70)
    print(f"Phase 0   (1개, 장애물 없음):   {phase0_success*100:5.1f}%")
    print(f"Phase 1   (1개, 모서리 1개):    {phase1_success*100:5.1f}%")
    print(f"Phase 2   (1개, 중앙 1개):      {phase2_success*100:5.1f}%")
    print(f"Phase 3   (1개, 장애물 3개):    {phase3_success*100:5.1f}%")
    print(f"Phase 3.5 (2개, 장애물 없음):   {phase35_success*100:5.1f}%")
    print(f"Phase 4   (2개, 모서리 1개):    {phase4_success*100:5.1f}%")
    print("=" * 70)
    print("\n저장된 모델:")
    print("  outputs/curriculum_simple_phase0.pth")
    print("  outputs/curriculum_simple_phase1.pth")
    print("  outputs/curriculum_simple_phase2.pth")
    print("  outputs/curriculum_simple_phase3.pth")
    print("  outputs/curriculum_simple_phase3.5.pth  ← 로봇 2개 (장애물 없음)")
    print("  outputs/curriculum_simple_phase4.pth    ← 로봇 2개 + 장애물!")
    print("\n평가 명령어:")
    print("  python3.11 evaluate.py --model outputs/curriculum_simple_phase3.5.pth --num-robots 2")
    print("  python3.11 evaluate.py --model outputs/curriculum_simple_phase4.pth --num-robots 2 --obstacles '(2,2)'")
    print("=" * 70)

if __name__ == "__main__":
    main()

