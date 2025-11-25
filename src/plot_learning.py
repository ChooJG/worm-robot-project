"""
학습 진행 상황 시각화 (간단 버전)

사용법:
  python3.11 plot_learning.py --log training_log.txt
"""

import argparse
import re
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """
    학습 로그 파일 파싱
    
    예상 형식:
    Ep   10/100 | Reward:  -45.2 | Steps: 12.3 | Loss: 0.0234 | ε: 0.904 | Win:   0 | Fail:   8
    
    Returns:
        dict: {'episodes': [...], 'rewards': [...], 'wins': [...], 'fails': [...]}
    """
    episodes = []
    rewards = []
    wins = []
    fails = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 정규식으로 파싱
            match = re.search(r'Ep\s+(\d+)/\d+.*Reward:\s*([-\d.]+).*Win:\s*(\d+).*Fail:\s*(\d+)', line)
            if match:
                ep = int(match.group(1))
                reward = float(match.group(2))
                win = int(match.group(3))
                fail = int(match.group(4))
                
                episodes.append(ep)
                rewards.append(reward)
                wins.append(win)
                fails.append(fail)
    
    return {
        'episodes': episodes,
        'rewards': rewards,
        'wins': wins,
        'fails': fails
    }


def plot_learning_curve(data, save_path='learning_curve.png'):
    """학습 곡선 그리기"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. 보상 그래프
    axes[0].plot(data['episodes'], data['rewards'], linewidth=1.5, color='blue', alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Learning Progress - Reward')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 승률 그래프
    win_rates = [w / (w + f) * 100 if (w + f) > 0 else 0 
                 for w, f in zip(data['wins'], data['fails'])]
    axes[1].plot(data['episodes'], win_rates, linewidth=1.5, color='green', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_title('Learning Progress - Win Rate')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 그래프 저장: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='학습 로그 시각화')
    parser.add_argument('--log', type=str, required=True, help='로그 파일 경로')
    parser.add_argument('--output', type=str, default='learning_curve.png', help='출력 이미지 경로')
    
    args = parser.parse_args()
    
    print(f"📊 로그 파일 읽는 중: {args.log}")
    data = parse_log_file(args.log)
    
    if not data['episodes']:
        print("❌ 로그 데이터를 찾을 수 없습니다.")
        return
    
    print(f"✅ {len(data['episodes'])}개 에피소드 데이터 로드")
    print(f"   최종 보상: {data['rewards'][-1]:.2f}")
    print(f"   최종 승률: {data['wins'][-1]/(data['wins'][-1]+data['fails'][-1])*100:.1f}%")
    
    plot_learning_curve(data, args.output)


if __name__ == "__main__":
    main()

