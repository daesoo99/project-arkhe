#!/usr/bin/env python3
"""
경제적 지능 실험 래퍼
run_experiment.py의 경제적 지능 부분만 실행하는 얇은 래퍼
"""

import sys
import subprocess
from pathlib import Path

def main():
    """경제적 지능 실험만 실행"""
    print("=" * 60)
    print("Project Arkhe - 경제적 지능 실험")
    print("=" * 60)
    print("동적 모델 선택 vs 고정 모델 성능 비교")
    print()
    
    # run_experiment.py 호출하되 경제적 지능 테스트만 활성화
    script_dir = Path(__file__).parent
    main_script = script_dir / "run_experiment_simple.py"
    
    if not main_script.exists():
        print(f"❌ 메인 실험 스크립트를 찾을 수 없습니다: {main_script}")
        return 1
    
    try:
        # Python subprocess로 메인 실험 실행
        result = subprocess.run([
            sys.executable, str(main_script)
        ], capture_output=True, text=True, encoding='utf-8')
        
        # 출력에서 경제적 지능 관련 부분만 필터링
        lines = result.stdout.split('\n')
        in_economic_section = False
        
        for line in lines:
            if "경제적 지능" in line:
                in_economic_section = True
            elif "통합 Arkhe" in line or "실험 결과 요약" in line:
                in_economic_section = False
            
            if in_economic_section or "비용" in line or "Cost Group" in line:
                print(line)
        
        if result.stderr:
            print("⚠️ 경고:", result.stderr)
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ 실험 실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())