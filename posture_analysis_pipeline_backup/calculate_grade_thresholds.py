#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
등급 기준값 계산 스크립트
기존 CSV 데이터에서 실시간 등급 분류에 필요한 기준값들을 계산합니다.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def calculate_grade_thresholds(csv_path: str):
    """CSV 파일에서 등급 분류 기준값들을 계산"""
    
    print(f"📊 기준값 계산을 시작합니다...")
    print(f"📁 입력 파일: {csv_path}")
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_path)
        print(f"📈 로드된 데이터: {len(df)}행")
    except Exception as e:
        print(f"❌ CSV 파일 읽기 오류: {e}")
        return None
    
    # CVA 각도 데이터 추출
    cva_angles = df['cva_angle'].dropna().values
    if len(cva_angles) == 0:
        print("❌ CVA 각도 데이터가 없습니다.")
        return None
    
    # 절댓값 기준으로 계산
    abs_angles = np.abs(cva_angles)
    min_abs = abs_angles.min()
    max_abs = abs_angles.max()
    angle_range = max_abs - min_abs
    
    print(f"📐 CVA 각도 분석:")
    print(f"  - 원본 범위: {cva_angles.min():.2f}° ~ {cva_angles.max():.2f}°")
    print(f"  - 절댓값 범위: {min_abs:.2f}° ~ {max_abs:.2f}°")
    
    # 10단계로 나누기
    if angle_range == 0:
        stages = np.ones_like(abs_angles, dtype=int)
    else:
        stages = ((abs_angles - min_abs) / angle_range * 9 + 1).astype(int)
        stages = np.clip(stages, 1, 10)
    
    # 1단계에 해당하는 각도들
    stage1_angles = abs_angles[stages == 1]
    stage1_threshold = np.percentile(stage1_angles, 50) if len(stage1_angles) > 0 else min_abs
    
    # 결과 출력
    print(f"\n✅ 기준값 계산 완료!")
    print(f"📊 계산된 기준값:")
    print(f"  - min_abs: {min_abs:.6f}")
    print(f"  - max_abs: {max_abs:.6f}")
    print(f"  - stage1_threshold: {stage1_threshold:.6f}")
    print(f"  - 1단계 데이터 개수: {len(stage1_angles)}개")
    
    return {
        'min_abs': min_abs,
        'max_abs': max_abs,
        'stage1_threshold': stage1_threshold,
        'stage1_count': len(stage1_angles)
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='등급 기준값 계산 스크립트')
    parser.add_argument('--csv', '-c', required=True,
                       help='입력 CSV 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.csv).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
        return 1
    
    # 기준값 계산
    thresholds = calculate_grade_thresholds(args.csv)
    
    if thresholds is None:
        print("❌ 기준값 계산 실패")
        return 1
    
    print(f"\n🎉 기준값 계산이 완료되었습니다!")
    print(f"💡 이 값들을 실시간 자세 분석 코드에 사용하세요.")
    return 0


if __name__ == "__main__":
    exit(main()) 