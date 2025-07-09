#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자세 등급 분류 스크립트

CVA 각도를 기준으로 10단계 등급을 분류합니다.
- 1-5단계: A등급 (가장 바른 자세)
- 6-10단계: B등급 (보통 자세)  
- 11-15단계: C등급 (나쁜 자세)

# 실행 방법
# python3 posture_grade_classifier.py --csv side_angle_analysis.csv --side right
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime


class PostureGradeClassifier:
    """자세 등급 분류 클래스"""
    
    def __init__(self):
        """초기화"""
        # 등급별 분류 기준 (CVA 각도 기준)
        self.grade_criteria = {
            'A': {'range': (1, 1), 'description': 'A등급 (가장 바른 자세 - 1단계)'},
            'B': {'range': (2, 5), 'description': 'B등급 (보통 자세 - 2-5단계)'},
            'C': {'range': (6, 10), 'description': 'C등급 (나쁜 자세)'}
        }
        
        # 단계별 설명
        self.stage_descriptions = {
            1: "1단계 - A등급 (최고 자세)",
            2: "2단계 - B등급 (매우 좋은 자세)",
            3: "3단계 - B등급 (좋은 자세)",
            4: "4단계 - B등급 (양호한 자세)",
            5: "5단계 - B등급 (보통 자세)",
            6: "6단계 - C등급 (보통 이하 자세)",
            7: "7단계 - C등급 (개선 필요 자세)",
            8: "8단계 - C등급 (나쁜 자세)",
            9: "9단계 - C등급 (매우 나쁜 자세)",
            10: "10단계 - C등급 (최악의 자세)"
        }
    
    def classify_cva_angles(self, cva_angles: List[float]) -> Tuple[List[int], List[str], Dict]:
        """
        CVA 각도들을 10단계로 분류 (수치 구간 기준)
        
        Args:
            cva_angles: CVA 각도 리스트
            
        Returns:
            (단계 리스트, 등급 리스트, 통계 딕셔너리)
        """
        if not cva_angles:
            return [], [], {}
        
        # 절댓값 기준으로 10단계 수치 구간 분할
        abs_angles = [abs(angle) for angle in cva_angles]
        min_abs = min(abs_angles)
        max_abs = max(abs_angles)
        
        print(f"🔍 CVA 각도 분석:")
        print(f"  - 원본 범위: {min(cva_angles):.2f}° ~ {max(cva_angles):.2f}°")
        print(f"  - 절댓값 범위: {min_abs:.2f}° ~ {max_abs:.2f}°")
        
        # 10단계 수치 구간 계산
        angle_range = max_abs - min_abs
        if angle_range == 0:
            # 모든 각도가 같으면 모두 1단계
            stages = [1] * len(cva_angles)
            grades = ['A'] * len(cva_angles)
        else:
            stages = []
            
            for abs_angle in abs_angles:
                # 0-1 범위로 정규화
                normalized = (abs_angle - min_abs) / angle_range
                # 1-10 단계로 변환
                stage = int(normalized * 9) + 1
                stage = max(1, min(10, stage))  # 1-10 범위 제한
                stages.append(stage)
            
            # 등급 결정
            grades = []
            for i, abs_angle in enumerate(abs_angles):
                stage = stages[i]
                
                # 단계별 등급 결정 (1단계: A, 2-5단계: B, 6-10단계: C)
                if stage == 1:  # 1단계: A등급 (가장 바른 자세)
                    grade = 'A'
                elif stage <= 5:  # 2-5단계: B등급 (보통 자세)
                    grade = 'B'
                else:  # 6-10단계: C등급 (나쁜 자세)
                    grade = 'C'
                grades.append(grade)
        
        # 통계 계산
        stats = {
            'total_frames': len(cva_angles),
            'min_angle': min(cva_angles),
            'max_angle': max(cva_angles),
            'min_abs_angle': min_abs,
            'max_abs_angle': max_abs,
            'mean_angle': np.mean(cva_angles),
            'std_angle': np.std(cva_angles),
            'stage_distribution': {},
            'grade_distribution': {}
        }
        
        # 단계별 분포
        for stage in range(1, 11):
            count = stages.count(stage)
            stats['stage_distribution'][stage] = {
                'count': count,
                'percentage': (count / len(stages)) * 100,
                'description': self.stage_descriptions[stage]
            }
        
        # 등급별 분포
        for grade in ['A', 'B', 'C']:
            count = grades.count(grade)
            stats['grade_distribution'][grade] = {
                'count': count,
                'percentage': (count / len(grades)) * 100,
                'description': self.grade_criteria[grade]['description']
            }
        
        return stages, grades, stats
    
    def analyze_csv(self, csv_path: str, side: str = 'right', output_dir: str = None) -> pd.DataFrame:
        """
        CSV 파일 분석 및 등급 분류
        
        Args:
            csv_path: 입력 CSV 파일 경로
            side: 측면 ('right' 또는 'left')
            output_dir: 출력 디렉토리
            
        Returns:
            분석 결과 DataFrame
        """
        print(f"🎯 자세 등급 분류를 시작합니다...")
        print(f"📁 입력 파일: {csv_path}")
        print(f"📐 측면: {side}")
        
        # CSV 파일 읽기
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 로드된 데이터: {len(df)}행, {len(df.columns)}컬럼")
        except Exception as e:
            print(f"❌ CSV 파일 읽기 오류: {e}")
            return pd.DataFrame()
        
        # CVA 각도 데이터 추출
        cva_angles = df['cva_angle'].dropna().tolist()
        
        if not cva_angles:
            print("❌ CVA 각도 데이터가 없습니다.")
            return pd.DataFrame()
        
        print(f"📐 분석할 CVA 각도: {len(cva_angles)}개")
        print(f"  - 최소: {min(cva_angles):.2f}°")
        print(f"  - 최대: {max(cva_angles):.2f}°")
        print(f"  - 평균: {np.mean(cva_angles):.2f}°")
        
        # 등급 분류
        stages, grades, stats = self.classify_cva_angles(cva_angles)
        
        # 결과 DataFrame 생성
        result_df = df.copy()
        
        # NaN이 아닌 CVA 각도에 대해서만 등급 추가
        valid_indices = df['cva_angle'].notna()
        result_df.loc[valid_indices, 'posture_stage'] = stages
        result_df.loc[valid_indices, 'posture_grade'] = grades
        
        # NaN인 경우 빈 값으로 설정
        result_df['posture_stage'] = result_df['posture_stage'].fillna('')
        result_df['posture_grade'] = result_df['posture_grade'].fillna('')
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path("data/posture_grades")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 출력
        print(f"\n✅ 등급 분류 완료!")
        print(f"📊 처리된 프레임: {len(cva_angles)}개")
        
        # 단계별 통계
        print(f"\n📈 단계별 분포:")
        for stage in range(1, 11):
            stage_info = stats['stage_distribution'][stage]
            print(f"  - {stage}단계: {stage_info['count']}개 ({stage_info['percentage']:.1f}%) - {stage_info['description']}")
        
        # 등급별 통계
        print(f"\n🏆 등급별 분포:")
        for grade in ['A', 'B', 'C']:
            grade_info = stats['grade_distribution'][grade]
            print(f"  - {grade}등급: {grade_info['count']}개 ({grade_info['percentage']:.1f}%) - {grade_info['description']}")
        
        # 파일 저장
        output_filename = f"posture_grades_{side}.csv"
        csv_output_path = output_dir / output_filename
        result_df.to_csv(csv_output_path, index=False, encoding='utf-8')
        print(f"\n📄 분석 결과 저장: {csv_output_path}")
        
        # JSON 통계 파일 저장
        json_output_path = output_dir / f"posture_grades_{side}_stats.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📄 통계 결과 저장: {json_output_path}")
        
        # 상세 분석 리포트 생성
        self.generate_report(stats, side, output_dir)
        
        return result_df
    
    def generate_report(self, stats: Dict, side: str, output_dir: Path):
        """상세 분석 리포트 생성"""
        report_path = output_dir / f"posture_analysis_report_{side}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"자세 분석 리포트 - {side.upper()} 측면\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📊 기본 통계:\n")
            f.write(f"  - 총 프레임: {stats['total_frames']}개\n")
            f.write(f"  - CVA 각도 범위: {stats['min_angle']:.2f}° ~ {stats['max_angle']:.2f}°\n")
            f.write(f"  - 평균 CVA: {stats['mean_angle']:.2f}°\n")
            f.write(f"  - 표준편차: {stats['std_angle']:.2f}°\n\n")
            
            f.write(f"📈 단계별 상세 분석:\n")
            for stage in range(1, 11):
                stage_info = stats['stage_distribution'][stage]
                f.write(f"  {stage:2d}단계: {stage_info['count']:2d}개 ({stage_info['percentage']:5.1f}%) - {stage_info['description']}\n")
            
            f.write(f"\n🏆 등급별 요약:\n")
            for grade in ['A', 'B', 'C']:
                grade_info = stats['grade_distribution'][grade]
                f.write(f"  {grade}등급: {grade_info['count']:2d}개 ({grade_info['percentage']:5.1f}%) - {grade_info['description']}\n")
            
            f.write(f"\n💡 분석 의견:\n")
            a_percentage = stats['grade_distribution']['A']['percentage']
            if a_percentage >= 60:
                f.write(f"  - A등급 비율이 {a_percentage:.1f}%로 매우 좋은 자세를 보여줍니다.\n")
            elif a_percentage >= 40:
                f.write(f"  - A등급 비율이 {a_percentage:.1f}%로 양호한 자세입니다.\n")
            else:
                f.write(f"  - A등급 비율이 {a_percentage:.1f}%로 자세 개선이 필요합니다.\n")
        
        print(f"📄 상세 리포트 저장: {report_path}")

    def get_grade_for_angle(self, cva_angle: float, min_abs: float, max_abs: float, stage1_threshold: float) -> str:
        """
        단일 CVA 각도에 대한 등급을 실시간으로 반환
        
        Args:
            cva_angle: CVA 각도
            min_abs: 절댓값 최소값 (기준값)
            max_abs: 절댓값 최대값 (기준값)
            stage1_threshold: 1단계 중간값 (기준값)
            
        Returns:
            등급 ('A', 'B', 'C')
        """
        abs_angle = abs(cva_angle)
        angle_range = max_abs - min_abs
        
        if angle_range == 0:
            stage = 1
        else:
            # 0-1 범위로 정규화
            normalized = (abs_angle - min_abs) / angle_range
            # 1-10 단계로 변환
            stage = int(normalized * 9) + 1
            stage = max(1, min(10, stage))  # 1-10 범위 제한
        
        # 단계별 등급 결정 (1단계: A, 2-5단계: B, 6-10단계: C)
        if stage == 1:  # 1단계: A등급 (가장 바른 자세)
            return 'A'
        elif stage <= 5:  # 2-5단계: B등급 (보통 자세)
            return 'B'
        else:  # 6-10단계: C등급 (나쁜 자세)
            return 'C'


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='자세 등급 분류 스크립트')
    parser.add_argument('--csv', '-c', required=True,
                       help='입력 CSV 파일 경로')
    parser.add_argument('--side', '-s', choices=['right', 'left'], default='right',
                       help='측면 (right 또는 left)')
    parser.add_argument('--output', '-o', 
                       help='출력 디렉토리 (기본값: data/posture_grades)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.csv).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
        return 1
    
    # 분류기 생성
    classifier = PostureGradeClassifier()
    
    # 분석 실행
    result_df = classifier.analyze_csv(args.csv, args.side, args.output)
    
    if result_df.empty:
        print("❌ 분석 실패")
        return 1
    
    print(f"\n🎉 자세 등급 분류가 완료되었습니다!")
    return 0


if __name__ == "__main__":
    exit(main()) 