#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제출 파일에서 True 개수를 200개에서 170개로 조정
===============================================
"""

import pandas as pd
import numpy as np

def modify_submission_to_170(input_file, output_file):
    """
    제출 파일에서 확률이 가장 낮은 170개만 True로 설정
    
    Args:
        input_file: 원본 제출 파일 경로
        output_file: 수정된 제출 파일 경로
    """
    
    print(f"📁 원본 파일 로딩: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"   - 총 행 수: {len(df)}")
    print(f"   - 현재 True 개수: {df['decision'].sum()}")
    
    # 모든 decision을 False로 초기화
    df['decision'] = False
    
    # L 타입과 P 타입 분리
    l_type = df[df['ID'].str.contains('_L')].copy()
    p_type = df[df['ID'].str.contains('_P')].copy()
    
    print(f"   - L 타입: {len(l_type)}개")
    print(f"   - P 타입: {len(p_type)}개")
    
    # 각 타입에서 확률이 가장 낮은 75개씩 선택 (총 170개)
    l_type_sorted = l_type.sort_values('probability')
    p_type_sorted = p_type.sort_values('probability')
    
    # 상위 75개씩 선택
    selected_l_ids = l_type_sorted.iloc[:170]['ID']
    selected_p_ids = p_type_sorted.iloc[:170]['ID']
    
    # decision을 True로 설정
    df.loc[df['ID'].isin(selected_l_ids), 'decision'] = True
    df.loc[df['ID'].isin(selected_p_ids), 'decision'] = True
    
    # 결과 저장
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 수정 완료!")
    print(f"   - 새로운 True 개수: {df['decision'].sum()}")
    print(f"   - L 타입 선택: {len(selected_l_ids)}개")
    print(f"   - P 타입 선택: {len(selected_p_ids)}개")
    print(f"   - 저장된 파일: {output_file}")
    
    # 선택된 제품들의 확률 통계
    selected_probs = df[df['decision'] == True]['probability']
    print(f"\n📊 선택된 제품들의 확률 통계:")
    print(f"   - 평균: {selected_probs.mean():.4f}")
    print(f"   - 중앙값: {selected_probs.median():.4f}")
    print(f"   - 최솟값: {selected_probs.min():.4f}")
    print(f"   - 최댓값: {selected_probs.max():.4f}")
    
    return df

if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "hybrid_submission.csv"
    output_file = "hybrid_submission_170.csv"
    
    # 수정 실행
    modified_df = modify_submission_to_170(input_file, output_file)
    
    print(f"\n🎉 완료! {output_file} 파일이 생성되었습니다.")
