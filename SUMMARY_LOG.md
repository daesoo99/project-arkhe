# Project Arkhē - Summary Log

[20250101-0000_initial-multi-agent] Multi-Agent 완전 실패 - Single 모델이 압도적 우위 | 50.2% vs 87.7% 정확도, 1,766 vs 152 토큰, 11배 비용 차이 | Next: 정보 비대칭 실험으로 원인 분석 (Decision: 아키텍처 재설계 필요)

[20250101-0001_information-asymmetry] 부분 공유가 최악 성능 - "Goldilocks zone" 가설 반박 | NONE/COMPLETE 80% vs PARTIAL 60% 정확도 | Next: 토큰 계산 방식 문제 해결 (Decision: 근본 아키텍처 재설계)

[20250810-1947_token-calculation-fix] 토큰 계산 비효율성 원인 규명 - 누적 프롬프트 8배 차이 확인 | 275토큰(Multi) vs 35토큰(Single) 예시 분석 완료 | Next: A/B 방안 병행 구현 (Decision: ThoughtAggregator + 프롬프트 개선)

[20250811-1100_ab-comparison] A/B 방안 성능 비교 실패 - 0.5B 모델 한계 확인 | 압축 실패, 구조화 실패로 Single 대비 심각한 성능 저하 | Next: 7B 모델 업그레이드 (Decision: 모델 성능이 핵심 제약)

[20250811-1200_pipeline-integration] 전체 파이프라인 Multi-Agent 완전 실패 - Single 압도적 우위 | Single(0.0375) vs Multi(0.0008) 효율성 47배 차이, 토큰 30-40배 낭비 | Next: 7B 모델 전환 (Decision: 구조 개선 한계, 모델 업그레이드 필수)

[20250811-1300_model-upgrade] 7B 모델 업그레이드 성공 - Multi-Agent가 Single 최초 역전 달성! | B안 80% vs Single 60% 정확도, A안은 압축 왜곡 지속 | Next: 계층적 구조 실험 (Decision: B안 베이스로 Judge 모델 업그레이드)

[20250811-1400_hierarchical-experiment] 계층적 Multi-Agent 실험 완료 - Review 단계 가치 부재 확인 | Option 1(1687토큰, 35.3초) vs Option 2(753토큰, 10.9초) vs Single(16토큰, 0.7초), 동일 정확도(100%) | Next: 복잡한 추론 문제 재검증 (Decision: 간단한 질문에서는 Single 압도적 우위)

[20250811-1500_complex-problem-test] 복잡한 문제에서도 Single Model 우위 입증 - Multi-Agent 최종 실패 | Single(67.8%, 364토큰, 24.5초) vs Multi(66.7%, 432토큰, 51.8초), 효율성 비율 0.83배 | Next: 최종 결론 도출 (Decision: 전 영역에서 Single Model 압도적 우수성 확인)

[20250912-0100_creative-collaboration] ★ Multi-Agent 진정한 우위 영역 발견! 창의성에서 압도적 승리 | 단어 다양성 2.2배(391 vs 177), 내용 풍부함 3.5배(7408 vs 2130자), 창의적 협업 ROI 압도적 | Next: Shannon Entropy 정보 이론 최적화 (Decision: 창의성/브레인스토밍 특화 전략 확립)

[20250915-1430_multiroom-conversation] ★ MultiRoom 대화 시스템 효과 입증! 원칙 충돌 상황에서 25% 개선 | 평균 준수율 +8.3%, 하드코딩 위반 감지→차단 성공, 룸별 컨텍스트 원칙 강화 효과 확인 | Next: 복잡한 대화 시나리오 확장 (Decision: 특정 상황에서 유용한 도구로 활용)