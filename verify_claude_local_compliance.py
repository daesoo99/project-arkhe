# -*- coding: utf-8 -*-
"""
Project Arkhe - CLAUDE.local Compliance Verification
Phase 1, 2, 3 모든 구현의 CLAUDE.local 규칙 준수 검증

CLAUDE.local 핵심 원칙:
1. 하드코딩 Zero Tolerance
2. 의존성 주입 완전 적용  
3. 인터페이스 우선 설계
4. 느슨한 결합 아키텍처
"""

import sys
import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ClaudeLocalComplianceChecker:
    """CLAUDE.local 규칙 준수 검증기"""
    
    def __init__(self):
        self.violations = []
        self.compliances = []
        self.project_root = Path(__file__).parent
        
    def check_hardcoding_violations(self, file_path: Path) -> List[Dict[str, Any]]:
        """하드코딩 Zero Tolerance 검증"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # AST 파싱
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # 하드코딩된 리스트/딕셔너리 검출
                if isinstance(node, ast.List) and len(node.elts) > 3:
                    # 긴 하드코딩된 리스트 검출
                    if hasattr(node, 'lineno'):
                        violations.append({
                            'type': 'hardcoded_list',
                            'line': node.lineno,
                            'message': f'Hardcoded list with {len(node.elts)} elements'
                        })
                
                # 하드코딩된 숫자 상수 검출 (특정 패턴) - Python 3.8+ 호환
                if isinstance(node, (ast.Num, ast.Constant)):
                    if hasattr(node, 'n'):  # Python < 3.8
                        value = node.n
                    elif hasattr(node, 'value'):  # Python >= 3.8
                        value = node.value
                    else:
                        continue
                        
                    if isinstance(value, (int, float)) and value in [16, 100, 1000] and hasattr(node, 'lineno'):
                        violations.append({
                            'type': 'magic_number',
                            'line': node.lineno,
                            'message': f'Potential magic number: {value}'
                        })
                
                # if-elif 체인으로 하드코딩 검출
                if isinstance(node, ast.If):
                    elif_count = len([n for n in ast.walk(node) if isinstance(n, ast.If)]) - 1
                    if elif_count > 3:
                        violations.append({
                            'type': 'hardcoded_if_chain',
                            'line': node.lineno,
                            'message': f'Long if-elif chain ({elif_count} branches) suggests hardcoding'
                        })
            
            # 문자열 패턴 검사
            hardcoded_patterns = [
                r'if\s+\w+\s*==\s*["\'][\w_]+["\']',  # if variable == "hardcoded_string"
                r'model_name\s*=\s*["\'][\w\-_]+["\']',  # model_name = "hardcoded"
                r'\[\s*["\'][^"\']+["\']\s*,\s*["\'][^"\']+["\']\s*\]',  # ["item1", "item2"]
            ]
            
            for pattern in hardcoded_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        'type': 'hardcoded_pattern',
                        'line': line_num,
                        'message': f'Hardcoded pattern detected: {match.group()[:50]}...'
                    })
                    
        except Exception as e:
            violations.append({
                'type': 'parse_error',
                'line': 0,
                'message': f'Could not parse file: {e}'
            })
            
        return violations
    
    def check_dependency_injection(self, file_path: Path) -> List[Dict[str, Any]]:
        """의존성 주입 패턴 검증"""
        compliances = []
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 생성자 주입 패턴 검출
            constructor_injection_patterns = [
                r'def\s+__init__\s*\([^)]*config[^)]*\)',  # __init__(self, config)
                r'def\s+__init__\s*\([^)]*registry[^)]*\)',  # __init__(self, registry)
                r'def\s+__init__\s*\([^)]*engine[^)]*\)',  # __init__(self, engine)
            ]
            
            for pattern in constructor_injection_patterns:
                if re.search(pattern, content):
                    compliances.append({
                        'type': 'constructor_injection',
                        'message': 'Constructor dependency injection detected'
                    })
            
            # 팩토리 패턴 검출
            factory_patterns = [
                r'def\s+create_\w+',  # create_xxx functions
                r'def\s+get_\w+_registry',  # get_xxx_registry functions
                r'registry\.get_\w+',  # registry.get_xxx calls
            ]
            
            for pattern in factory_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    compliances.append({
                        'type': 'factory_pattern',
                        'message': f'Factory pattern usage: {len(matches)} instances'
                    })
            
            # 직접 import 대신 주입 사용 검증
            direct_imports = re.findall(r'from\s+[\w\.]+\s+import\s+\w+', content)
            relative_imports = [imp for imp in direct_imports if 'from .' in imp]
            
            if len(relative_imports) > 5:  # 너무 많은 직접 의존성
                violations.append({
                    'type': 'too_many_direct_imports',
                    'message': f'Too many direct imports ({len(relative_imports)}), consider injection'
                })
                
        except Exception as e:
            violations.append({
                'type': 'parse_error',
                'message': f'Could not check dependency injection: {e}'
            })
        
        return compliances, violations
    
    def check_interface_first_design(self, file_path: Path) -> List[Dict[str, Any]]:
        """인터페이스 우선 설계 검증"""
        compliances = []
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 추상 클래스/인터페이스 검출
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # ABC 상속 검사
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'ABC':
                            compliances.append({
                                'type': 'abstract_class',
                                'message': f'Abstract class defined: {node.name}'
                            })
                        elif isinstance(base, ast.Attribute) and base.attr == 'ABC':
                            compliances.append({
                                'type': 'abstract_class',
                                'message': f'Abstract class defined: {node.name}'
                            })
                    
                    # @abstractmethod 데코레이터 검사
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            for decorator in item.decorator_list:
                                if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                                    compliances.append({
                                        'type': 'abstract_method',
                                        'message': f'Abstract method: {node.name}.{item.name}'
                                    })
            
            # 프로토콜/타입 힌트 사용 검증
            type_hints = re.findall(r':\s*[A-Z]\w+', content)
            if type_hints:
                compliances.append({
                    'type': 'type_hints',
                    'message': f'Type hints usage: {len(type_hints)} instances'
                })
            
            # 인터페이스 명명 규칙 검증 (I로 시작)
            interface_classes = re.findall(r'class\s+(I[A-Z]\w+)', content)
            if interface_classes:
                compliances.append({
                    'type': 'interface_naming',
                    'message': f'Interface naming convention: {interface_classes}'
                })
                
        except Exception as e:
            violations.append({
                'type': 'parse_error',
                'message': f'Could not check interface design: {e}'
            })
        
        return compliances, violations
    
    def check_loose_coupling(self, file_path: Path) -> List[Dict[str, Any]]:
        """느슨한 결합 아키텍처 검증"""
        compliances = []
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 레지스트리 패턴 사용 검증
            registry_usage = len(re.findall(r'registry\.[get|register]', content))
            if registry_usage > 0:
                compliances.append({
                    'type': 'registry_pattern',
                    'message': f'Registry pattern usage: {registry_usage} calls'
                })
            
            # 설정 기반 동작 검증
            config_usage = len(re.findall(r'config\.get\(', content))
            if config_usage > 0:
                compliances.append({
                    'type': 'configuration_driven',
                    'message': f'Configuration-driven behavior: {config_usage} usages'
                })
            
            # 직접 클래스 인스턴스화 검출 (결합도 높음)
            direct_instantiation = re.findall(r'\w+\s*=\s*[A-Z]\w+\(', content)
            if len(direct_instantiation) > 3:
                violations.append({
                    'type': 'tight_coupling',
                    'message': f'Direct class instantiation detected: {len(direct_instantiation)} instances'
                })
            
            # 싱글톤 패턴 검증
            singleton_patterns = re.findall(r'_global_\w+|get_\w+_registry', content)
            if singleton_patterns:
                compliances.append({
                    'type': 'singleton_pattern',
                    'message': f'Singleton pattern usage: {len(singleton_patterns)} instances'
                })
                
        except Exception as e:
            violations.append({
                'type': 'parse_error',
                'message': f'Could not check coupling: {e}'
            })
        
        return compliances, violations
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """개별 파일 분석"""
        result = {
            'file': str(file_path.relative_to(self.project_root)),
            'hardcoding_violations': [],
            'dependency_injection': {'compliances': [], 'violations': []},
            'interface_design': {'compliances': [], 'violations': []},
            'loose_coupling': {'compliances': [], 'violations': []}
        }
        
        # 1. 하드코딩 검사
        result['hardcoding_violations'] = self.check_hardcoding_violations(file_path)
        
        # 2. 의존성 주입 검사
        di_compliances, di_violations = self.check_dependency_injection(file_path)
        result['dependency_injection']['compliances'] = di_compliances
        result['dependency_injection']['violations'] = di_violations
        
        # 3. 인터페이스 설계 검사
        if_compliances, if_violations = self.check_interface_first_design(file_path)
        result['interface_design']['compliances'] = if_compliances
        result['interface_design']['violations'] = if_violations
        
        # 4. 느슨한 결합 검사
        lc_compliances, lc_violations = self.check_loose_coupling(file_path)
        result['loose_coupling']['compliances'] = lc_compliances
        result['loose_coupling']['violations'] = lc_violations
        
        return result
    
    def analyze_phases(self) -> Dict[str, Any]:
        """Phase 1, 2, 3 전체 분석"""
        
        # Phase별 주요 파일들
        phase_files = {
            'Phase 1 - Model Registry': [
                'src/registry/model_registry.py',
                'tests/basic_model_test.py',
                'tests/simple_hierarchical_test.py', 
                'tests/benchmark_comparison.py'
            ],
            'Phase 2 - Experiment Framework': [
                'src/registry/experiment_registry.py',
                'config/experiments.yaml'
            ],
            'Phase 3 - Plugin System': [
                'src/plugins/interfaces.py',
                'src/plugins/registry.py',
                'src/plugins/builtin/legacy_scorers.py',
                'src/plugins/builtin/standard_aggregators.py',
                'src/evaluation/plugin_engine.py',
                'src/integration/plugin_experiment_adapter.py',
                'config/plugin_config.json'
            ]
        }
        
        analysis_results = {}
        
        for phase_name, files in phase_files.items():
            print(f"\n=== {phase_name} 분석 중... ===")
            phase_results = []
            
            for file_path_str in files:
                file_path = self.project_root / file_path_str
                
                if file_path.exists() and file_path.suffix == '.py':
                    print(f"분석 중: {file_path_str}")
                    result = self.analyze_file(file_path)
                    phase_results.append(result)
                elif file_path.exists() and file_path.suffix in ['.yaml', '.json']:
                    # 설정 파일은 별도 검사
                    print(f"설정 파일 확인: {file_path_str}")
                    phase_results.append({
                        'file': file_path_str,
                        'type': 'config_file',
                        'status': 'exists'
                    })
                else:
                    print(f"파일 없음: {file_path_str}")
            
            analysis_results[phase_name] = phase_results
        
        return analysis_results
    
    def generate_compliance_report(self, analysis_results: Dict[str, Any]) -> str:
        """CLAUDE.local 준수 보고서 생성"""
        
        report = []
        report.append("# CLAUDE.local 규칙 준수 검증 보고서")
        report.append("=" * 60)
        report.append("")
        
        total_files = 0
        total_violations = 0
        total_compliances = 0
        
        for phase_name, phase_results in analysis_results.items():
            report.append(f"## {phase_name}")
            report.append("")
            
            phase_violations = 0
            phase_compliances = 0
            
            for result in phase_results:
                if result.get('type') == 'config_file':
                    report.append(f"[CONFIG] {result['file']} (설정 파일)")
                    continue
                    
                total_files += 1
                file_violations = (
                    len(result['hardcoding_violations']) +
                    len(result['dependency_injection']['violations']) +
                    len(result['interface_design']['violations']) +
                    len(result['loose_coupling']['violations'])
                )
                
                file_compliances = (
                    len(result['dependency_injection']['compliances']) +
                    len(result['interface_design']['compliances']) +
                    len(result['loose_coupling']['compliances'])
                )
                
                phase_violations += file_violations
                phase_compliances += file_compliances
                
                status = "[OK] GOOD" if file_violations == 0 else f"[WARN] {file_violations} issues"
                report.append(f"{status} {result['file']}")
                
                # 상세 내용
                if file_violations > 0:
                    if result['hardcoding_violations']:
                        report.append(f"  - 하드코딩: {len(result['hardcoding_violations'])}개")
                    if result['dependency_injection']['violations']:
                        report.append(f"  - 의존성 주입: {len(result['dependency_injection']['violations'])}개")
                    if result['interface_design']['violations']:
                        report.append(f"  - 인터페이스 설계: {len(result['interface_design']['violations'])}개")
                    if result['loose_coupling']['violations']:
                        report.append(f"  - 결합도: {len(result['loose_coupling']['violations'])}개")
                
                if file_compliances > 0:
                    report.append(f"  [GOOD] 준수사항: {file_compliances}개")
            
            total_violations += phase_violations
            total_compliances += phase_compliances
            
            report.append(f"**{phase_name} 요약**: {phase_violations}개 위반, {phase_compliances}개 준수")
            report.append("")
        
        # 전체 요약
        report.append("## [SUMMARY] 전체 요약")
        report.append("")
        report.append(f"- **분석 파일**: {total_files}개")
        report.append(f"- **CLAUDE.local 위반**: {total_violations}개")
        report.append(f"- **CLAUDE.local 준수**: {total_compliances}개")
        
        compliance_rate = total_compliances / (total_compliances + total_violations) if (total_compliances + total_violations) > 0 else 0
        report.append(f"- **준수율**: {compliance_rate:.1%}")
        
        if total_violations == 0:
            report.append("")
            report.append("[SUCCESS] **CLAUDE.local 규칙 완전 준수!**")
        elif total_violations < 5:
            report.append("")
            report.append("[GOOD] **CLAUDE.local 규칙 대부분 준수** (미세 조정 필요)")
        else:
            report.append("")
            report.append("[WARN] **CLAUDE.local 규칙 개선 필요**")
        
        return "\n".join(report)

def main():
    """메인 검증 실행"""
    print("CLAUDE.local 규칙 준수 검증 시작...")
    print("=" * 60)
    
    checker = ClaudeLocalComplianceChecker()
    
    # Phase 1, 2, 3 분석
    analysis_results = checker.analyze_phases()
    
    # 보고서 생성
    report = checker.generate_compliance_report(analysis_results)
    
    # 결과 출력
    print("\n" + report)
    
    # 보고서 파일 저장
    report_path = Path(__file__).parent / "docs" / "CLAUDE_LOCAL_COMPLIANCE_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[REPORT] 상세 보고서 저장: {report_path}")

if __name__ == "__main__":
    main()