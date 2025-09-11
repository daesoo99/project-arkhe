# -*- coding: utf-8 -*-
"""
Project Arkhe - 간단한 벤치마크 러너
다른 AI가 추천한 코드를 기반으로 한 최소 의존성 벤치마크
"""
import os, time, json, csv, uuid, argparse, requests, sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 강화된 채점 시스템 임포트
from src.utils.scorers import score_task

def env(k, d=None):
    return os.environ.get(k, d)

OLLAMA = env("OLLAMA_HOST", "http://127.0.0.1:11434")
TEMPERATURE = float(env("ARKHE_TEMP", "0.2"))  # Windows TEMP 환경변수와 충돌 방지
MAXTOK = int(env("MAX_TOKENS", "512"))
TIMEOUT= int(env("TIMEOUT_S", "120"))

def ask_ollama(model: str, prompt: str):
    """Ollama API 호출 및 스트리밍 응답 파싱"""
    t0 = time.time()
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": prompt,
              "options":{"temperature": TEMPERATURE, "num_predict": MAXTOK}},
        timeout=TIMEOUT,
    )
    t1 = time.time()
    # 스트리밍 라인 파싱
    merged, last_meta = [], {}
    for line in r.text.splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        if "response" in obj: merged.append(obj["response"])
        last_meta = obj
    resp = "".join(merged)
    return {
        "response": resp,
        "latency_ms": int((t1 - t0)*1000),
        "eval_count": last_meta.get("eval_count"),
        "eval_ms": int(last_meta.get("eval_duration",0)/1e6),
        "load_ms": int(last_meta.get("load_duration",0)/1e6),
        "total_ms": int(last_meta.get("total_duration",0)/1e6),
    }

def score(example: dict, out: dict):
    """강화된 채점 시스템 - 태스크별 전문 채점기 사용"""
    task_type = example.get("type", "fact")
    response = out["response"]
    ground_truth = example.get("answer", "")
    
    # 강화된 채점기 사용
    try:
        result = score_task(task_type, ground_truth, response)
        accuracy = result.get("score", 0)
        method = result.get("method", "unknown")
        details = result.get("details", "")
        
        # 형식 검증 (format 태스크만)
        ok_format = True
        if task_type == "format":
            ok_format = accuracy > 0.5  # 50% 이상 맞으면 형식 OK
        
        return {
            "ok_format": ok_format, 
            "accuracy": accuracy,
            "score_method": method,
            "score_details": details
        }
        
    except Exception as e:
        # 폴백: 기존 간단한 채점 방식
        print(f"    채점 오류, 기본 방식 사용: {e}")
        return score_simple_fallback(example, out)

def score_simple_fallback(example: dict, out: dict):
    """기본 채점 방식 (폴백)"""
    t = example.get("type")
    resp = out["response"]
    expected = example.get("answer", "")
    
    ok_format = True
    acc = 0
    
    if t == "format" and expected:
        try:
            obj = json.loads(resp.strip())
            expected_obj = json.loads(expected)
            ok_format = isinstance(obj, dict)
            acc = 1 if obj == expected_obj else 0
        except:
            ok_format = False
            acc = 0
    elif expected:
        # 간단한 포함 여부 체크
        acc = 1 if expected.lower() in resp.lower() else 0
    
    return {
        "ok_format": ok_format, 
        "accuracy": acc,
        "score_method": "simple_fallback",
        "score_details": "Basic keyword matching"
    }

def load_tasks(path: str):
    """JSONL 태스크 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str,
                    default="gemma:2b")  # 기본적으로 설치된 모델만
    ap.add_argument("--tasks", type=str, default=os.path.join("prompts","tasks.jsonl"))
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--limit", type=int, default=5, help="처리할 태스크 수 제한")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
    out_csv = os.path.join(args.outdir, f"bench_{run_id}.csv")

    print(f"Project Arkhe 벤치마크 시작")
    print(f"모델: {models}")
    print(f"태스크 파일: {args.tasks}")
    print(f"결과 파일: {out_csv}")
    print("="*60)

    with open(out_csv, "w", encoding="utf-8", newline="") as w:
        wr = csv.writer(w)
        wr.writerow(["run_id","model","task_id","type","complexity","latency_ms","eval_ms","total_ms","ok_format","accuracy","score_method","score_details","resp_head"])
        
        task_count = 0
        for ex in load_tasks(args.tasks):
            if task_count >= args.limit:
                break
            
            print(f"\\n[{ex.get('type')}] {ex.get('id')}: {ex.get('prompt', '')[:60]}...")
            
            for m in models:
                try:
                    print(f"  -> {m} 실행중...")
                    out = ask_ollama(m, ex["prompt"])
                    sc = score(ex, out)
                    
                    print(f"     응답: {out['response'][:80]}...")
                    print(f"     정확도: {sc.get('accuracy', 'N/A'):.3f}, 형식: {sc.get('ok_format')}")
                    print(f"     채점방식: {sc.get('score_method', 'unknown')}")
                    print(f"     시간: {out.get('total_ms')}ms")
                    
                except Exception as e:
                    print(f"     오류: {e}")
                    out = {"response": f"[ERROR] {e}", "latency_ms": -1, "eval_ms": -1, "total_ms": -1}
                    sc  = {"ok_format": False, "accuracy": 0, "score_method": "error", "score_details": str(e)}
                
                head = (out["response"] or "").replace("\\n"," ")[:120]
                wr.writerow([
                    run_id, m, ex.get("id"), ex.get("type"), ex.get("complexity"),
                    out.get("latency_ms"), out.get("eval_ms"), out.get("total_ms"),
                    sc.get("ok_format"), sc.get("accuracy"), 
                    sc.get("score_method", "unknown"), sc.get("score_details", ""),
                    head
                ])
            
            task_count += 1

    print(f"\\n벤치마크 완료! 결과: {out_csv}")
    
    # 간단한 결과 요약
    print("\\n=== 결과 요약 ===")
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        if model_results:
            accuracies = [float(r["accuracy"]) for r in model_results if r["accuracy"] and r["accuracy"] != "None"]
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            avg_time = sum(int(r["total_ms"]) for r in model_results if r["total_ms"] and int(r["total_ms"]) > 0) / len(model_results)
            
            print(f"{model}: 정확도 {avg_acc:.2%}, 평균시간 {avg_time:.0f}ms")

if __name__ == "__main__":
    main()