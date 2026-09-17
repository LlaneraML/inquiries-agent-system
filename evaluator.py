import time
import csv
import requests

API_URL = "http://127.0.0.1:8000/chat"

TEST_SUITE = [
    {"query": "hello", "expected_source": "rule_based"},
    {"query": "contact", "expected_source": "rule_based"},
    {"query": "portal", "expected_source": "rule_based"},
    {"query": "Who is Dr. Elena Rostova?", "expected_source": "rag_hybrid_gemini"},
    {"query": "Where do I apply for a scholarship?", "expected_source": "fallback_router"},
]

def run_benchmark():
    print("🚀 Running ISO/IEC 25010 Evaluation Suite...")
    results = []

    for item in TEST_SUITE:
        start = time.time()
        try:
            res = requests.post(API_URL, json={"message": item["query"], "session_id": "eval_test"}).json()
            latency = res.get("response_time_seconds", round(time.time() - start, 4))
            source = res.get("source", "unknown")
            is_accurate = (source == item["expected_source"])

            results.append({
                "Query": item["query"],
                "Expected Source": item["expected_source"],
                "Actual Source": source,
                "Accuracy Match": is_accurate,
                "Latency (s)": latency
            })
        except Exception as e:
            print(f"Error testing query '{item['query']}': {e}")

    with open("iso_25010_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Expected Source", "Actual Source", "Accuracy Match", "Latency (s)"])
        writer.writeheader()
        writer.writerows(results)

    print("✅ Benchmark complete. Saved to 'iso_25010_benchmark_results.csv'.")

if __name__ == "__main__":
    run_benchmark()