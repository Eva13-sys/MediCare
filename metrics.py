import time

metrics_store = {
    "total_requests": 0,
    "total_latency": 0,
    "per_endpoint": {}
}

def log_request(latency, endpoint):
    metrics_store["total_requests"] += 1
    metrics_store["total_latency"] += latency

    if endpoint not in metrics_store["per_endpoint"]:
        metrics_store["per_endpoint"][endpoint] = 0

    metrics_store["per_endpoint"][endpoint] += 1

def get_metrics():
    avg = 0
    if metrics_store["total_requests"] > 0:
        avg = metrics_store["total_latency"] / metrics_store["total_requests"]

    return {
        "total_requests": metrics_store["total_requests"],
        "avg_latency_ms": round(avg, 2)
    }