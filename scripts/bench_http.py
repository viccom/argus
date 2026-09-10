#!/usr/bin/env python3
"""对称 HTTP 压测：对单个端点做闭环固定并发测量(改自 yolo-rs scripts/bench_http.py)。

与原版差异：wall-time QPS(而非 sum/c 推导)、错误计数不中断、可选鉴权头、n 随 c 扩展。
"""
import argparse
import concurrent.futures as cf
import os
import statistics
import time

import requests


def one_request(session, url, image_path, headers, expect_status):
    start = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            resp = session.post(
                url, files={"file": ("img.jpg", f, "image/jpeg")},
                headers=headers, timeout=120,
            )
        elapsed = time.perf_counter() - start
        ok = resp.status_code == 200
        if ok:
            try:
                ok = resp.json().get("status") == expect_status
            except Exception:
                ok = False
        return elapsed, ok
    except Exception:
        return time.perf_counter() - start, False


def run_level(url, image_path, headers, expect, c, n):
    latencies, errors = [], 0
    wall_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as pool:
        sessions = [requests.Session() for _ in range(c)]
        futures = [
            pool.submit(one_request, sessions[i % c], url, image_path, headers, expect)
            for i in range(n)
        ]
        for f in futures:
            lat, ok = f.result()
            if ok:
                latencies.append(lat)
            else:
                errors += 1
        for s in sessions:
            s.close()
    wall = time.perf_counter() - wall_start

    latencies.sort()

    def pct(p):
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)] if latencies else float("nan")

    return {
        "n": n, "ok": len(latencies), "errors": errors,
        "qps": len(latencies) / wall if wall > 0 else 0.0,
        "mean": statistics.fmean(latencies) if latencies else float("nan"),
        "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": latencies[-1] if latencies else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--expect-status", default="ok")
    ap.add_argument("--header", action="append", default=[])
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--warmup", type=int, default=6)
    args = ap.parse_args()

    headers = {}
    for h in args.header:
        k, _, v = h.partition(":")
        headers[k.strip()] = v.strip()

    s = requests.Session()
    for i in range(args.warmup):
        lat, ok = one_request(s, args.url, args.image, headers, args.expect_status)
        if not ok and i == 0:
            raise SystemExit(f"预热请求失败(检查服务/鉴权/端点): {lat}")
    s.close()
    print(f"目标 {args.url} | 图 {os.path.basename(args.image)} | 预热 {args.warmup} 次 | load={os.getloadavg()[0]:.2f}\n")
    print(f"{'C':>3} {'n':>4} {'ok':>4} {'err':>4} {'QPS':>7} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  (ms)")

    for c in args.concurrency:
        n = max(40, min(150, c * 12))
        r = run_level(args.url, args.image, headers, args.expect_status, c, n)
        print(
            f"{c:>3} {r['n']:>4} {r['ok']:>4} {r['errors']:>4} {r['qps']:>7.2f} "
            f"{r['mean']*1e3:>8.1f} {r['p50']*1e3:>8.1f} {r['p95']*1e3:>8.1f} "
            f"{r['p99']*1e3:>8.1f} {r['max']*1e3:>8.1f}"
        )
    print(f"\n结束 load={os.getloadavg()[0]:.2f}")


if __name__ == "__main__":
    main()
