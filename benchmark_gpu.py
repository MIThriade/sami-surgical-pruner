#!/usr/bin/env python3
"""
Automated llama-server Benchmark & Validation Harness.

Tests pruned GGUF models against native llama-server with full GPU offload (-ngl 99),
measuring tokens/sec, VRAM consumption, and output coherence.
"""

import sys
import os
import time
import subprocess
import json
import urllib.request
import urllib.error

# Force UTF-8 output for Romanian diacritics
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

LLAMA_SERVER_BIN = r"tools\llama-server\llama-server.exe"

def get_vram_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "used_mb": int(parts[0]),
            "free_mb": int(parts[1]),
            "total_mb": int(parts[2]),
        }
    except Exception as e:
        return {"error": str(e)}

def wait_for_server(url="http://127.0.0.1:8080/health", timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def query_llama_server(prompt, port=8080, max_tokens=128, temperature=0.7):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - t0
            res_data = json.loads(resp.read().decode("utf-8"))
            msg = res_data["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            full_text = ""
            if reasoning:
                full_text += f"[Thinking: {reasoning.strip()}]\n\n"
            full_text += content.strip()
            
            usage = res_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            tok_per_sec = completion_tokens / max(elapsed, 0.001)
            
            return {
                "success": True,
                "text": full_text.strip(),
                "content": content.strip(),
                "reasoning": reasoning.strip(),
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed_sec": round(elapsed, 2),
                "tok_per_sec": round(tok_per_sec, 1)
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 2)
        }

def run_benchmark(model_path, ngl=99, ctx=2048, port=8080):
    print(f"\n{'='*70}")
    print(f"BENCHMARKING MODEL: {model_path}")
    print(f"GPU Offload: -ngl {ngl} | Context: {ctx} tokens | Port: {port}")
    print(f"{'='*70}\n")

    vram_before = get_vram_info()
    print(f"VRAM Before Server Start: {vram_before.get('used_mb')} MB used / {vram_before.get('total_mb')} MB total")

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", model_path,
        "-ngl", str(ngl),
        "-c", str(ctx),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-t", "4",
    ]

    print(f"Launching server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    try:
        print("Waiting for llama-server to initialize...")
        ready = wait_for_server(f"http://127.0.0.1:{port}/health", timeout=90)
        if not ready:
            print("Server failed to respond to /health in 90 seconds. Checking logs:")
            try:
                out, _ = proc.communicate(timeout=3)
                print(out[-2000:] if out else "No output")
            except Exception:
                pass
            return False

        print("llama-server is READY!\n")
        time.sleep(2)
        vram_loaded = get_vram_info()
        print(f"VRAM with Model Loaded:   {vram_loaded.get('used_mb')} MB used ({vram_loaded.get('used_mb') - vram_before.get('used_mb', 0)} MB delta)")

        test_prompts = [
            ("English Coherence", "What is the capital of France and what is it famous for? Answer in 2 short sentences."),
            ("Romanian Coherence", "Care este capitala Frantei si pentru ce este faimoasa? Raspunde in 2 propozitii."),
            ("Code Generation", "Write a python function `is_even(n)` that returns True if n is even, False otherwise.")
        ]

        results = []
        for name, p in test_prompts:
            print(f"\n--- Testing: {name} ---")
            print(f"Prompt: {p}")
            res = query_llama_server(p, port=port, max_tokens=100)
            if res["success"]:
                print(f"Speed:  {res['tok_per_sec']} tokens/sec ({res['completion_tokens']} tokens in {res['elapsed_sec']}s)")
                print(f"Output:\n{res['text']}")
            else:
                print(f"ERROR: {res.get('error')}")
            results.append({"test": name, "result": res})

        vram_after = get_vram_info()
        print(f"\nVRAM After Inference:     {vram_after.get('used_mb')} MB used")

        report = {
            "model": model_path,
            "ngl": ngl,
            "ctx": ctx,
            "vram_delta_mb": vram_loaded.get("used_mb", 0) - vram_before.get("used_mb", 0),
            "results": results
        }

        report_path = model_path.replace(".gguf", "_benchmark.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nBenchmark report saved to: {report_path}")
        return True

    finally:
        print("\nStopping llama-server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped cleanly.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_qwen_server_benchmark.py <model.gguf> [ngl] [ctx]")
        sys.exit(1)
    m = sys.argv[1]
    ngl = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    run_benchmark(m, ngl=ngl, ctx=ctx)
