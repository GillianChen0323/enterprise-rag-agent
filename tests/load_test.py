"""Locust load test. Targets: P50 <= 2000 ms, P95 <= 5000 ms, error rate < 1 percent, and
1000 users for 10 minutes. These are targets, not verified results. Headless command:
locust -f tests/load_test.py --headless --users 1000 --spawn-rate 50 --run-time 10m
--host http://localhost:8000 --html load_test_report.html. For the web UI, omit headless
options and open http://localhost:8089."""

import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# ─────────────────────────────────────────────────────
# Synthetic queries for three use cases.
# ─────────────────────────────────────────────────────

KB_QA_QUERIES = [
    "员工年假政策是什么？",
    "如何申请报销？",
    "公司的远程工作政策",
    "试用期转正标准",
    "绩效考核流程",
    "医疗保险覆盖范围",
    "如何申请内部转岗",
    "出差补贴标准是多少",
    "产假和陪产假政策",
    "IT设备申请流程",
]

HELPDESK_QUERIES = [
    "用户无法登录系统怎么处理",
    "密码重置流程",
    "VPN连接失败如何排查",
    "打印机显示离线",
    "Excel文件打开报错",
    "邮件无法发送",
    "会议室预订系统使用说明",
    "新员工账号开通",
    "软件安装权限申请",
    "数据备份在哪里",
]

COMPLIANCE_QUERIES = [
    "这份合同的保密条款是否合规",
    "数据处理协议需要哪些必要条款",
    "GDPR合规检查清单",
    "合同违约金条款审查",
    "供应商协议风险评估",
]

TENANT_IDS = [f"tenant_{i:03d}" for i in range(10)]  # Ten simulated tenants.


# ─────────────────────────────────────────────────────
# User behavior.
# ─────────────────────────────────────────────────────

class RAGAgentUser(HttpUser):
    """Simulate enterprise queries with a 1 to 3 second wait between requests."""
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize a session with a randomly selected tenant."""
        self.tenant_id = random.choice(TENANT_IDS)

    @task(5)  # Knowledge-base weight: 5.
    def query_kb_qa(self):
        """Send a knowledge-base query."""
        query = random.choice(KB_QA_QUERIES)
        self._send_chat(query, "kb_qa")

    @task(3)  # Support weight: 3.
    def query_helpdesk(self):
        """Send a support query."""
        query = random.choice(HELPDESK_QUERIES)
        self._send_chat(query, "helpdesk")

    @task(2)  # Compliance weight: 2.
    def query_compliance(self):
        """Send a compliance query."""
        query = random.choice(COMPLIANCE_QUERIES)
        self._send_chat(query, "compliance")

    def _send_chat(self, query: str, use_case: str):
        """Send a chat request and validate latency and response format. Locust records the
        response-time distribution."""
        payload = {
            "query": query,
            "use_case": use_case,
            "tenant_id": self.tenant_id,
            "stream": False,
        }
        with self.client.post(
            "/v1/chat",
            json=payload,
            name=f"/v1/chat [{use_case}]",  # Group by use case.
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                # Validate response format.
                if not data.get("answer"):
                    resp.failure("Response missing 'answer' field")
                    return
                # Record latency outliers; Locust calculates percentiles.
                if data.get("latency_ms", 0) > 5000:
                    resp.failure(f"Latency too high: {data['latency_ms']}ms")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")


class StreamUser(HttpUser):
    """Simulate streaming clients and exercise the streaming response path."""
    wait_time = between(2, 5)
    weight = 2  # Streaming clients.

    @task
    def stream_query(self):
        use_case = random.choice(["kb_qa", "helpdesk"])
        query = random.choice(KB_QA_QUERIES + HELPDESK_QUERIES)
        with self.client.post(
            "/v1/chat/stream",
            json={"query": query, "use_case": use_case,
                  "tenant_id": "tenant_001", "stream": True},
            name="/v1/chat/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                # Consume streaming response.
                content = b""
                for chunk in resp.iter_content(chunk_size=1024):
                    content += chunk
                resp.success()
            else:
                resp.failure(f"Stream failed: {resp.status_code}")


# ─────────────────────────────────────────────────────
# Reporting hooks.
# ─────────────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Print SLA checks after the load test."""
    stats = environment.runner.stats
    total = stats.total

    print("\n" + "="*55)
    print("  Load Test Results")
    print("="*55)
    print(f"  Total requests:    {total.num_requests}")
    print(f"  Failures:          {total.num_failures} "
          f"({total.fail_ratio:.1%})")
    print(f"  P50 latency:       {total.get_response_time_percentile(0.5):.0f}ms")
    print(f"  P95 latency:       {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"  P99 latency:       {total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  RPS:               {total.total_rps:.1f}")
    print("="*55)

    # SLA assertions.
    p50 = total.get_response_time_percentile(0.5)
    p95 = total.get_response_time_percentile(0.95)
    fail_ratio = total.fail_ratio

    sla_pass = True
    if p50 > 2000:
        print(f"  ❌ SLA FAIL: P50 {p50:.0f}ms > 2000ms")
        sla_pass = False
    if p95 > 5000:
        print(f"  ❌ SLA FAIL: P95 {p95:.0f}ms > 5000ms")
        sla_pass = False
    if fail_ratio > 0.01:
        print(f"  ❌ SLA FAIL: Error rate {fail_ratio:.1%} > 1%")
        sla_pass = False

    if sla_pass:
        print("  ✅ All SLA targets met!")
    print("="*55 + "\n")
