# Oracle 良性参考验证

2026-11-12，服务器 CPU 验证。

[
  {
    "file": "ramp_b.py",
    "checks": [
      "benign-only pool",
      "quantiles computed inside benign pool",
      "median",
      "malicious update invariance",
      "benign-only fallback",
      "partial participation",
      "no benign explicit failure",
      "mean option"
    ]
  },
  {
    "file": "ramp_zero.py",
    "checks": [
      "benign-only pool",
      "quantiles computed inside benign pool",
      "median",
      "malicious update invariance",
      "benign-only fallback",
      "partial participation",
      "no benign explicit failure",
      "mean option"
    ]
  }
]

运行中旧进程不受磁盘源文件修改影响，重新运行需新实验名称。
