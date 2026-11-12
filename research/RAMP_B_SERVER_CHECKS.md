# 独立 RAMP B 实现验证

服务器：/home/risegpu/sunjiahao/Alignins；2026-11-12。

- src/ramp_b.py 是完整实现，无 research 模块导入，无旧实验入口转调。
- CPU 对比：CIFAR10 前 512 张训练、前 200 张测试，ResNet9、BadNets、2 客户端/1 恶意、1 轮、local_ep=1。
- 与原 B 实现逐项完全一致：初始模型、数据划分、投毒计划、第一轮攻击更新、第一轮正常更新、最终模型 SHA256。
- 全局 ACC/ASR/触发后原标签 ACC 与原实现一致；报告和图片生成通过。
- 补充校验：与原生三次评估的结果及评估后的 Python/NumPy/CPU RNG 完全一致；额外干净目标类评估不改变训练随机流。
- 稀疏修复只修改选中坐标；静态检查确认方法实现均在 RAMPB 类中。
- 正式 100 轮未由本次工作启动；现有 B/D 实验未打断。临时检查产物已清理。

src/ramp_b.py SHA256: 6be96840c7fef4a5f654a10ef6713185a5b0e8a0a1a22f59d9124787cd7763c7
