# 唯一实现修正：三臂共同去掉阻塞式group-map提交

上一组A/B/C/C/B/A完整结束：A 8.93894/8.99518s；B 9.53268/9.48301s；C 9.35855/9.33289s。C均值比A慢4.22%，当前实现无净收益，不能称GO。原始输出全部保留在results_opt_r01。

明确的新工程原因：`torch.tensor(mapping, device="cuda")`为当前compute stream的阻塞式host提交；在重叠实现中，host先等当前组ready及map上传完成，随后才组织下一组H2D和当前kernel。原生候选保留了这个旧路径。下一轮只移除此具体同步税，不修改分组宽度、缓存量、输入、输出长度或挑选策略。

复用已有CPU-qualified/CUDA-UNRUN的partial_map_staging.py（源SHA600bc3a703a4c586fbfe232d97c6d861165721f9b29954a9df449e2e9af860ef），三臂全部采用pinned/preallocated map、同consumer stream异步上传；每臂同样额外24KiB host pinned和24KiB GPU，每层6个独立map槽。保留copy/compute依赖与scratch slot保护，host staging复用前只等其上次DMA。

预声明仍为A/B/C/C/B/A，16原序完整提示、各32强制输出token、cap24、KV1GiB、budget512、各臂fresh engine+full warmup。A异步map+原24组串行；B异步map+12组串行；C异步map+12组重叠。主比较为本组C/A和C/B，不把跨组差异直接解释为map因果效应。仍需全部请求/输出一致性、真实wall/flow及加载量，所有尝试保留。

若C仍不胜同组A，结束这个cap24/budget512流水实现，不继续调group size/threshold搜收益。若出现>=3%净正，再扩展新prompt/EOS与更有代表性的HBM压力域。此实验不自动建立论文新颖性。
