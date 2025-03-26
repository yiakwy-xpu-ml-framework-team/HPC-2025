> 通过完全启用并发多块执行，支持任意专家数量（MAX_EXPERT_NUMBER==256），并积极利用共享内存（5kB LDS）和寄存器（52 VGPRs，48 SGPRs），MoE Align & Sort 逻辑被精心设计，实现了显著的性能提升：
📈 A100 提升 3 倍🎉，📈 H200 提升 3 倍🎉，📈 MI100 提升 10 倍🎉，📈 MI300X/MI300A 提升 7 倍🎉 ...

作者：[LEI WANG](https://github.com/yiakwy-xpu-ml-framework-team) (yiak.wy@gmail.com)

## SGLang Fused MoE 中的高效 MoE Align & Sort 设计

MoE（Mixture of Experts）模型 模仿了人脑的低功耗运作模式：功能被划分为多个独立的部分，在思考时通过自适应路由部分激活，从而提高计算效率。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/brain.jpg" alt="human-brain cortex from Oxford university research paper, archived from internet" style="width:50%">
</p>
<figcaption style="text-align:center">牛津大学研究论文中的人脑皮层示意图，来源于互联网</figcaption>
</figure>

<br />

首个可在 CUDA 真正可行的版本是 SwitchTransformer[1]，随后 是 通过循环利用 (Up Cycling) 稠密模型 Mistral[2] 进一步优化了该设计。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/switch-transformer-moe.png" alt="switchTransformer-moe" style="width:50%">
</p>
<figcaption style="text-align:center">SwitchTransformer-MoE</figcaption>
</figure>

<br />

随后，DeepSeek V2/V3/R1 [3][4][5] 通过引入共享专家 [3] 和 门控偏差（gating bias） [4][5] 进一步改进了 MoE，最终实现了无辅助损失（auxiliary loss free）的 MoE 模型 [4][5]。这一优化本质上归因于一个关键事实：当使用共享专家（DeepSeek 团队选择的值为 1）时，可以通过在 **较大的专家池（256 个** 上施加偏差分数的惩罚，从而缓解专家路由的不均衡问题 [11]。

<br />

MoE 层 本质上是由多个专家前馈网络（FFN） 组成的层，其中包含门控函数（gating functions），用于根据 Top-K 门控分数（DeepSeek V3/R1 中引入偏差）进行激活路由，并在所选的 FFN 层 上通过 Group GEMM 计算 logits。

<br />

该功能在很大程度上依赖于基数排序（radix sort）逻辑。借助 MoE Align & Sort，机器学习研究人员和实践者可以按照专家 ID 对 tokens 进行排序。

<br />

在某些应用中，例如 TransformerEngine [6][7]，该操作最初是通过已废弃的 cub::DeviceRadixSort 实现的，而增加的 permute 操作用于记录 源（左） 到 目标（右） 的映射，其梯度操作为 unpermute。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/nv_moe_permute_op.png" alt="moe-permute-illustration" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">MoE Permute 示例</figcaption>
</figure>

<br />

尽管 cub::DeviceRadixSort **大量使用共享内存**，相比于基于 __shfl_xor_sync（仅使用线程本地内存）的实现略慢，但它**不支持对齐排序（alignment sorting**）。

<br />

对齐排序 对于 Group GEMM 的效率至关重要，因为它允许专家以 **块（block** 为单位处理 tokens。

<br />

SGLang 中的 **MoE Align & Sort** 算法采用了 对齐排序，但在 **支持多达 256 个专家的大规模 prefill 操作** 时效率并不理想。该问题已在 [issue#2732](https://github.com/sgl-project/sglang/issues/2732) 中被确认。

<br />

目前的实现将 **MoE Align & Sort** 拆分为两个 kernel 启动（kernel launches）：

<br />

- 对齐（alignment）：在 **单个 block** 内执行 传统 基数排序算法对齐后的偏移计算（alignment-based offsets computation）; 

- 放置（placement）：根据在多个 block 并行计算出的偏移量，并行放置 tokens;

<br />

我们提出并编写了 AMD 友好的 CUDA 设备代码，采用了我们设计的 MoE Align & Sort 算法。因此，在 AMD 平台 上的性能分析和优化将被充分考虑。

<br />

通过在不同的工作负载下使用 **RocProfiler-Compute** 进行分析，我们可以清楚地看到，即使 **不计入多次设备函数启动** 的额外开销，第一个 kernel 仍然消耗了 **33W** 个周期，第二个 kernel 消耗了 **8W** 个周期，总计 **41W** 周期：

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/moe_align_k1.png" alt="moe_align_k1" style="width:80%">
</p>
<figcaption style="text-align:center">the moe align kernel 1</figcaption>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/moe_align_k2.png" alt="moe_align_k2" style="width:80%">
</p>
<figcaption style="text-align:center">the moe align kernel 2</figcaption>
</figure>

<br />

在 ROCm SDK 6.3.0 中，omniperf 已更名为 rocprof-compute。尽管 MI300X/MI300A 已得到积极支持，但该工具默认未随 ROCm SDK 6.3.0 一同发布。不过，在 [Tools-dockerhub](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub) 中的展示一样，ROCm 计算分析工具的设置仅需简单三步。.

<br />

现在，在 [PR#3613](https://github.com/sgl-project/sglang/pull/3613) 中应用我们提出的 [优化方案](https://github.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork/blob/790a832385a02d5f52ad627af333ca1c992e24de/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu#L233) 后，片上计算开销将从之前的 **41W** 个周期立即降低至 **20W** 个周期。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/moe_align_after_opt.png" alt="optimize moe align kernel" style="background-color:white;width:80%">
</p>
<figcaption style="text-align:center">在 SGLang 中实现高效的多块（multi-blocks）MoE-Align</figcaption>
</figure>

<br />

通过 **完全地多块（multiple blocks）并发执行**，并支持 **任意专家数量**（MAX_EXPERT_NUMBER==256），结合 **激进使用共享内存（5kB LDS）和寄存器（52 VGPRs，48 SGPRs）** ，MoE Align & Sort 逻辑被优化，实现了以下性能提升 [📈3x in A100🎉](#a100_bench), [📈3x in H200🎉](#h200_bench), [📈10x in MI100🎉](#mi100_bench), and [📈7x in MI300X/Mi300A🎉](#mi300_bench):

<br />

|    opt bench (all cases)    |  opt bench (snapshot) | GPU
:----------------------------:|:---------------------:|:-----:
![moe-align-block-size-performance](https://github.com/user-attachments/assets/53b177ba-88ef-4d5a-b833-e112160a2b15) | <img width="200" alt="A100-bench" src="https://github.com/user-attachments/assets/19d0daf3-f2b9-4acc-a2d8-c8be2a9c3049" /> | A100
![mi100-moe-align-block-size-performance](https://github.com/user-attachments/assets/addcdfa8-0fba-4fe4-b8ed-68711d3eebe4) | <img width="400" alt="MI00-bench" src="https://github.com/user-attachments/assets/0a474f35-305e-42c4-95a2-bf51f46cdbf9" /> | MI100 (gfx908)

<br />

借助 **Rocprof-Compute**，我们可以轻松收集捕获设备代码的一些关键指标，并在远程 GUI 服务器上进行可视化展示：

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/rocprof-compute.png" alt="start rocprof-compute in server side" style="background-color:white;width:80%">
</p>
<figcaption style="text-align:center">服务端开启 Rocprof-Compute</figcaption>
</figure>

<br />

总而言之，在 AMD MI300X/MI300A 上，所提出的高效 多块（multi-blocks）MoE Align & Sort 算法充分利用了每个 wave 的向量寄存器（52 个），且无寄存器溢出（我已将初始线程块大小调整至最佳值）；同时，每个 CU 使用 5kB LDS，且仅有 6.8% 的存储银行冲突率。

<br />

我们还分析了 MoE Sort & Align 的 Roofline 模型。该模型显示，设备代码的性能在受限于内存带宽的区域有所下降。

<br />

在 [AMD Compute Profile](#amd_compute_profile) 部分，我们详细介绍了在 ROCm 平台上我们算法设计的影响与性能数据。

<br />

本质上，MI300X/MI300A 是全球首款基于多芯片（multi-die）设计的高性能 AI 加速器架构。因此，在该芯片上进行算子优化的方式将与 NVIDIA 平台略有不同。

<br />

基本规则是，XCDs（加速计算芯片）之间的同步代价较高，因此最好充分利用XCDs，并利用L2缓存的局部性亲和性来提高性能。

<br />

此外，我们应避免昂贵的同步开销，具体方法包括：

- 当网格大小小于每颗芯片上的 XCD 数量（MI300X 为 8，MI300A 为 6）时，优先使用最低速计算单元（MI300X 使用 XCD7，MI300A 使用 XCD5）。

- 当网格大小大于每颗芯片上的 XCD 数量时，将其调整为 XCD 数量的整数倍。

<br />

使用 **hipCooperativeLaunch** 启动协作设备代码可能会增加 **L2 缓存压力**（与 **纹理寻址器停滞率** 和 **忙碌率** 相关），特别是在 **数据交换（尤其是 Die-Die 交换** 增多的情况下。

<br />

在此示例中，之前 **main** 分支的实现使用了 **39** 个活跃 CU，这已经 **接近最佳**，因为本质上使用了两个 Die。

<br />

我们的实现在 **多块（multi-blocks）执行** 中使用了 **66** 个活跃 CU，跨越两个 Die，并且 **块级归约（block-wise reduction）** 过程中 **Die-Die 数据交换** 是不可避免的。我们将在本季度晚些时候向 **SGLang** 提交进一步的 **V4 优化**。

<br />

具体细节将在 **性能分析（profiling）** 部分进一步讨论。

## SGLang 中 Fused MoE 的回顾

SGLang 团队采用 **Triton First** 方法实现了相关逻辑，并在 2024 年 12 月成功实现 **DeepSeek V3** 的 **Day-0 支持**。  

<br />

SGLang 的 [MoE](https://github.com/sgl-project/sglang/blob/8baf9a0c18c6bc700e89ad6deb200739a8242e09/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L952) 调用了使用 **Triton** 实现的 [Fused MoE 设备代码](https://github.com/sgl-project/sglang/blob/8baf9a0c18c6bc700e89ad6deb200739a8242e09/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L56)。 

<br />

在设备代码启动之前，会应用 **MoE Align & Sort** 算法。MoE Align & Sort 的 Triton 设备代码被拆分为 **四个阶段**，其中 **直接访问 DRAM**，而不使用共享内存，这与 [向量化 Triton 版本](https://github.com/sgl-project/sglang/pull/2913) 形成对比。  

<br />

与 **单块（single block wise）CUDA 实现** 相比，Triton 版本的 **多次设备代码触发** 以及对 **LDS、本地缓存和寄存器（例如 VGPR）** 的低效利用，导致了在 **小规模工作负载** 上的单次测试执行效率较低。

<br />

随后，CUDA 实现最终被拆分为 **两个阶段**，其中 **仅第二阶段** 的执行在 **多块（multiple blocks）** 上进行了加速。  

## MoE Align & Sort CUDA 算法在其他开源平台的实现

#### FasterTransfomer

在 **Mistral[2]** 和 **DeepSeek V2[3]** 之前，**开放式稠密模型（open dense models）** 在推理场景中更为流行。这也是 **FasterTransformer[8]** 诞生的时期。

<br />

在 **FasterTransformer[8]** 项目中（由 NVIDIA 发起），MoE 模型的支持主要依赖于 **cub::DeviceRadixSort**，以及诸如 **moe_softmax**（本质上是 **cub::BlockReduce** 实现的 softmax）、**moe_top_k** 及其融合版本 **topk_gating_softmax**、用于排列潜在向量 logits 的 **permute**，最终执行 [group gemm](https://github.com/NVIDIA/FasterTransformer/blob/df4a7534860137e060e18d2ebf019906120ea204/src/fastertransformer/kernels/moe_kernels.cu#L622)。

<br />

因此，融合优化主要（按计算开销计算）限制在 **topk gating softmax** 和 **biased topk gating softmax**，后续这些优化被整合进 **SGLang**。

#### Megatron

在本文发表之前，**Megatron** 在 FP16/BF16 计算中主要采用 **FasterTransformer** 方法，但额外添加了 **permute** 的梯度操作 **unpermute**，以支持 [训练任务](https://github.com/fanshiqing/grouped_gemm)。

<br />

这意味着 MoE 仍然没有得到高效融合。

#### vLLM

**SGLang** 使用了许多 **vLLM** 设备代码，但 **vLLM** 的 Fused MoE 最初是由 **SGLang** 团队贡献的。因此，它们采用了相同的方法进行部署。

#### CK

首个 **AMD 友好的 Fused MoE** 版本于 2024 年 11 月 26 日在 [CK#1634](https://github.com/ROCm/composable_kernel/pull/1634) 中提出。随后，**MoE Align & Sort** 被添加到 [CK#1771](https://github.com/ROCm/composable_kernel/pull/1771) 和 [CK#1840](https://github.com/ROCm/composable_kernel/pull/1840) 中。

<br />

核心思路是将 **MoE 排序** 与 **Group GEMM** 进行融合。此外，CK 中的 **MoE & Sorting** 在很大程度上采用了 **SGLang 团队** 的方法，但在 **CK pipeline 及 partitioner** 方面有所不同。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/ck-fused-moe-v1.png" alt="ck fused moe" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">CK 融合 MoE 思路[9]</figcaption>
</figure>

<br />

融合 **per_group_token_quant**（用于在线 FP8 量化）、**MoE 排序** 和 **Group GEMM** 可以通过将 **Radix Sort 计算逻辑** 纳入 **Group GEMM pipeline** 轻松解决：即 **统计出现次数以计算偏移量**，随后进行 **并行放置**。

<br />

其中最关键的问题之一是 **如何平衡 Radix Sorting 和 Group GEMM 这两种计算负载**。

<br />

在 AMD 数据中心芯片中，**Group GEMM 片段更可能均匀分布在 XCD 内的所有可用计算单元**。然而，当涉及多个 XCD 时，不同 CU 之间的 **数据交换** 主要通过 **低速 L2 Cache 及其互联结构（L2 Cache fabric）** 进行。

<br />

编写 **CK 设备代码** 需要先编写 **主机端 CK 解决方案启动器**：

```
    // Here is the entry of fused MoE : 
    //   https://github.com/ROCm/composable_kernel/blob/1342ecf7fbf64f43d8621cf6665c583fdc49b2c6/example/ck_tile/15_fused_moe/instances/fused_moegemm_api_internal.hpp
    using f_pipeline    = ck_tile::FusedMoeGemmPipeline_FlatmmUk<f_problem>;
    using f_partitioner = ck_tile::FusedMoeGemmTilePartitioner_Linear<f_shape>;
    using f_kernel      = ck_tile::FusedMoeGemmKernel<f_partitioner, f_pipeline, void>;

    const dim3 grids                       = f_kernel::GridSize(a);
    constexpr dim3 blocks                  = f_kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    static int printed = 0;

    auto kargs = f_kernel::MakeKargs(a);
    if(s.log_level_ > 0 && printed == 0)
    {
        std::cout << ", " << f_kernel::GetName() << std::flush;
        printed = 1;
    }

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(f_kernel{}, grids, blocks, 0, kargs));
```

, [设备代码入口](https://github.com/ROCm/composable_kernel/blob/1342ecf7fbf64f43d8621cf6665c583fdc49b2c6/include/ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp#L238), 分块器, 和 多阶段流水线.

<br />

AMD CK **分区器** 和 **阶段流水线（stages pipeliner）** 在 **Fused MoE** 的最终汇编过程中扮演了重要角色，确实值得深入研究，但已超出本文讨论范围。

<br />

但需要记住，**MoE Align & Sort 是生产者代码的一部分**：

```
// https://github.com/ROCm/composable_kernel/blame/fdaff5603ebae7f8eddd070fcc02941d84f20538/include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp#L438
CK_TILE_DEVICE void moe_align_block_size_kernel(...) 
{
        const index_t tid       = static_cast<index_t>(threadIdx.x);
        const index_t start_idx = tid * tokens_per_thread;
...
#if 1
        if(tid < num_experts){ // each thread reduce a column segment of tokens_cnts with # blockDim.x elements
          ...
        }
#else
...
#endif
        __syncthreads();

        // do cumsum to compute offsets based on condition
        ...
        // do parallel placement based on the offsets computed
        ...
}
```

<br />

因此，在 **AMD CK 方案** 中，**MoE Align & Sort** 的实现几乎与 **SGLang** 主实现保持一致，仅在 **分区器（partitioner）** 和 **流水线（pipeliner）** 方面有所不同。

<br />

需要注意的是，该实现并不总是能在 **AMD** 平台上提供最佳性能（请参考 **AITER** 中的 **asm MoE**）。

<br />

由于 **AMD CDNA3 架构** 并不支持类似 **Graphcore** 的 **片上（on-chip）** 洗牌操作（我们在 2023 年已经将 **PopART[12] & PopRT** 的 **Remapping** 操作进行抽象与泛化），而这一特性已在 **NVIDIA H100/H200/B200** 中得到了支持，并通过高效的 **SM<->SM** 片上通信实现。

<br />

因此，在 **AMD 开源解决方案** 中，如何以低开销方式在 **块（block）** 之间优化数据布局将是一个非常有趣的研究方向。

<br />

从哲学上讲，这两类不同工作负载的 **基于 Tiling 的融合代码** 可能并不总是比 **非融合版本** 更优。相关研究的详细内容将在 **V4 版本** 发布时进一步探讨。

<br />

#### AITER

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/aiter.png" alt="Fused MoE in AI Tensor Engine for ROCm" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">AI Tensor Engine[10]</figcaption>
</figure>

<br />

**AITER** 在今年早些时候被引入，以便整合在不同项目中使用的 **LLM 设备代码**。它通过 [ck moe](https://github.com/ROCm/aiter/pull/95)、[asm 版本的 MoE 通过 hipModule](https://github.com/ROCm/aiter/blob/52085276ad4710e1a0c9ce2f62ca177a2af35ffa/csrc/py_itfs_cu/asm_fmoe.cpp#L69) 和 triton **fused moe** 支持 MoE 融合。

<br />

因此，AITER 是部分开源的，因为不透明的汇编代码和开发计划是针对 MI300X 开发者的。

<br />

AITER 中 **fused MoE 的三倍加速** [10] 已由 **Bruce Xu** [13] 验证，并且这一加速主要来自于在不同形状的 **Group GEMM** 中观察到的加速：一个 GEMM 操作，其中每个专家的 FFN 权重与一块隐藏状态的 token 进行相乘。

<br />

这一证明可以在 [PR#199](https://github.com/ROCm/aiter/pull/199) 中找到，asm gemm 几乎带来了 **三倍的性能提升**。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/asm_flatmm_kernel.png" alt="asm flat matrix multiply" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">ASM 版本 扁平矩阵乘</figcaption>
</figure>

<br />

值得注意的是，仍然有一些情况下，选择了来自 **SGLang 社区** 的 **triton 设备代码**。为了在 **MI300X/MI300A** 上高效运行 **triton 设备代码**，它们采用了基于 **多芯片架构** 的特定逻辑，将线程块映射到不同的 **计算单元（dies）** 上：

```
    # https://github.com/ROCm/triton/blob/f669d3038f4c03ee7a60835e875937c65b5cec35/python/perf-kernels/gemm.py#L115
    ...
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    ...
```

此外，在 **CK fused MoE** 中使用了多种 **AMD 芯片内建函数（intrinsics）**，例如：

- **__builtin_nontemporal_load**, 

- **__builtin_amdgcn_ds_swizzle**, 

- **__builtin_amdgcn_ds_permute**/**__builtin_amdgcn_ds_bpermute**, 

- **_builtin_amdgcn_mov_dpp** 

等等。这些内建函数可能最终影响 **fused MoE** 的汇编实现和性能。

<br />

例如，使用 **__builtin_nontemporal_load** 可以跳过 L2 缓存，从而为预测将被重复使用的数据留出更多 L2 缓存行空间。

#### Cutlass v3.8

截至本文撰写时，**Fused MoE** 尚未在 **NVIDIA Cutlass 3.8.0** 中公开支持。因此，当前该仓库中没有提供 **MoE Align & Sort** 功能。

#### TRT-LLM

在 **v0.16.0** 之前，**TRT-LLM** 基本上遵循了 **FasterTransformer** 的方法。自 **v0.17.0** 版本起，**MoE** 部分开始公开。

## 编写 对 AMD 设备 友好的 CUDA 实现，并带来 超过 3x ~ 7x 加速

该算法采用了多块执行方案，并由三个不同的部分（D-C-P）组成：

- 分布式并发计数
- 计算累积和（cumsum）
  - 并行非对齐本地累积和
  - 减少非对齐累积和
  - 对齐全局累积和
  - 存储全局累积和
- 并行放置

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/our_moe_align_sort.drawio.png" alt="our moe align sort overview" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">我们提出的高效 MoE Align & Sort 算法</figcaption>
</figure>

<br />

#### 并行非对齐本地累积和

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/parallel_local_unaligned_cumsum.png" alt="parallel local unaligned cumsum" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">我们提出的并行非对齐本地累积和
</figcaption>
</figure>

<br />

该算法首次由我们在 [PR#2970](https://github.com/sgl-project/sglang/pull/2970) 中提出并实现。

<br />

我们将每个块中的累积和执行进行了负载均衡，分配给 **kElementsPerThr(16)** 个线程，每个线程需要处理 **kElementsPerThr + kElementsPerThr + threadIdx.x** 次加法操作。

<br />

因此，与当前仓库中的单线程版本相比，波前（wavefront）更快地到达，我们观察到此版本实现的性能提升了 **30%**。

#### 减少非对齐累积和（Reduce Unaligned Cumsum）

一旦我们获得了每个块中的本地非对齐累积和，就可以在预分配的 HBM 缓冲区中进行块级别的累积和归约。

<br />

我们选择了 **FRAG_SIZE_M(16) x FRAG_SIZE_N(16) x FRAGS_PER_BLOCK(4)** 的 SRAM 块进行块级归约，其中 **FRAGS_PER_BLOCK** 是可调的：

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/block-wise-reduction.drawio.png" alt="block-wise reduction" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">块级规约
</figcaption>
</figure>

<br />

在AMD平台上，计算是基于“1 warp 加载 / 1 warp 计算”的方式进行的，而在NVIDIA平台上则是“2 warps 加载和 1 warp 计算”。

<br />

该设计充分利用了AMD CDNA3架构中64个SIMD通道的优势。并且，在这种多芯片架构中，块的数量始终是XCD数量的倍数。

<br />

**FRAGS_PER_BLOCK** 被设置为4，以便在多轮中复用SMEM。

<br />

#### 对齐全局累积和和存储全局累积和

我们改进了向量化代码，并处理了如果输入数据大小与 **kElementsPerAccess** 常量不对齐时的循环尾部情况。

基准测试显示，合并率有所提高，但仍然限制在 **30%** 左右。我们将在V4版本中继续优化此问题。

#### 编写AMD友好的CUDA代码

编写PyTorch扩展可以自动将CUDA设备代码转换为HIP设备代码，配合ROCm SDK进行使用。

但是，有些情况下HIP设备代码与CUDA设备代码表现不同：

- Warp大小是一个与架构相关的全局变量，并在ROCm SDK中定义为 **warpSize**；在CDNA3架构中，**warpSize** 定义为 **64**。

- 设备函数签名可能与CUDA不完全对齐，因此需要条件编译来支持这些符号。

- 需要特别关注多芯片架构中的L2缓存优化。

## 基准测试

我们在没有CUDA图捕获的情况下，针对DeepSeek V3模型的大规模工作负载进行了广泛测试。因此，专家数量设置为256。当前的算法不支持在CUDA图捕获下运行，我们将在V4版本中解决此问题。

<br />

由于GPU虚拟化和测试节点上分配的CPU数量，性能可能会与裸机测试时有所不同。

<br />

因此，我们使用Triton实现作为基准，展示我们提出的MoE Align & Sort算法在加速倍数和效率上的表现。

<br />

每个测试首先进行了验证，之后才开始基准测试。在基准测试中，我们观察到，在AMD平台上，Triton的运行时间显著长于在NVIDIA平台上的运行时间。我们因此建议进一步优化Triton的MLIR，以获得比NVIDIA Triton更高效的降级过程。

<br />

对于AMD Triton，我们观察到MI300X的速度比MI100快1.5倍，因此MI300X的性能提升幅度不像MI100那么显著。此外，尽管普遍认为MI300X比MI100更快，但在我们的测试中，MI100上的算法性能要优于MI300X。

这部分归因于内存瓶颈操作，在多芯片之间的通信降低了执行速度。

<br />

在两个平台上，我们都观察到了应用我们提出的算法后显著的性能改进，其中现有的CUDA实现几乎与Triton消耗相同的时间。

#### AMD系统准备

为了最大化使用AMD异构系统，建议进行以下检查。

- NVIDIA Grace CPU和AMD EPYC 9004系统通常建议禁用NUMA自动平衡，以便与GPU协同工作；然而，在某些情况下，可能[不建议禁用](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#) NUMA自动平衡。

- 启用虚拟化时，建议启用IOMMU直通模式，以消除DMA翻译，从而带来性能提升。

<div id="mi100_bench"></div>

#### MI100基准测试

> git clone https://github.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork.git -b optimize_moe_align_v3 && cd sgl-kernel && python setup_rocm.py install

可以验证不同输入令牌和专家数量组合的可行性 :

> cd ../benchmark/kernels/fused_moe_trition && python benchmark_deepseekv3_moe_align_blocks.py --verify


| num_tokens  | experts | SGLang    | Triton (AMD) | GPU  
:------------:|:-------:|:---------:|:------------:|------
8192          | 256     |   79.36   | 426.71       | MI100
16384         | 256     |   86.4    | 681.12       | MI100
16384 x 128   | 256     |   3047.68 | 62442.85     | MI100
32768 x 128   | 256     |   7211.37 | 129388.43    | MI100


<div id="a100_bench"></div>

#### A100 性能测试


| num_tokens  | experts | SGLang     | Triton (NV) | GPU  
:------------:|:-------:|:---------:|:------------:|------
8192          | 256     |   77.44    | 124.92      | A100
16384         | 256     |   \        | \           | A100
16384 x 128   | 256     |   5966.81  | 17396.51    | A100
32768 x 128   | 256     |   12450.05 | 34711.14    | A100


<div id="h200_bench"></div>

#### H200 性能测试

| num_tokens  | experts | SGLang     | Triton (NV) | GPU  
:------------:|:-------:|:---------:|:------------:|------
8192          | 256     |   \        | \           | H200
16384         | 256     |   \        | \           | H200
16384 x 128   | 256     |   4508.42  | 12361.15    | H200
32768 x 128   | 256     |   9023.48  | 24683.70    | H200


<div id="mi300_bench"></div>

#### MI300X 性能测试

| num_tokens  | experts | SGLang     | Triton (AMD) | GPU  
:------------:|:-------:|:----------:|:-----------:|------
8192          | 256     |   88.16    | 281.64      | MI300X
16384         | 256     |   134.02   | 448.88      | MI300X
16384 x 128   | 256     |   6865.64  | 43266.09    | MI300X
32768 x 128   | 256     |   13431.80 | 89788.58    | MI300X

<div id="amd-compute-profile"></div>

## AMD Compute Profile

#### 设置

在ROCm 6.3.3版本中，设置**rocprof-compute**只需三步即可完成，详细的设置步骤可以在这里找到：[Tools-dockerhub中的rocprof-compute设置](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub/tree/main)。

#### 向量L1缓存的分析结果

在分析中，工作负载为**16384**个tokens x（从**256**个专家中选择**8**个），除非另有说明。

| kernel                                              | VGPRs | SGPRs| active CUs | Vector L1 cache hit rate | coalescing rate / utils
:----------------------------------------------------:|:-----:|:----:|:----------:|:------------------------:|-----
[old main](https://github.com/sgl-project/sglang/blob/fb8886037c32138e418cfc333baaef43b1e1f68b/sgl-kernel/csrc/moe/moe_align_kernel.cu#L44) moe_align_block_size_kernel (k1)        | 20    | 48   | 3          | 0%                       | 25% / 7%
[old main](https://github.com/sgl-project/sglang/blob/fb8886037c32138e418cfc333baaef43b1e1f68b/sgl-kernel/csrc/moe/moe_align_kernel.cu#L28) count_and_sort_expert_tokens_kernel (k2)| 8     | 32   | 39         | 27%                      | NaN
[our](https://github.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork/blob/790a832385a02d5f52ad627af333ca1c992e24de/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu#L233) moe_align_block_size_kernel                  | 52    | 48   | 66         | 61%                      | 36% / 18%

我们在算法中最大化了VGPRs的使用，但减少了SGPRs的总使用量。数据也表明，VGPRs/SGPRs的溢出为零，这表明寄存器的使用是健康的，并且此设备代码没有性能损失。

<br />

向量L1缓存（vL1D）是每个CU的本地单元，命中率记录了从L2缓存请求到CU时的缓存行命中率。**30%**的L2缓存请求通过vL1D的纹理寻址器合并，达到了**61%**的命中率，如果需要，稍后可以进一步提升。

<br />

当数据从CU请求到vL1D的寻址处理单元（纹理寻址器）时，复杂的决策逻辑决定是否接受数据请求或回滚数据请求。以下是四种状态：

- **Busy**（忙碌）：纹理寻址器正在处理地址。
- **Address Stall**（地址停顿）：纹理寻址器无法发送地址到vL1D。
- **Data Sending Stall**（数据发送停顿）：纹理寻址器无法发送数据到vL1D。
- **Data Waiting Stall**（数据等待停顿）：纹理寻址器等待发送数据到vL1D的数据处理单元。

<br />

有关这种微架构行为的详细信息，可以在AMD CDNA3的ISA文档以及[rocProfiler-compute文档](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/vector-l1-cache.html#desc-td)中找到。

<br />

<figure>
<p align="center">
<img src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/HPC-2025/main/2025-3/Efficient%20MoE%20Align%20%26%20Sort%20in%20SGLagn%20Fused%20MoE/assets/img/vL1D-addresser-stall.png" alt="vL1D addresser stall" style="background-color:white;width:80%">
</p>
<figcaption style="text-align:center">vL1D 寻址器停顿</figcaption>
</figure>

<br />

我们在该算法设计中观察到了 **18.61%** 的数据等待停顿率来自于向量 L1 缓存。

<br />

数据的读写负载平衡大大减少，从 **8 kB** 的读取操作和 **27 B** 的写入操作，转变为 **109 B** 的读取操作，**468 B** 的写入操作和 **202 B** 的原子操作的组合。

##### L2 缓存的分析结果

在 CDNA3 架构中，L2 缓存是所有计算单元（CU）共享的，且是线程块之间共享数据的主要通道，这些线程块分布在不同的 CUs 上。

<br />

通过多通道和地址交错设计，向 L2 缓存的请求可以大大并行处理。

<br />

此外，使用 AMD 特有的内置函数如 **__builtin_nontemporal_load**，我们可以绕过 L2 缓存来处理那些不需要再次访问的数据。

<br />

更多 L2 缓存研究细节将在 V4 版本中揭示。

## 结论

新的算法通过最大化使用 LDS 和向量寄存器，显著加速了 CUDA 和 ROCm 平台上的 MoE Align & Sort，提升幅度高达 **3x ~ 7x**。

<br />

我们还观察到，相较于单个芯片，内存密集型操作在多芯片架构下可能表现更差，这表明在多芯片如 MI300X/MI300A 和 B200/B300 设备上编程时，可能需要新的微调方向。

<br />

然而，该算法的细节仍有进一步优化空间，以提高缓存命中率和主内存合并率。

## 致谢

特别感谢来自 NUS 团队的秦章含教授 (hanzhangqin8@gmail.com)，王昀鸿博士 (yunhongwang2000@gmail.com) 在 MI100/MI250 性能验证中的合作，Zev Rekhter (Connect@reishi.ai) 在 MI300X 性能验证中的合作，范舒宜 (fsygd1996@163.com) 在 H200 验证中的合作，以及 [BBuf](https://github.com/BBuf)(1182563586@qq.com) 在 SGLang 解决方案的讨论和审阅。

<br />

请注意，这是 SGLang 社区的独立工作。

<br />

我还要深深感谢 Bingqing、Peng Sun 和 ShawHai，他们抽空审阅文章并提供修改建议时给予的帮助。

## 参考文献

1. W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. CoRR, abs/2101.03961, 2021. URL [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961).
2. A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
3. DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR, abs/2405.04434, 2024c. URL [https://doi.org/10.48550/arXiv.2405.04434](https://doi.org/10.48550/arXiv.2405.04434).
4. DeepSeek V3 : [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437); Retrieved on 2025-03-18
5. DeepSeek R1 : [https://arxiv.org/pdf/2501.12948](https://arxiv.org/pdf/2501.12948); Retrieved on 2025-03-18
6. TransformerEngine : [https://github.com/NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine); Retrieved on 2025-03-18
7. NV Group GEMM : [https://github.com/yiakwy-xpu-ml-framework-team/NV_grouped_gemm](https://github.com/yiakwy-xpu-ml-framework-team/NV_grouped_gemm); Retrieved on 2025-03-18
8. FasterTransformer : [https://github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer); Retrieved on 2025-03-18
9. CK Fused MoE V1 : [https://github.com/ROCm/composable_kernel/pull/1634](https://github.com/ROCm/composable_kernel/pull/1634)
10. AMD 3X MOE : [https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)
11. Lean Wang and Huazuo Gao and Chenggang Zhao and Xu Sun and Damai Dai Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts, 2024. URL [https://arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664).
12. PopART on chip TensorRemap : [https://github.com/graphcore/popart/tree/sdk-release-3.4](https://github.com/graphcore/popart/tree/sdk-release-3.4)
13. DeepSeek V3 Optimization based on AITER backend : [https://github.com/sgl-project/sglang/pull/4344](https://github.com/sgl-project/sglang/pull/4344)

## 赞助者渠道

请前往 [reishi.ai](https://reishi.ai/blog/Moe-align-and-sort) 和 [huggingface](https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang)
