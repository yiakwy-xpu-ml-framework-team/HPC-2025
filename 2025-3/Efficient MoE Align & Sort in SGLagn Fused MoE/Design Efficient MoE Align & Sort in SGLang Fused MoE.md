> By fully enabling concurrent multiple blocks execution with arbitrary expert numbers (MAX_EXPERT_NUMBER==256), and with aggresive usage of shared memory (5kB LDS) and registers (52 VGPRs, 48 SGPRs), the MoE Align & Sort logics was crafted to achieve 📈3x in A100🎉, 📈3x in H200🎉, 📈10x in MI100🎉, and 📈7x in MI300X/Mi300A🎉: ...

Author : LEI (yiak.wy@gmail.com)

## Efficient MoE Align & Sort design in SGLang Fused MoE

MoE model mimics low power consumption pattern in human brain : functions are divided into divisions, partial activated via adaptive routing when thinking. 

<br />

<figure>
<p align="center">
<img src="assets/img/brain.jpg" alt="human-brain cortex from Oxford university research paper, archived from internet" style="width:50%">
</p>
<figcaption style="text-align:center">human-brain cortex from Oxford university research paper, archived from internet</figcaption>
</figure>

<br />


The first truely workable version in CUDA is SwitchTransformer[1], then improved by Mistral[2] by upcycling dense models:

<br />

<figure>
<p align="center">
<img src="assets/img/switch-transformer-moe.png" alt="switchTransformer-moe" style="width:50%">
</p>
<figcaption style="text-align:center">switchTransformer-moe</figcaption>
</figure>

<br />

Later DeepSeek V2/V3/R1 [3][4][5] imroved MoE by introducing shared experts [3] and gating bias [4][5], which finally leads to auxiliar loss free MoE models [4][5]. This is essentially attributed to the fact that when shared experts (decided as 1 by deepseek team) are used, imbalance of experts routing problem can be mitigated by forcing a punishment of a bias score over a large pool of experts (256).

<br />

The MoE layer is implemented as multi experts FFN layers, which consists gating functions to route activations according to topk gating scores (with bias in DeepSeek V3/R1), and producing logits by Group GEMM upon selected FFN layers.

<br />

The function relies heavily on radix sorting logics underlying. With MoE Align & Sort, ML researchers and practioners can sort tokens in the order of expert ids. 

<br />

In some application, such as **TransformerEngine** [6][7], the operation was implemented by deprecated **cub::radix_sort**, and **permute** was implemented to record the **src(left)** to **dest(right)** mapping, the gradient of which is **unpermuate**.

<br />

<figure>
<p align="center">
<img src="assets/img/nv_moe_permute_op.png" alt="moe-permute-illustration" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">moe-permute-illustration</figcaption>
</figure>

<br />


Despite the fact that **cub::radix_sort** uses intensively shared memory, which is slighly slower than implementation based on **__shfl_xor_sync** where only thread local memory is used, it does not allow **alignment sorting**.

<br />

The MoE Align & Sort algorithm in SGLang employed **alignment sorting**, yet was not efficient when serving large scale prefill operations for MoE models up to 256 experts. The issue was identified in the [issue#2732](https://github.com/sgl-project/sglang/issues/2732). The current implementation split MoE Align & Sort into two kernel launches : 

<br />

- alignment : to conduct traditional alignment version of offsets computing of radix sorting algorithm in **a single block**;

- placement : place tokens according to the offsets computed in **multiple blocks**;

<br />

With **RocProfiler-Compute** at different workload, we clealy see that the first kernel takes **33W** cycles and second kernel takes **8W** cycles even without counting multiple kernels launch overhead in a trace profile :

<br />

<figure>
<p align="center">
<img src="assets/img/moe_align_k1.png" alt="moe_align_k1" style="width:80%">
</p>
<figcaption style="text-align:center">the moe align kernel 1</figcaption>
<p align="center">
<img src="assets/img/moe_align_k2.png" alt="moe_align_k2" style="width:80%">
</p>
<figcaption style="text-align:center">the moe align kernel 2</figcaption>
</figure>

<br />

In ROCm SDK 6.3.0, omniperf is rebranded as **rocprof-compute**. Dispite the active support of MI300X/MI300A, it is not by default shipped with ROCm SDK **6.3.0**. But setting up the ROCm compute profiler is nothing more than three easier steps demonstrated in [Tools-dockerhub](https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub).

<br />

Now, on chip overhead will be immedately reduced to **20W** from **41W** cycles after the optimization we proposed:

<br />

<figure>
<p align="center">
<img src="assets/img/moe_align_after_opt.png" alt="optimize moe align kernel both in CUDA and ROCm platform" style="background-color:white;width:80%">
</p>
<figcaption style="text-align:center">enable efficient multi-blocks moe-align execution in SGLang</figcaption>
</figure>

<br />

By fully enabling concurrent multiple blocks execution with arbitrary expert numbers (MAX_EXPERT_NUMBER==256), and with aggresive usage of shared memory (5kB LDS) and registers (52 VGPRs, 48 SGPRs), the MoE Align & Sort logics was crafted to achieve [📈3x in A100🎉](#a100_bench), [📈3x in H200🎉](#h200_bench), [📈10x in MI100🎉](#mi100_bench), and [📈7x in MI300X/Mi300A🎉](#mi300_bench):

|    opt bench (all cases)    |  opt bench (snapshot) | GPU
:----------------------------:|:---------------------:|:-----:
![moe-align-block-size-performance](https://github.com/user-attachments/assets/53b177ba-88ef-4d5a-b833-e112160a2b15) | <img width="200" alt="A100-bench" src="https://github.com/user-attachments/assets/19d0daf3-f2b9-4acc-a2d8-c8be2a9c3049" /> | A100
![mi100-moe-align-block-size-performance](https://github.com/user-attachments/assets/addcdfa8-0fba-4fe4-b8ed-68711d3eebe4) | <img width="400" alt="MI00-bench" src="https://github.com/user-attachments/assets/0a474f35-305e-42c4-95a2-bf51f46cdbf9" /> | MI100 (gfx908)

<br />

With **rocprof-compute**, we can easily collect some key indictors for a captured kernel and visualize them in a remote GUI server:

![rocprof-compute](assets/img/rocprof-compute.png)

To summary, in AMD MI300A, the proposed efficient multi-blocks moe align execution algorithm uses aggressively vector regsiters (52) per wave with no registers spills (I adjust initial threads block size to its best), and LDS (5kB) with only 6.8% bank conflicts rates, which can be improved later if necessary.

<br />

We also analyzed the roofline model of MoE Sort & Align. The roofline model shows the kernel performance drops in memory bound region. 

In section [AMD Compute Profile](#amd_compute_profile), we gives details of the profiling data and analysis of our algorithm design and implementation in ROCm platform. 

<br />

Essentially, MI300X/MI300A is a first high performance AI accelerator based on multi-dies architecture, and tuning of operations will be slightly different from those in NVIDIA.

<br />

The basic rule is, synchronization among XCDs (accelerated computing dies) is expensive, making full use of XCDs and L2 cache locality affinity increase the performance. 


<br />

And we should avoid expensive synchronization by either using the lowest speed computing die (XCD7 for MI300X, XCD5 for MI300A) when grid size is smaller than the numbers of XCDs per chip (8 for MI300X, 6 for MI300A), or adapting grid size to multiple of the numbers of XCDs when grid size is bigger than the number.

<br />

Launching cooperative kernels by **hipCooperativeLuanch** may increase L2 cache pressure (relate to texture addresser stall rate and busy rate) when data exchange (espeically Die-Die Exchange) increases among blocks.

<br />

In this example, the main implementation uses **39** active CUs which is **almost good** since essentially two dies were used.

Our implementation uses 66 active CUs in multi-blocks excution that acrossing two dies and Die-Die exchange is inevitable in block-wise reduction. We will submit further V4 optimization to SGLang later in this quarter.

Details will be further discussed in profiling section.

## Review of Fused MoE in SGLang

SGLang team used triton first approach to implement the logics and gained great successes in day 0 support of DeepSeek V3 in Dec 2024.

The SGLang [MoE](https://github.com/sgl-project/sglang/blob/8baf9a0c18c6bc700e89ad6deb200739a8242e09/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L952) launches [fused MoE kernel](https://github.com/sgl-project/sglang/blob/8baf9a0c18c6bc700e89ad6deb200739a8242e09/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L56) implemented in triton.

Before the kernel launch, the MoE Align & Sort algorithm is applied. the MoE Align & Sort triton kernel splitted into 4 phases where direct accesses to DRAM without shared memory are employed [vectorize triton operation](https://github.com/sgl-project/sglang/pull/2913).

Multiple launches and inefficient of use LDS and local caches, registers (VGPR for example) contributed to inefficient single test execution for small workload, compared to single block CUDA counterpart.

Then CUDA implementation is finally splitted into two phaeses and only the second phase execution is accelerated in multiple blocks.

## MoE Align & Sort CUDA Algorithm in other Open Source Platform

#### FasterTransfomer

Before Mistral[2] and DeepSeek V2[3], open dense models are more popular in inference scenarios. This was when **FasterTransfomer**[8] was born.

<br />

In **FasterTransformer**[8] project, initiated by NVIDIA, MoE models are supported essentailly via **cub::DeviceRadixSort** and kernels like **moe_softmax** (which is essentially softmax in **cub::BlockReduce**), **moe_top_k** and its fused version **topk_gating_softmax**, **permute** to order latent vector logits, and finally [group gemm](https://github.com/NVIDIA/FasterTransformer/blob/df4a7534860137e060e18d2ebf019906120ea204/src/fastertransformer/kernels/moe_kernels.cu#L622). 

<br />

Hence fusion is largely (by cost) limited to topk gating softmax, biased topk gating softmax, which are later incoroperated in SGLang.

#### Megatron

Megatron, before the publication of this article, for FP16/BF16, largely uses **FasterTransformer** approach but added gradient operation of **permute** : **unpermute**, to facilitate [training workload](https://github.com/fanshiqing/grouped_gemm).

That means MoE is also not efficiently fused.

#### vLLM

SGLang uses many vLLM kernels, but vLLM 's Fused Moe was initially contributed by SGLang team. Hencey they deploy the same approach.

#### CK

The first version of AMD friendly fused MoE was proposed in [CK#1634](https://github.com/ROCm/composable_kernel/pull/1634) in NOV 26, 2024. Later, MoE Align & Sort was added in [CK#1771](https://github.com/ROCm/composable_kernel/pull/1771) and [CK#1840](https://github.com/ROCm/composable_kernel/pull/1840) in Feb 11, 2025.

<br />

The high level idea is to fuse MoE sorting with Group GEMM. Adn MoE & Sorting largely employes SGLang's team approach.

<br />

<figure>
<p align="center">
<img src="assets/img/ck-fused-moe-v1.png" alt="Fused MoE V1, NOV 26, 2024" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">CK fused MoE High Level Idea[9]</figcaption>
</figure>

<br />

Fusion of **per_group_token_quant** (for online fp8 quantization), **MoE sorting** and **Group GEMM** can be immediately resolved by incorporating Radix Sort computing logics into Group GEMM pipeliner: count occurencies to compute offsets, then do parallel placement.

<br />

One of the most critical problems is that how the two kinds of workloads (Radix Sorting & Group GEMM) is balanced. 

<br />

In AMD data center chips, Group GEMM fragment is more likely to be evenly distributed to all the available blocks in an XCD. While, the data exchange among blocks in different CUs are through low speed of L2 Cache and L2 Cache fabric if multiple XCDs involved. 

<br />

Writing CK kernels requires writing host side CK solution launcher:

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

, [device entry of the kernel](https://github.com/ROCm/composable_kernel/blob/1342ecf7fbf64f43d8621cf6665c583fdc49b2c6/include/ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp#L238), tile partitioner, and stages pipliner.

<br />

The AMD CK partitioner and stages pipliner for fused moe is also very interesting, yet out of scope of this article.

<br />

But just remember its MoE Align & Sort is part of producer codes :

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

        // do parallel placement based on the offsets computed

}
```

<br />

So MoE Align & Sort in the AMD CK solution alomost aligns with our implementation execept for codes scheduler of partitioner and pipliner. 

<br />

Note the implementation does not always promises the best performance in AMD platform.

<br />

Since AMD CDNA3 arch does not support **Graphcore** alike on-chip shuffling (we abstracted and generalized it as **Remapping** in 2023) magics, -- which was now supported in NVIDIA H100/H200/B200 throughout high efficient on chip **SM-SM** communication.

<br />

As a result, adapting the data layout cheaply among blocks to its best will be very an intersting section in AMD's open source solution.

<br />

Hence, in philosophy, tiling based fusion code of these two different workloads may not alwasy exceed the non-fused version. Details of the research will be conducted in our V4 release.

<br />

#### Cutlass v3.8

Fused MoE is not currently supported in NVIDIA Cutlass 3.8.0 at the time I am writing this article. Hence no MoE Align & Sort available this repo.

#### TRT-LLM

Before v0.16.0, the TRT-LLM basic follows **FasterTransformer** approach. After v0.17.0, the MoE part is disclosed.

## Make AMD Friendly CUDA Implementation wtih more than 3x ~ 7x acceleration

The algorithm employes multiple blocks execution schemes and consists of 3 different sections (D-C-P) : 

- Distributed concurrencies counting
- Computing cumsum
  - parallel unaligned local cumsum
  - reduce unaligned cumsum
  - align global cumsum
  - store global cumsum
- Parallel placement

<br />

<figure>
<p align="center">
<img src="assets/img/our_moe_align_sort.drawio.png" alt="Fused MoE V1, NOV 26, 2024" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">Our proposed efficent multi-blocks MoE Align & Sort algorithm</figcaption>
</figure>

<br />

#### Parallel unaligned local cumsum

<br />

<figure>
<p align="center">
<img src="assets/img/parallel_local_unaligned_cumsum.png" alt="Fused MoE V1, NOV 26, 2024" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">Our proposed parallel local unaligned cumsum</figcaption>
</figure>

<br />

The algorithm was first proposed and implemented by us in [PR#2970](https://github.com/sgl-project/sglang/pull/2970).

<br />

We load balanced the cumsum execution in each block to **kElementsPerThr(16)** threads, where each thread , where **kElementsPerThr + kElementsPerThr + threadIdx.x** Add Operations needed to be processed. 

<br />

Hence wavefront is faster to reach compared to the single thread version in curren repo and we hereby observed **30%** improvement in this version of implementation.

#### Reduce unaligned cumsum

Once we get local unligned cumsum in each block, we proceed to block-wise reduction among the cumsum stored in the pre-allocated HBM buffer. 

<br />

We choosed **FRAG_SIZE_M(16) x FRAG_SIZE_N(16) x FRAGS_PER_BLOCK(4)** SRAM fragments for block-wise reduction, and **FRAGS_PER_BLOCK** is tunable :

<br />

<figure>
<p align="center">
<img src="assets/img/block-wise-reduction.drawio.png" alt="Fused MoE V1, NOV 26, 2024" style="background-color:white;width:50%">
</p>
<figcaption style="text-align:center">Our proposed parallel local unaligned cumsum</figcaption>
</figure>

<br />

In AMD platform, calculation is performend on a 1 warp to load / 1 warp to compute basis, while 2 warps to load and 1 warp to compute in NVIDIA platform. 

<br />

The design makes use of full advantages of AMD 64 SIMD lanes in CDNA3 architecture. And the number blocks is always multiple of the number of XCDs in this multi-die arch chip.

<br />

FRAGS_PER_BLOCK was set to 4 to facilitate re-use of SMEM in multiple rounds.

<br />

#### Align global cumsum & store global cumsum

We improved the vectorization codes and take care of loop tails if input data size is not aligned with **kElementsPerAccess** constant.

The benchmarks show coalescing rate is improvmed but still limited to **30%**. We will work on it in V4 release. 

<div id="amd-compute-profile"></div>
## AMD Compute Profile

#### Setup

In ROCm 6.3.3, setup a **rocprof-compute** can be easily as three steps setup, details can be found here : https://github.com/yiakwy-xpu-ml-framework-team/Tools-dockerhub/tree/main


#### Profiling Results of Vector L1 Cache

The workload **16384** tokens x (top **8** out of **256** experts) unless otherwise specified.

| kernel                                              | VGPRs | SGPRs| active CUs | Vector L1 cache hit rate | coalescing rate / utils
:----------------------------------------------------:|:-----:|:----:|:----------:|:------------------------:|-----
[old main](#) moe_align_block_size_kernel (k1)        | 20    | 48   | 3          | 0%                       | 25% / 7%
[old main](#) count_and_sort_expert_tokens_kernel (k2)| 8     | 32   | 39         | 27%                      |
[our](#) moe_align_block_size_kernel                  | 52    | 48   | 66         | 61%                      | 36% / 18%

We maximize the usage of VGPRs but reduce total usage of SGPRs in our algorithm. The data also indicates Zero VGPRs/SGPRs spills usage that healthy usage of registers and no performance panelty for this kernel. 

Vector L1 cache (vL1D) is unit local to each CU, the hit rate records cache line hit rates when data requestd from L2 Cache to CU. **30%** L2 cache requests was coalesced by vL1D's texture addresser and **61%** hit rates achieved, which can also be improved later if necessary.

At the time data requested from CU to vL1D's addressing processing unit (texture addresser), there are for states for unit to decide whether to accept or roll back the data request to CU via data processor unit in vL1D.

- Busy : the texture addresser is processing address

- Address Stall : the texture addresser is stalled from sending address to vL1D

- Data Sending Stall : the texture addresser is stalled from sending data to vL1D

- Data Waiting Stall : the texture addresser is stalled waiting to send data to data processor unit in vL1D

Detials of this micro arch behavior can be found in AMD CDNA3 ISA and [rocProfiler-compute docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/vector-l1-cache.html#desc-td). 

<br />

<figure>
<p align="center">
<img src="assets/img/vL1D-addresser-stall.png" alt="optimize moe align kernel both in CUDA and ROCm platform" style="background-color:white;width:80%">
</p>
<figcaption style="text-align:center">ours vL1D addresser stall</figcaption>
</figure>

<br />

We witnessed 18.61% Data Waiting Stall rate from vector L1 cache in this aglorithm design.

<br />

The load balance of data R/W is greatly reduced from **8 kB** Reading Op, **27 B** Writing Op to combination of **109 B** Reading Op, **468 B** Writing Op and **202 B** Atomic Op.

##### Profiling Results of L2 Cache

In CDNA3, L2 Cache is shared by all CUs and is the main entry to share data among thread blocks distruted to different CUs. 

<br />

With multiple channels and address interleaving design, requests to L2 cache can be largely handled concurrently.

<br />

## Conclusion

The new algorithm accelerate MoE Align & Sort in both CUDA and ROCm platform significantly up to 3x ~ 7x by maximize the usage of LDS and vector registers.

However details of the algorithm can be still polished to improve cache hit rate and main memory coalecsing rate.

## Acknowledgement

Special thanks to Prof Zhang Han, Doctor Wang HanJun from NUS team for collabration in MI100 verification, and Zev Rekhter in MI300X verication. 

## Reference

1. W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models
with simple and efficient sparsity. CoRR, abs/2101.03961, 2021. URL https://arxiv.org/
abs/2101.03961.
2. A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand,
G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
3. DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language
model. CoRR, abs/2405.04434, 2024c. URL https://doi.org/10.48550/arXiv.2405.
04434.
4. DeepSeek V3 : https://arxiv.org/abs/2412.19437; Retrieved on 2025-03-18
5. DeepSeek R1 : https://arxiv.org/pdf/2501.12948; Retrieved on 2025-03-18
6. TransformerEngine : https://github.com/NVIDIA/TransformerEngine; Retrieved on 2025-03-18
7. NV Group GEMM : https://github.com/yiakwy-xpu-ml-framework-team/NV_grouped_gemm; Retrieved on 2025-03-18
8. FasterTransformer : https://github.com/NVIDIA/FasterTransformer; Retrieved on 2025-03-18
9. CK Fused MoE V1 : https://github.com/ROCm/composable_kernel/pull/1634