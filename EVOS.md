https://gemini.google.com/share/951349d736e8

[EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](https://weixiang-zhang.github.io/proj-evos/)



这篇文章主要提出了EVOS（进化选择器）来加速隐式神经表示（INR）的训练，其核心思路包含以下三个关键方法：

1. **稀疏适应度评估 (Sparse Fitness Evaluation)** 

   - **思路总结**: 为了避免每次迭代都计算所有坐标点的适应度（即重建误差），该方法只在特定的“关键迭代”时才计算完整的适应度，并将结果缓存。在两次关键迭代之间的“中间步骤”中，直接使用缓存的适应度来进行坐标选择 。关键迭代的频率会随着训练的进行而**线性增加**（即间隔线性减小），以平衡计算效率和选择的准确性。

   - **代码/伪代码实现** (源自 `img_sampling_trainer.py`):

     

     ```python
     # 判断是否是关键迭代 (对应论文 Eq. 2)
     def _evos_is_fitness_eval_iter(self, epoch):
         _cur_interval = self._evos_get_cur_interval(epoch) # 计算当前间隔 τ - θt/T
         return epoch % _cur_interval == 1 # 使用取模判断
     
     # 计算当前关键迭代间隔 (对应论文 Eq. 2 中的 τ - θt/T)
     def _evos_get_cur_interval(self, epoch):
         if self.args.profile_interval_method == "fixed": # 固定间隔 τ
             return self.args.init_interval
         elif self.args.profile_interval_method == "lin_dec": # 线性递减间隔
             _start = self.args.init_interval # 初始间隔 τ
             _end = self.args.end_interval # 最终间隔 τ - θ
             # 线性插值计算当前间隔
             _cur_interval = _start + ((_end - _start) / self.args.num_epochs) * epoch
             return int(_cur_interval)
     
     # 在关键迭代中计算并缓存适应度信息 (在交叉函数内实现)
     def _evos_frequency_aware_crossover(self, pred, gt, epoch):
         # ... (计算 error_map, 即适应度 D(Fθ(x), y)) ...
         # 缓存适应度图和排序后的索引 (对应论文中的 fˆ(x))
         self.book["freeze_profile_pred"] = pred.detach() # 缓存预测结果 (用于跨频监管)
         self.book["error_map"] = error_map.detach()       # 缓存适应度图
         self.book["sorted_map_index"] = sorted_map_index # 缓存排序索引
     
     # 在训练循环中根据是否是关键迭代决定行为
     # (体现在 _sampler_get_coords_gt 和 _sampler_compute_loss 方法中)
     def _sampler_get_coords_gt(self, epoch):
         coords, gt = self.full_coords, self.full_gt
         # ...
         elif _st == "evos":
             if self._evos_is_fitness_eval_iter(epoch):
                 # 关键迭代：返回全部坐标用于计算适应度
                 return coords, gt
             else:
                 # 中间步骤：使用缓存的适应度信息 (self.book["sorted_map_index"]) 进行选择
                 selection_mask = self._evos_get_selection_mask(epoch)
                 _coords = self.full_coords[selection_mask]
                 _gt = self.full_gt[selection_mask]
                 return _coords, _gt
         # ...
     ```

2. 

   **频率引导交叉 (Frequency-Guided Crossover)**

   - **思路总结**: 为了克服INR偏爱低频信息的缺陷，该方法同时从低频和高频两个角度评估适应度 6。低频适应度使用平方误差 ($||F_{\theta}(x) - y||^2$) 7777，高频适应度使用拉普拉斯算子计算的误差 ($||L_{lap}(F_{\theta}(x)) - L_{lap}(y)||^2$) 8888。分别选出两组“父代”坐标 $x_t^{\prime}$ 和 $x_t^{\prime\prime}$ 9。最终的“后代”坐标 $w_t$ 由两部分组成：一是两个父代集合的交集（即低频和高频误差都大的点）；二是从两个父代集合的差集中按比例 $\Psi$ 选取，比例 $p$ 根据当前低频和高频的整体重建质量动态调整，以平衡两者 。

   - **代码/伪代码实现** (源自 `img_sampling_trainer.py` 的 `_evos_frequency_aware_crossover` 方法，对应 `crossover_method == "select"`):

     Python

     ```Python
     def _evos_frequency_aware_crossover(self, pred, gt, epoch):
         # 计算低频适应度 error_map (L2误差)
         error_map = F.mse_loss(pred, gt, reduction="none").mean(1)
         # 根据低频适应度排序索引
         sorted_map_index = torch.argsort(error_map.flatten())
         # 缓存低频适应度和索引
         self.book["error_map"] = error_map.detach()
         self.book["sorted_map_index"] = sorted_map_index
     
         # 如果交叉方法是 "select"
         if self.args.crossover_method == "select":
             # 计算高频适应度 laplace_error_map (拉普拉斯误差)
             r_img = self.reconstruct_img(pred)
             laplace_map = F.mse_loss(
                 compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
             )
             cross_lap_coff = self.args.lap_coff if self.args.lap_coff > 0 else 1e-5
             laplace_error_map = cross_lap_coff * laplace_map.flatten()
             # 根据高频适应度排序索引
             sorted_lap_map_index = torch.argsort(laplace_error_map.flatten())
             # 缓存高频适应度和索引
             self.book["sorted_lap_map_index"] = sorted_lap_map_index
     
             # --- 确定选择数量 ---
             # ... (计算 selected_num) ...
     
             # --- 选择父代候选 ---
             l2_error_selected_index = sorted_map_index[-selected_num:] # 低频父代 x'_t
             lap_error_selected_index = sorted_lap_map_index[-selected_num:] # 高频父代 x''_t
     
             # --- 生成后代 ---
             # 1. 计算交集 (x'_t ∩ x''_t)
             isin = torch.isin(l2_error_selected_index, lap_error_selected_index)
             selected_index = l2_error_selected_index[isin] # 交集直接加入后代
     
             # 2. 计算差集和需要补充的数量
             remain_num = selected_num - selected_index.shape[0]
             l2_remain_index = l2_error_selected_index[~isin] # x'_t \ x''_t
             isin2 = torch.isin(lap_error_selected_index, l2_error_selected_index)
             lap_remain_index = lap_error_selected_index[~isin2] # x''_t \ x'_t
     
             # 3. 计算平衡比例 p (对应论文 Eq. 5)
             # 根据低频和高频的平均误差决定从哪个差集中选取更多
             l2_error_mean = error_map.mean()
             lap_error_mean = laplace_error_map.mean()
             p = l2_error_mean / (lap_error_mean + l2_error_mean) # 低频误差占比
             l2_remain_num = int(remain_num * p) # 从低频差集中选取的数量 pl
             l2_remain_num = min(l2_remain_num, l2_remain_index.shape[0])
             lap_remain_num = remain_num - l2_remain_num # 从高频差集中选取的数量 (1-p)l
     
             # 4. 合并得到最终选择的索引 (对应后代 w_t)
             all_selected_index = torch.cat(
                 [
                     lap_remain_index[-lap_remain_num:], # 从高频差集中选
                     l2_remain_index[-l2_remain_num:], # 从低频差集中选
                     selected_index,                   # 加入交集
                 ]
             )
             # 重新组合排序索引，将选中的放在最后 (高适应度)
             all_remain_index = sorted_map_index[
                 ~torch.isin(sorted_map_index, all_selected_index)
             ]
             select_sorted_index = torch.cat([all_remain_index, all_selected_index])
             # 更新缓存的排序索引，供后续选择和突变使用
             self.book["sorted_map_index"] = select_sorted_index
     ```
   
3. 

   **增强无偏突变 (Augmented Unbiased Mutation)** 

   - **思路总结**: 为了缓解因重复使用缓存适应度而可能产生的选择偏差，该方法在每次迭代中引入了随机性 12。具体来说，除了选择适应度高的坐标外，还会从**未被选中**的坐标（适应度较低的）中**随机均匀采样**一小部分（比例由 $\alpha$ 控制），并将它们也加入到最终送入网络训练的坐标子集 $z_t$ 中 。这有助于维持种群多样性，减轻训练偏差。

   - **代码/伪代码实现** (源自 `img_sampling_trainer.py` 的 `_evos_get_selection_mask` 方法):

     

     ```Python
     def _evos_get_selection_mask(self, epoch):
         # 计算当前 epoch 的突变比例 αk/N (mutation_ratio 对应 αk)
         mutation_ratio = self._evos_get_mutation_ratio(epoch)
         # 计算主要选择比例 (采样比例 - 突变比例)
         first_select_ratio = self.cur_use_ratio - mutation_ratio
         # 计算主要选择数量 k - αk
         first_select_num = int(first_select_ratio * self.sample_num)
     
         # 获取缓存的排序索引
         sorted_map_index = self.book["sorted_map_index"]
         # 选择适应度最高的点作为主要选择 (对应 w_t 中的非突变部分)
         first_select_indices = sorted_map_index[-first_select_num:]
     
         # --- 增强无偏突变 (对应论文 Eq. 7) ---
         # 计算突变数量 αk
         mutation_num = int(mutation_ratio * self.sample_num)
         # 获取未被主要选择选中的索引 (x \ w_t)
         remain_indices = sorted_map_index[:-first_select_num]
         # 从未被选中的索引中随机采样 mutation_num 个
         sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[
             :mutation_num
         ]
         # 得到突变的索引 m_t
         mutation_indicies = remain_indices[sample_index]
     
         # --- 合并得到最终用于训练的索引 z_t = w_t U m_t ---
         selected_indices = torch.cat([first_select_indices, mutation_indicies])
     
         # --- 创建掩码 ---
         # _mask 标记未被选中的点 (用于跨频监管)
         _mask = torch.ones(self.sample_num, dtype=torch.bool, device=self.device)
         _mask[selected_indices] = False
         self.book["freeze_mask"] = _mask
     
         # selection_mask 标记被选中的点 (用于实际采样)
         selection_mask = torch.zeros(
             self.sample_num, dtype=torch.bool, device=self.device
         )
         selection_mask[selected_indices] = True
         # 返回选择掩码
         return selection_mask
     ```

此外，文章还提到了**跨频监管 (Cross-Frequency Supervision)** 作为交叉操作的补充，将拉普拉斯损失也加入到总损失中（在代码 `_sampler_compute_loss` 和 `_evos_cross_frequency_loss` 中实现）。