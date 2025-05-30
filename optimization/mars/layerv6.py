from peft.tuners.tuners_utils import BaseTunerLayer
import torch
import torch.nn as nn
import numpy as np

from research.experiments import generate_complementary_matrices

class MarsLayer(BaseTunerLayer):

    adapter_layer_names = ("up_project",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer

        self.up_project = nn.ModuleDict({})
        self.down_project = nn.ModuleDict({})

        if kwargs.get("mixture", False):
            # TODO
            self.mixture = nn.ModuleDict({})
            #self.mixture_vector = nn.ParameterDict({})
            self.mixtures = True
        else:
            self.mixtures = False

        if self.mixtures:
            self.tunable_mixture = True
            self.adapter_layer_names = ("up_project", "mixture")
        else:
            self.tunable_mixture = False

        # Analysis
        self.count = 0
        self.window_size = kwargs.get("window_size", 100)
        self.current_window = 0

        self.analysis_phase = False
        self.svd_decompose = False
        self.sparse_channels = False
        self.scale_adapters = False
        self.manual_seed = True
        
        self.switch = True
        self.combined = False
        self.method = 'orth_init' # orthogonal, none
        self.adaptive_subspace = self.method == 'complementary'
        self.complementary_learning = self.method == 'complementary'
        self.orthogonal_learning = self.method == 'orthogonal'

        self.switch_interval = 100
        self.combine_interval = 200

        self.channel_stats = {
            "in_magnitudes": [],  
            "out_magnitudes": [],
            "l1_weight_down_project": [],
            "l1_weight_up_project": [],
            "base_scaling": [],
            "adapter_scaling": [],
            "gradient_norm": []
        }

        # Rolling buffers for the current window
        self.in_activation_stats = []
        self.out_activation_stats = []
        self.base_scaling_factors = []
        self.adapter_scaling_factors = []
        self.weight_magnitudes = []

    def update_layer(self, original_weights, adapter_name, rank, alpha, subspace, mixture, seed=42):
        self.internal_rank = rank

        self.in_features = original_weights.in_features
        self.out_features = original_weights.out_features
        # TODO
        self.use_mixture_vector = False
        self.alpha = alpha / self.internal_rank

        if mixture:
            self.adapter_layer_names = ("up_project", "mixture",)

        gen = None
        gen2 = None

        # TODO: Replace seed
        if self.manual_seed:
            gen = torch.Generator()
            gen.manual_seed(seed)

            gen2 = torch.Generator()
            gen2.manual_seed(seed + 1)

        # Split S into two equal orthogonal complementary subspaces
        A1, A2 = generate_complementary_matrices(m=self.in_features, n=subspace[0], gen1=gen, gen2=gen2)

        A = None

        if subspace[1] == 1:
            A = A2[:, :self.internal_rank]
        else:
            A = A1

        """
        orthogonality_check = A1.T @ A2[:, :2]
        print("A1^T A2 result:\n", orthogonality_check)
        print("Max absolute value in A1^T A2:", torch.abs(orthogonality_check).max())
        print("Mean value in A1^T A2:", orthogonality_check.mean().item())

        # Check distributions
        print("A1 mean:", A1.mean().item(), "A std:", A1.std().item())
        print("A2 mean:", A2.mean().item(), "B std:", A2.std().item())
        """

        self.down_project[adapter_name] = nn.Linear(self.in_features, self.internal_rank, bias=False)
        self.down_project[adapter_name].weight.data = A.T.contiguous() * self.alpha
        self.down_project[adapter_name].weight.requires_grad = False

        if mixture or self.mixtures:
            mixture_linear = nn.Linear(self.internal_rank, self.internal_rank, bias=False)
            mixture_linear.weight.data = torch.eye(self.internal_rank).contiguous()
            torch.nn.init.normal_(mixture_linear.weight.data, mean=0, std=1/self.internal_rank).contiguous()
            mixture_linear.weight.requires_grad = self.tunable_mixture
            self.mixture[adapter_name] = mixture_linear
            
            #self.mixture_vector[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty(self.in_features), std=1/self.internal_rank))

        self.up_project[adapter_name] = nn.Linear(self.internal_rank, self.out_features, bias=False)
        self.up_project[adapter_name].weight.data = torch.nn.init.zeros_(self.up_project[adapter_name].weight.data).contiguous()
        self.up_project[adapter_name].weight.requires_grad = True

        #self.down_project[adapter_name] = nn.Parameter(A, requires_grad=False)
        #if mixture or self.mixtures:
        #    self.mixture[adapter_name] = nn.Parameter(torch.eye(self.internal_rank), requires_grad=self.tunable_mixture)
        #self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.internal_rank, original_weights.out_features))), requires_grad=True)

        """   
        if self.svd_decompose:
            U, S, Vt = torch.linalg.svd(original_weights.weight.T, full_matrices=False)
            
            # Truncate to the desired rank
            U_truncated = U[:, :self.inner_rank]
            S_truncated = S[:self.inner_rank]
            Vt_truncated = Vt[:self.inner_rank, :]

            # Compute down_project = U * Sigma (truncated)
            down_project = U_truncated @ torch.diag(S_truncated)

            up_project = Vt_truncated

            self.down_project[adapter_name] = nn.Parameter(down_project)
            self.up_project[adapter_name] = nn.Parameter(up_project)
        elif self.sparse_channels:

            self.input_channels = int(self.in_features)
            self.output_channels = int(self.out_features * 0.75)

            self.down_project[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.input_channels, self.inner_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.inner_rank, self.output_channels))))

            input_ordering = torch.randint(size=(self.input_channels,), high=self.in_features)
            self.register_buffer("input_ordering", input_ordering)
            output_ordering = torch.randint(size=(self.output_channels,), high=self.out_features)
            self.register_buffer("output_ordering", output_ordering)

            self.in_features = self.input_channels
            self.out_features = self.output_channels
        elif self.method == 'orthogonal':

            # Split S into two equal orthogonal complementary subspaces

            A1, A2 = generate_complementary_matrices(m=self.in_features, n=self.internal_rank, gen1=gen, gen2=gen2)

            self.down_project[adapter_name] = nn.Parameter(A1, requires_grad=False)


            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.internal_rank, original_weights.out_features))), requires_grad=True)
            
        elif self.method == 'complementary':

            # Split S into two complementary subspaces
            self.subspace_1 = None
            self.subspace_2 = None

            # Initialize trainable bias
            self.bias_terms[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty(self.internal_rank)), requires_grad=True)

            self.current_subspace = self.subspace_1  # Start with S1

            self.down_project = nn.Parameter(torch.nn.init.normal_(torch.empty((original_weights.in_features, self.subspace_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            self.subspace[adapter_name] = nn.Parameter(torch.nn.init.normal_(torch.empty((self.subspace_rank, self.internal_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.internal_rank, original_weights.out_features))))
        elif self.method == 'orth_init':
            # Split S into two equal orthogonal complementary subspaces

            A1, A2 = generate_complementary_matrices(m=self.in_features, n=subspace[0], gen1=gen, gen2=gen2)

            A = None

            if subspace[1] == 1:
                A = A2[:, :self.internal_rank]
            else:
                A = A1

            orthogonality_check = A1.T @ A2[:, :2]
            print("A1^T A2 result:\n", orthogonality_check)
            print("Max absolute value in A1^T A2:", torch.abs(orthogonality_check).max())
            print("Mean value in A1^T A2:", orthogonality_check.mean().item())

            # Check distributions
            print("A1 mean:", A1.mean().item(), "A std:", A1.std().item())
            print("A2 mean:", A2.mean().item(), "B std:", A2.std().item())

            self.down_project[adapter_name] = nn.Parameter(A, requires_grad=False)

            if mixture or self.mixtures:
                self.mixture[adapter_name] = nn.Parameter(torch.eye(self.internal_rank), requires_grad=self.tunable_mixture)

            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.internal_rank, original_weights.out_features))), requires_grad=True)
        else:
            self.down_project = nn.Parameter(torch.nn.init.normal_(torch.empty((original_weights.in_features, self.internal_rank)), mean=0, std=0.1, generator=gen), requires_grad=False)
            self.up_project[adapter_name] = nn.Parameter(torch.nn.init.zeros_(torch.empty((self.internal_rank, original_weights.out_features))), requires_grad=True)
            
        self.alpha = 100.0

        self.adapter_name = adapter_name
        """

        self.adapter_name = adapter_name
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

class Linear(nn.Module, MarsLayer):

    adapter_layer_names = ("up_project",)

    def __init__(self,
                 base_layer,
                 adapter_name,
                 r,
                 alpha,
                 subspace=(8, 0),
                 mixture=False,
                 seed=42,
                 fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 **kwargs) -> None:
        
        super().__init__()
        MarsLayer.__init__(self, base_layer, mixture=mixture, subspace=subspace, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.update_layer(
            base_layer, adapter_name, r, alpha, subspace, mixture, seed
        )

        if mixture:
            self.adapter_layer_names = ("up_project","mixture",)

    def _store_channel_metrics(self, active_adapter):
        """
        Store the per-channel metrics for the current window.
        """
        # Average collected statistics over the window
        avg_in_magnitudes = np.mean([stat['magnitude'] for stat in self.in_activation_stats], axis=0)
        avg_out_magnitudes = np.mean([stat['magnitude'] for stat in self.out_activation_stats], axis=0)
        avg_weight_magnitudes = {
            "down_norm": np.mean([wm["down_norm"] for wm in self.weight_magnitudes], axis=0),
            "up_norm": np.mean([wm["up_norm"] for wm in self.weight_magnitudes], axis=0)
        }
        #avg_base_scaling = np.mean(self.base_scaling_factors, axis=0)
        

        # Store the averaged statistics
        self.channel_stats["in_magnitudes"].append(avg_in_magnitudes)
        self.channel_stats["out_magnitudes"].append(avg_out_magnitudes)
        self.channel_stats["l1_weight_down_project"].append(avg_weight_magnitudes["down_norm"])
        self.channel_stats["l1_weight_up_project"].append(avg_weight_magnitudes["up_norm"])
        #self.channel_stats["base_scaling"].append(avg_base_scaling)

        if self.scale_adapters:
            avg_adapter_scaling = np.mean(self.adapter_scaling_factors, axis=0)
            self.channel_stats["adapter_scaling"].append(avg_adapter_scaling)

        norm = self.up_project[active_adapter].grad.detach().norm(p=2).item()

        # Store up_projection gradient norms
        self.channel_stats["gradient_norm"].append(
            norm
        )

        # Reset rolling buffers
        self.in_activation_stats.clear()
        self.out_activation_stats.clear()
        self.weight_magnitudes.clear()
        self.adapter_scaling_factors.clear()
        self.base_scaling_factors.clear()
        self.current_window += 1

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            
            result = self.base_layer(x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.up_project.keys():
                    continue
                
                # Apply mixture if available
                if self.mixtures:
                    # Use the mixture Linear layer
                    if self.use_mixture_vector:
                        result = result + self.up_project[active_adapter](
                            self.down_project[active_adapter](x * self.mixture_vector[active_adapter])
                        )
                    else:
                        result = result + self.up_project[active_adapter](self.mixture[active_adapter](self.down_project[active_adapter](x)))
                else:
                    # Skip mixture, go directly to up projection
                    result = result + self.up_project[active_adapter](self.down_project[active_adapter](x)) * self.alpha
                
                # Add the adapter output to the result
                #result += up_output
                """
                if self.sparse_channels:
                    x_sliced = torch.index_select(x, dim=2, index=self.input_ordering)

                    out = x_sliced @ self.down_project @ self.up_project[active_adapter]

                    if self.scale_adapters:
                        adapter_scaler = self.scaler(self.alpha * self.adapter_scaling[active_adapter])
                        out = adapter_scaler * out
                    
                    result.scatter_add_(dim=2, index=self.output_ordering.expand(batch_size, seq_len, -1), src=out)
                elif self.complementary_learning:
                    out = x @ self.down_project[:, :self.subspace_rank] @ self.current_subspace
                    out += self.bias_terms[active_adapter]

                    out = out @ self.up_project[active_adapter]

                    result += out
                elif self.orthogonal_learning:
                    
                    if not self.combined:
                        if self.switch:
                            out = x @ self.down_project_1 @ self.up_project_1[active_adapter]
                        else:
                            out = x @ self.down_project_2 @ self.up_project_2[active_adapter]
                    else:
                        out = x @ self.down_project_combined @ self.up_project_combined[active_adapter]
                    result += out
                else:

                    

                if self.analysis_phase:

                    # Per-channel metrics
                    in_magnitude = torch.mean(torch.abs(x), dim=(0, 1))
                    out_magnitude = torch.mean(torch.abs(out), dim=(0, 1))

                    self.in_activation_stats.append({
                        'magnitude': in_magnitude.cpu().detach().numpy()
                    })
                    self.out_activation_stats.append({
                        'magnitude': out_magnitude.cpu().detach().numpy()
                    })

                    # Compute weight magnitudes for each matrix
                    down_norm = torch.norm(self.down_project, p=1, dim=1)  # Shape: (input_channels,)
                    up_norm = torch.norm(self.up_project[active_adapter], p=1, dim=0)      # Shape: (output_channels,)

                    # Store the individual norms if needed
                    self.weight_magnitudes.append({
                        "down_norm": down_norm.cpu().detach().numpy(),
                        "up_norm": up_norm.cpu().detach().numpy()
                    })

                    #self.base_scaling_factors.append(base_scaler.cpu().detach().numpy())
                    if self.scale_adapters:
                        self.adapter_scaling_factors.append(adapter_scaler.cpu().detach().numpy())

                    self.count += 1
                    # Store metrics if the window size is reached
                    if self.count % self.window_size == 0:
                        self._store_channel_metrics(active_adapter)
                    """

            result = result.to(previous_dtype)
        return result
    
    def get_delta_weight(self, adapter) -> torch.Tensor:
        # This function is introduced in newer PEFT versions. we modify this function instead of modifying
        # the merge function (as we did previously for version 0.4.0 of PEFT).
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # TODO: Implement merging
        pass

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "mars." + rep