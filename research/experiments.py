import numpy as np
import torch
from transformers import AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import torch
import safetensors.torch
import os
from scipy.linalg import subspace_angles
import research.cca_core as cca_core


MODEL_ID = "TinyLlama/TinyLlama_v1.1"

import torch
def create_orthogonal_matrices(m, num_matrices, mean=0.0, std=1.0, initial_matrix=None, device='cpu', generator=None):
    """
    Create mutually orthogonal matrices using null space construction.
    
    This approach is based on the fact that if B is in the null space of A^T,
    then A^T @ B = 0, which means the matrices are orthogonal.
    
    Args:
        m (int): Dimension of each square matrix (m x m)
        num_matrices (int): Number of matrices to generate
        mean (float): Mean for random initialization
        std (float): Standard deviation for random initialization  
        initial_matrix (torch.Tensor, optional): Initial matrix of size (m, m)
        device (str): Device to create tensors on
        generator (torch.Generator, optional): Random number generator
    
    Returns:
        torch.Tensor: Stacked matrices of size (m, m * num_matrices)
    """
    
    if initial_matrix is not None:
        if initial_matrix.shape != (m, m):
            raise ValueError(f"Initial matrix must be of size ({m}, {m})")
        A = initial_matrix.to(device)
        remaining = num_matrices - 1
    else:
        # Generate first matrix with controlled rank
        # We need null_dim >= remaining matrices
        # So rank should be <= m - (num_matrices - 1)
        remaining = num_matrices - 1
        max_rank = m - remaining
        
        if max_rank <= 0:
            raise ValueError(f"Cannot fit {num_matrices} matrices of size ({m}, {m}). Maximum possible: {m}")
        
        # Create a rank-deficient matrix by construction: A = U @ V^T
        # where U is (m, max_rank) and V is (m, max_rank)
        U = torch.nn.init.normal_(
            torch.empty((m, max_rank), device=device),
            mean=mean, std=std, generator=generator
        )
        V = torch.nn.init.normal_(
            torch.empty((m, max_rank), device=device),
            mean=mean, std=std, generator=generator
        )
        A = U @ V.T  # This has rank <= max_rank
    
    if remaining == 0:
        return A
    
    # Compute QR decomposition to get null space basis
    Q, _ = torch.linalg.qr(A, mode='complete')
    
    # The rank determines how much null space we have
    rank_A = torch.linalg.matrix_rank(A).item()
    null_dim = m - rank_A
    
    if null_dim == 0:
        raise ValueError(f"Matrix A is full rank (rank={rank_A}). No null space available for orthogonal matrices.")
    
    # Check if we have enough null space dimensions
    if remaining > null_dim:
        raise ValueError(f"Cannot generate {remaining} additional matrices. Null space dimension is only {null_dim}.")
    
    # Extract null space basis (last null_dim columns of Q)
    null_basis = Q[:, rank_A:]  # Shape: (m, null_dim)
    
    # Generate remaining matrices in the null space
    matrices = [A]
    
    for i in range(remaining):
        # Generate random coefficients for the null space basis
        coeffs = torch.nn.init.normal_(
            torch.empty((null_dim, m), device=device),
            mean=mean, std=std, generator=generator
        )
        
        # Create matrix in null space: null_basis @ coeffs
        B = null_basis @ coeffs  # Shape: (m, m)
        matrices.append(B)
    
    # Stack all matrices horizontally
    result = torch.cat(matrices, dim=1)
    return result

def verify_orthogonality(stacked_matrices, m, num_matrices, tolerance=1e-5):
    """
    Verify that the matrices are mutually orthogonal.
    """
    matrices = [stacked_matrices[:, i*m:(i+1)*m] for i in range(num_matrices)]
    
    print("Matrix statistics:")
    for i, matrix in enumerate(matrices):
        mean_val = torch.mean(matrix).item()
        std_val = torch.std(matrix).item()
        print(f"  Matrix {i}: mean = {mean_val:.4f}, std = {std_val:.4f}")
    
    print("\nOrthogonality verification:")
    all_orthogonal = True
    
    for i in range(num_matrices):
        for j in range(i+1, num_matrices):
            # Check A^T * B and B^T * A
            product1 = torch.matmul(matrices[i].T, matrices[j])
            product2 = torch.matmul(matrices[j].T, matrices[i])
            
            max_val1 = torch.max(torch.abs(product1)).item()
            max_val2 = torch.max(torch.abs(product2)).item()
            
            is_orthogonal = max_val1 < tolerance and max_val2 < tolerance
            print(f"  Matrices {i} and {j}: A^T*B max = {max_val1:.2e}, B^T*A max = {max_val2:.2e} {'✓' if is_orthogonal else '✗'}")
            
            if not is_orthogonal:
                all_orthogonal = False
    
    return all_orthogonal

def visualize_llama_layer_cosine_similarity(model_name: str, layer_name: str, output_file: str):
    """
    Load a Llama model and compute a heatmap of cosine similarities
    for a specific sublayer's weights across all decoder layers.
    
    Args:
        model_name (str): The Hugging Face model to load (e.g., 'tinyllama-1.1B').
        layer_name (str): The name of the sublayer to analyze (e.g., 'q_proj').
        output_file (str): Path to save the generated heatmap.
    """
    # Load the Llama model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Extract decoder layers
    decoder_layers = model.model.layers  # This is a ModuleList of LlamaDecoderLayer
    
    # Collect the weights of the specified sublayer from all decoder layers
    weights = []
    for i, layer in enumerate(decoder_layers):
        # Access the sublayer (e.g., q_proj)
        sublayer = getattr(layer.mlp, layer_name, None)
        if sublayer is not None:
            weights.append(sublayer.weight.detach().cpu().numpy())
        else:
            raise ValueError(f"Sublayer '{layer_name}' not found in decoder layer {i}.")
    
    # Compute pairwise cosine similarities for all weights
    num_layers = len(weights)
    similarity_matrix = torch.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            w1 = weights[i].reshape(1, -1)
            w2 = weights[j].reshape(1, -1)
            similarity = cosine_similarity(w1, w2)[0, 0]
            similarity_matrix[i, j] = float(similarity)
    
    # Convert to numpy array for visualization
    similarity_matrix = similarity_matrix.numpy()
    """
    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, interpolation='nearest', cmap='coolwarm')
    plt.colorbar(label='Cosine Similarity')
    plt.title(f"Cosine Similarity Heatmap: {layer_name}")
    plt.xlabel("Decoder Layer Index")
    plt.ylabel("Decoder Layer Index")
    
    # Add layer indices as tick marks
    plt.xticks(range(num_layers), range(num_layers))
    plt.yticks(range(num_layers), range(num_layers))
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()
    print(f"Heatmap saved to {output_file}")
    """

def generate_complementary_matrices(m, n, mean=0, std=0.1, gen1=None, gen2=None):
    if m <= n:
        raise ValueError("m must be strictly greater than n to ensure a nontrivial null space.")

    # Step 1: Generate a random A (m x n)
    A = torch.nn.init.normal_(torch.empty((m, n)), mean=mean, std=std, generator=gen1)

    # Step 2: Compute the null space of A^T using QR decomposition
    Q, R = torch.linalg.qr(A, mode='complete')  # Compute full QR decomposition
    null_space_B = Q[:, n:]  # Last (m-n) columns of Q form an orthonormal basis for the null space of A^T

    # Step 3: Construct a nontrivial B from the null space
    B = null_space_B @ torch.nn.init.normal_(torch.empty((m - n, n)), mean=mean, std=std, generator=gen2)
    return A, B


def train_peft_model(peft_model, dataloader, tokenizer, num_epochs=3, lr=5e-4, max_steps=500, device=None):
    """Training loop for the PEFT LoRA model while tracking token difficulty."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    loss_fct = CrossEntropyLoss(reduction="none")  # Per-token loss

    vocab_size = tokenizer.vocab_size
    token_difficulty = torch.zeros(vocab_size, device=device)  # Difficulty vector
    
    peft_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()  # Labels are input_ids (shifted internally)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Compute per-token loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())  # (batch_size, seq_len-1)

            # Compute batch loss
            batch_loss = per_token_loss.mean()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

            # Track token difficulty
            with torch.no_grad():
                for i in range(per_token_loss.size(0)):  # Iterate over batch
                    for j in range(per_token_loss.size(1)):  # Iterate over seq_len
                        token_id = shift_labels[i][j].item()
                        token_difficulty[token_id] += per_token_loss[i, j]  # Aggregate loss per token
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {batch_loss.item():.4f}")
            
            if step > max_steps:
                break

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")
        if step > max_steps:
            break
    
    return token_difficulty

def gradual_scaled_loss(inputs, logits, keytoken_scales, current_step, total_steps, warmup_ratio=0.1):
    """
    Computes the gradually weighted loss based on token difficulties.

    Args:
        inputs (torch.Tensor): Input token IDs.
        logits (torch.Tensor): Model output logits.
        keytoken_scales (torch.Tensor): Scaled difficulty tensor (shape: vocab_size).
        current_step (int): Current training step.
        total_steps (int): Total training steps.
        warmup_ratio (float): Ratio of total steps before scaling starts.

    Returns:
        torch.Tensor: Computed loss with gradual weighting.
    """

    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Compute per-token loss
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1))

    # Compute gradual scaling factor
    warmup_steps = int(warmup_ratio * total_steps)
    if current_step < warmup_steps:
        scaling_factor = 0  # No scaling during warmup
    else:
        scaling_factor = (current_step - warmup_steps) / (total_steps - warmup_steps)

    # Apply scaled token difficulty weights
    token_weights = 1.0 + keytoken_scales[inputs] * scaling_factor  # Gradual scaling

    # Compute weighted loss
    weighted_loss = (loss_per_sample * token_weights[:, 1:]).mean()
    
    return weighted_loss

def scale_token_difficulty(token_difficulty, beta=5.0):
    """
    Apply sigmoid-based smoothing to the token difficulty tensor.

    Args:
        token_difficulty (torch.Tensor): A tensor of shape [vocab_size] with aggregated losses.
        beta (float): Controls the sharpness of scaling.

    Returns:
        torch.Tensor: Scaled token weights in (0, 1).
    """
    token_difficulty = token_difficulty / token_difficulty.max()  # Normalize to [0,1]
    scaled_difficulty = torch.sigmoid(beta * (token_difficulty - 0.5))  # Smooth scaling
    return scaled_difficulty

def train_peft_model_with_gradual_loss(peft_model, dataloader, tokenizer, token_difficulty, 
                                       num_epochs=3, lr=5e-4, max_steps=500, device=None):
    """Training loop for PEFT LoRA model with gradual token difficulty weighting."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model.to(device)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    #loss_fct = CrossEntropyLoss(reduction="none")  # Per-token loss

    #vocab_size = tokenizer.vocab_size
    token_scales = token_difficulty / token_difficulty.max()  # Normalize difficulties to [0,1]

    peft_model.train()
    total_steps = len(dataloader) * num_epochs
    current_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()  # Labels are input_ids (shifted internally)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # Compute gradual loss
            loss = gradual_scaled_loss(input_ids, logits, token_scales, current_step, total_steps)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            
            current_step += 1
            if current_step > max_steps:
                break

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished, Avg Loss: {avg_loss:.4f}")
        if current_step > max_steps:
            break

def load_adapters(adapter_dir):
    """Load adapter weights from a safetensors file."""
    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    return safetensors.torch.load_file(adapter_path)


def compute_grassmann_similarity(A, B, top_k=None):
    """
    Compute the normalized subspace similarity based on Grassmann distance
    between two adapter matrices.
    
    Parameters:
    -----------
    A, B : numpy.ndarray
        Input matrices (the AB_product matrices from your comparison loop)
    top_k : int or None
        Number of top singular vectors to consider. If None, uses the minimum
        rank of both matrices
    
    Returns:
    --------
    float
        Average similarity measure in range [0, 1], where:
        - 1 represents complete overlap of subspaces
        - 0 represents complete separation
    """
    # Perform SVD to get left singular vectors
    U_A, s_A, _ = np.linalg.svd(A, full_matrices=False)
    U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
    
    # Determine effective rank based on singular values
    # This helps focus on the significant components
    effective_rank_A = sum(s_A > 1e-5)
    effective_rank_B = sum(s_B > 1e-5)
    
    # Determine how many singular vectors to use
    if top_k is None:
        # Use the minimum effective rank if not specified
        k = min(effective_rank_A, effective_rank_B)
    else:
        k = min(top_k, effective_rank_A, effective_rank_B)
    
    if k == 0:
        return 0.0  # No significant singular values
    
    # Take the top-k singular vectors
    U_k_A = U_A[:, :k]
    U_k_B = U_B[:, :k]
    
    # Compute the similarity measure
    product = U_k_A.T @ U_k_B
    frob_norm_squared = np.sum(np.square(product))
    
    # Normalize by k (min dimension)
    similarity = frob_norm_squared / k
    
    return similarity
def compute_subspace_similarity(A, B):
    """
    Compute the subspace similarity between two matrices using principal angles.
    Returns the largest principal angle similarity score.
    """
    # Ensure matrices have the same number of columns (latent space alignment)
    #min_dim = min(A.shape[1], B.shape[1])
    #A = A[:, :min_dim]
    #B = B[:, :min_dim]

    # Compute principal angles
    angles = subspace_angles(A.cpu().numpy(), B.cpu().numpy())

    # Convert angles to tensor before applying cosine
    angles_tensor = torch.tensor(angles, dtype=torch.float32)  # Convert to PyTorch tensor
    return torch.cos(angles_tensor).mean().item()  # Compute similarity

def compute_projection_distance(AB_A, AB_B):
    """
    Compute the Frobenius norm of the difference between the projection matrices
    of AB_A and AB_B. Here we compute the orthonormal bases for the column spaces,
    then form projection matrices P = U U^T.
    """
    # Compute SVD for AB_A and AB_B
    U_A, _, _ = torch.linalg.svd(AB_A, full_matrices=False)
    U_B, _, _ = torch.linalg.svd(AB_B, full_matrices=False)
    # Form projection matrices
    P_A = U_A @ U_A.T
    P_B = U_B @ U_B.T
    # Frobenius norm difference
    diff = torch.norm(P_A - P_B, p='fro')
    return diff.item()

def compute_svcca_similarity(AB_A, AB_B, k=None):
    """
    Compute SVCCA similarity between two matrices.
    
    Parameters:
    -----------
    AB_A, AB_B : torch.Tensor
        Input matrices
    k : int or None
        Number of top singular vectors to consider. If None, uses the minimum
        rank of both matrices
    
    Returns:
    --------
    float
        SVCCA similarity score in range [0, 1]
    """
    
    results = cca_core.get_cca_similarity(AB_A.cpu().numpy(), AB_B.cpu().numpy())

    print(results)

def compute_grassmann_similarity_matrix(A, B, max_k=None):
    """
    Compute the similarity matrix for different numbers of singular vectors,
    similar to the heatmap analysis in the LoRA paper.
    
    Parameters:
    -----------
    A, B : numpy.ndarray
        Input matrices (the AB_product matrices from your comparison loop)
    max_k : int or None
        Maximum number of singular vectors to consider. If None, uses
        the minimum rank of both matrices
    
    Returns:
    --------
    numpy.ndarray
        2D array containing similarity values for different i and j
    dict
        Dictionary with detailed similarity information
    """
    # Perform SVD to get left singular vectors
    U_A, s_A, _ = np.linalg.svd(A, full_matrices=False)
    U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
    
    # Determine effective rank based on singular values
    effective_rank_A = sum(s_A > 1e-5)
    effective_rank_B = sum(s_B > 1e-5)
    
    # Determine maximum k for analysis
    if max_k is None:
        max_k = min(effective_rank_A, effective_rank_B)
    else:
        max_k = min(max_k, effective_rank_A, effective_rank_B)
    
    similarity_matrix = np.zeros((max_k, max_k))
    
    for i in range(1, max_k + 1):
        for j in range(1, max_k + 1):
            # Take the top-i and top-j singular vectors
            U_i_A = U_A[:, :i]
            U_j_B = U_B[:, :j]
            
            # Compute the similarity measure
            product = U_i_A.T @ U_j_B
            frob_norm_squared = np.sum(np.square(product))
            
            # Normalize by min(i,j)
            similarity_matrix[i-1, j-1] = frob_norm_squared / min(i, j)
    
    # Compute additional metrics for detailed analysis
    info = {
        "similarity_matrix": similarity_matrix,
        "effective_rank_A": effective_rank_A,
        "effective_rank_B": effective_rank_B,
        "singular_values_A": s_A[:max_k],
        "singular_values_B": s_B[:max_k],
        "top_similarity": similarity_matrix[0, 0]  # Similarity of top singular vectors
    }
    
    return similarity_matrix, info

def visualize_similarity_matrices(detailed_analysis, output_dir=None):
    """
    Visualize the similarity matrices for each layer, similar to the heatmaps
    in the LoRA paper.
    
    Parameters:
    -----------
    detailed_analysis : dict
        Dictionary with detailed similarity analysis from analyze_adapter_similarity
    output_dir : str or None
        Directory to save visualizations, if None only displays them
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    for layer_name, info in detailed_analysis.items():
        sim_matrix = info["similarity_matrix"]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap="Blues", vmin=0, vmax=1, 
                   xticklabels=range(1, sim_matrix.shape[1]+1),
                   yticklabels=range(1, sim_matrix.shape[0]+1))
        
        plt.title(f"Grassmann Similarity - {layer_name}")
        plt.xlabel("Model B: Number of singular vectors")
        plt.ylabel("Model A: Number of singular vectors")
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{layer_name.replace('.', '_')}_similarity.png"))
            plt.close()
        else:
            plt.show()
            
        # Also plot singular value decay
        plt.figure(figsize=(10, 6))
        plt.plot(info["singular_values_A"], 'b-', label="Model A")
        plt.plot(info["singular_values_B"], 'r-', label="Model B")
        plt.title(f"Singular Value Decay - {layer_name}")
        plt.xlabel("Index")
        plt.ylabel("Singular Value")
        plt.yscale("log")
        plt.legend()
        
        # Save or show
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{layer_name.replace('.', '_')}_singular_values.png"))
            plt.close()
        else:
            plt.show()

def analyze_adapter_similarity(adapter_dir_a, adapter_dir_b, adapters="lora", detailed=True):
    """Compute the similarity between adapters from Model A and Model B."""
    adapter_a = load_adapters(adapter_dir_a)
    adapter_b = load_adapters(adapter_dir_b)

    similarity_scores = {}
    detailed_analysis = {}

    adapter_A_name = ("down_project" if adapters == "mars" else "lora_A")
    adapter_B_name = ("up_project" if adapters == "mars" else "lora_B")

    for key in adapter_a.keys():
        if adapter_A_name in key:
            key_b = key.replace(adapter_A_name, adapter_B_name)  # Find corresponding lora_B key

            if key_b in adapter_a and key_b in adapter_b:
                # Extract LoRA A and B matrices for both models
                A_modelA, B_modelA = adapter_a[key], adapter_a[key_b]
                A_modelB, B_modelB = adapter_b[key], adapter_b[key_b]

                if adapters == "mars":
                    # Compute A @ B for both models
                    AB_product_A = A_modelA @ B_modelA
                    AB_product_B = A_modelB @ B_modelB
                elif adapters == "lora":
                    # Compute A @ B for both models
                    AB_product_A = A_modelA.T @ B_modelA.T
                    AB_product_B = A_modelB.T @ B_modelB.T

                # Compute subspace similarity
                cosine_sim = compute_subspace_similarity(AB_product_A, AB_product_B)
                
                proj_distance = compute_grassmann_similarity(AB_product_A, AB_product_B)

                #svcca_sim = compute_svcca_similarity(AB_product_A, AB_product_B)

                # Detailed analysis if requested
                if detailed:
                    # Get the rank of each adapter
                    rank_A = min(A_modelA.shape)
                    rank_B = min(B_modelA.shape)
                    
                    print(f"  Model A adapter rank: {rank_A}, Model B adapter rank: {rank_B}")
                    
                    # Compute detailed similarity matrix for different numbers of singular vectors
                    sim_matrix, info = compute_grassmann_similarity_matrix(AB_product_A, AB_product_B)
                    
                    # Store detailed analysis
                    detailed_analysis[key] = {
                        "similarity_matrix": sim_matrix,
                        "effective_rank_A": info["effective_rank_A"],
                        "effective_rank_B": info["effective_rank_B"],
                        "singular_values_A": info["singular_values_A"].tolist(),
                        "singular_values_B": info["singular_values_B"].tolist(),
                        "top_singular_vector_similarity": info["top_similarity"]
                    }
                    
                    print(f"  Effective rank: Model A = {info['effective_rank_A']}, "
                          f"Model B = {info['effective_rank_B']}")
                    print(f"  Top singular vector similarity: {info['top_similarity']:.4f}")
                    
                    # Optionally visualize the similarity matrix here if using matplotlib

                similarity_scores[key] = {
                    "cosine_similarity": cosine_sim,
                    "projection_distance": proj_distance,
                    #"svcca_similarity": svcca_sim,
                }
                print(f"Layer {key}: Cosine Similarity = {cosine_sim:.4f}, "
                      f"Projection Distance = {proj_distance:.4f}, "
                      #f"SVCCA Similarity = {svcca_sim:.4f}")
                )

    return similarity_scores, detailed_analysis if detailed else None

"""
# Example usage
adapter_dir_a = "peft_model_a/"
adapter_dir_b = "peft_model_b/"
similarity_scores, detailed_analysis = analyze_adapter_similarity(adapter_dir_a, adapter_dir_b, adapters="lora")

# Visualize the results
visualize_similarity_matrices(detailed_analysis, output_dir="similarity_visualizations")
"""