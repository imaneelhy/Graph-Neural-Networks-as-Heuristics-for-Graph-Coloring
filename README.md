# Graph Neural Networks for Graph Coloring

This project studies whether a **Graph Neural Network (GNN)** can learn useful color
suggestions for graph-coloring instances and **reduce the search effort** of a classical
backtracking solver.

The code generates nontrivial graph-coloring instances, trains a **Residual GCN (ResGCN)**
to predict node colors, and plugs the GNN into a backtracking solver as a learned
heuristic.

---

## Problem

We consider **4-coloring** of undirected graphs:

> Assign one of 4 colors to each node so that no edge connects two nodes of the same color.

This is a classic **Constraint Satisfaction Problem (CSP)** and is closely related to
scheduling and resource allocation.

---

## Data & Classical Solver

We generate graphs using an **Erdős–Rényi** model:

- Number of nodes: `n ∈ [10, 18]`
- Edge probability: `p ∈ [0.20, 0.45]`
- Number of colors: `K = 4`

To label the data we use a **classical backtracking solver** with a
high-degree-first node ordering:

1. Generate a random graph.
2. Run the solver with 4 colors.
3. Keep the instance only if it:
   - is 4-colorable, and
   - requires between **5 and 150 backtracks** with the degree heuristic  
     (nontrivial but still solvable).

We build three splits:

- **Train**: 1000 graphs  
- **Validation**: 200 graphs  
- **Test**: 200 graphs  

Example train stats (from the run in `results/easy_resgcn_run.txt`):

- Train graph sizes: 16–18 nodes
- Train difficulty (backtracks):
  - min = 5
  - median = 15
  - max = 148

Global train label distribution (4 colors):

- `[4006, 4203, 4149, 3570]` (reasonably balanced)

---

## Model: Residual GCN for Node Coloring

Each graph is represented with:

- Adjacency matrix `A`
- Node features (per node):
  - normalized degree
  - clustering coefficient
  - normalized 2-hop degree
  - degree / n

The main model is a **Residual GCN (ResGCN)**:

- Input projection: `R^4 → R^256`
- 5 residual GCN blocks with LayerNorm
- Dropout: 0.1
- Output layer: `R^256 → R^4` (logits for 4 colors)

Training details:

- Loss: cross-entropy over node labels
- Optimizer: Adam
- LR: 3e-3, weight decay: 5e-5
- Early stopping on validation node accuracy

We also support:
- A plain multi-layer GCN
- An MLP baseline that ignores the graph structure (for ablations)

All models share the same training and evaluation pipeline.

---

## How the GNN is Used

We evaluate the GNN in two ways:

1. **GNN-only**  
   - Take argmax of the GNN’s predicted distribution at each node.
   - Measure:
     - node-wise accuracy against the solver’s solution,
     - fraction of edges that violate the coloring constraint,
     - fraction of graphs that are perfectly colored (no violations).

2. **GNN inside the solver (learned heuristic)**  
   We plug the GNN into the backtracking solver in several variants:

   - `random`: random node order, no learning.
   - `degree`: high-degree-first node order (classical baseline).
   - `gnn_ordering`: branch on nodes ordered by GNN confidence.
   - `gnn_warm_start`: keep degree ordering, but use the GNN to rank colors
     at each node.
   - `gnn_both`: GNN for both node ordering and color ranking.

For each method we report:

- success rate (fraction of test instances solved),
- average number of backtracks,
- average number of recursive steps,
- average runtime.

---

## Results (Easy Regime, ResGCN)

**Dataset & setup**

- Regime: `easy`
- `n ∈ [10, 18]`, `p ∈ [0.20, 0.45]`
- Backtracking hardness: 5–150 (degree heuristic)
- Model: ResGCN (5 blocks, hidden size 256)

### GNN-only performance

On the held-out test set:

- **Node accuracy**: **0.519**  
  (random 4-coloring baseline: 0.25)

- **Edge violation rate**: **0.136**  
  (fraction of edges whose endpoints share a color; random baseline ≈ 0.25)

- **Valid colorings (argmax)**: **0.01**  
  (≈1% of graphs are perfectly 4-colored by the GNN without any search)

This shows the GNN learns nontrivial structure: it approximately **doubles** node
accuracy compared to random and almost halves the rate of constraint violations.

### Solver performance with/without GNN

On the same test set:

- **Random ordering**

  - success_rate: 0.950  
  - avg_backtracks: 792.55  

- **Degree heuristic (classical baseline)**

  - success_rate: 1.000  
  - avg_backtracks: 26.08  

- **GNN-guided node ordering (`gnn_ordering`)**

  - success_rate: 0.965  
  - avg_backtracks: 405.66  

- **GNN-guided color ranking (`gnn_warm_start`)**

  - success_rate: 1.000  
  - avg_backtracks: **19.89**  

- **GNN for node ordering + colors (`gnn_both`)**

  - success_rate: 0.965  
  - avg_backtracks: 405.66  

The key observation is that:

> Using the GNN **only** for color ranking within a classical degree ordering
> reduces average backtracking from **26.1** to **19.9**,  
> corresponding to roughly **24% fewer backtracks** on this regime.

However, using the GNN to decide the **node ordering** (`gnn_ordering`, `gnn_both`)
actually hurts performance, with higher backtracking and slightly lower success rate.

---

## How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/imaneelhy/gnn-graph-coloring-solver.git
   cd gnn-graph-coloring-solver
