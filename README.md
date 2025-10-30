# Towards Efficient Mixture of Experts: CKA-guided Knowledge Distillation

### **The Problem with MoE**

Mixture of Experts (MoE) has become popular because it allows **sparse activation** (only a few experts fire per input), giving the power of very large models but with lower compute per token. However, MoEs come with **challenges**:

1. **Training Instability**
    - Routing (deciding which experts fire) can be noisy, leading to unstable optimization.
    - Some experts collapse (never get trained well), while others get overloaded.
2. **Knowledge Fragmentation**
    - Unlike dense models (where all parameters learn shared representations), MoE scatters knowledge across experts.
    - This makes *distillation* tricky: how do you transfer teacher knowledge when the student has multiple experts with different roles?
3. **Redundancy & Inefficiency**
    - Many experts learn overlapping functions. This wastes capacity.
    - If you distill blindly, you may copy redundant representations into the student.
4. **Distillation Gap**
    - Standard KD only looks at **final outputs (logits)**, which misses the rich internal structure of how knowledge is divided across experts.
    - Without guiding the student to capture *representational similarity*, KD may underperform.

 **So what are we solving?**

We’re addressing the **distillation gap in MoE training** by proposing to use **CKA as an auxiliary signal** to align the student’s internal representations with the teacher’s — not just at the logit level. This could:

- Help student experts specialize better.
- Reduce redundancy by highlighting diverse expert representations.
- Stabilize training/routing.

---

### **Centered Kernel Alignment (CKA)**

**What it is:**

- A statistical method to compare two sets of representations (e.g., hidden activations from teacher vs student).
- Given two matrices of representations X and Y, CKA measures their similarity in a **scale-invariant way** (unlike raw cosine similarity).
- Mathematically, CKA is related to **Hilbert-Schmidt Independence Criterion (HSIC)**, and normalizes to avoid trivial scaling issues.

**Why it’s used in deep learning:**

- To compare **which layers of two models learn similar representations** (e.g., Kornblith et al. 2019).
- In KD, it has been used to encourage the student to mimic not just outputs, but also the “shape” of the teacher’s representation space.

---

### **How CKA Fits into our Work**

1. **Beyond Logits:**
    - Standard KD: Student matches teacher logits via KL-divergence.
    - Our method: Add a **CKA loss** between student and teacher hidden layers, so the student learns to align feature spaces.
2. **Expert-Level Alignment:**
    - In MoE, different experts handle different input subspaces.
    - CKA can quantify *which expert representations align best with which student layers*.
    - You can selectively distill only from *useful experts* → reduces redundancy.
3. **Pruning Experts:**
    - Compute inter-expert CKA among teacher experts. If some experts are too similar (high redundancy), you can prune them.
    - Student inherits knowledge only from diverse experts → efficiency boost.
4. **Routing Stability:**
    - CKA as auxiliary supervision can encourage experts to specialize consistently.
    - Helps avoid mode collapse where routing sends most tokens to one expert.

---

### **Summary:**
**Problem:**

“MoE models distribute knowledge across multiple experts, but existing distillation methods fail to capture this fragmented representation structure, leading to inefficient and unstable student models.”

**Proposed Solution:**

“By incorporating Centered Kernel Alignment (CKA) as an auxiliary loss during knowledge distillation, this thesis aims to better align the student’s internal representations with the teacher’s diverse expert outputs, thereby improving generalization, expert diversity, and training stability.”

**“This research aims not just at improving distillation theoretically, but also at enabling practical deployment of powerful models on resource-constrained devices such as mobile phones, embedded systems, and edge AI hardware.”**


** issues:
    1. for student raw, not finetuned and perofrming badly so chances are it might return empty or corrupted wrong answers in which case I trigger fallback and assign responses to be empty strings. in this case the following evaluation pipeline fails (Metrics like BLEU, ROUGE, BERTScore then fail (division by zero or empty list)). to handle that i...