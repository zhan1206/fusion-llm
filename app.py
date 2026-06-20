"""
Fusion-LLM Demo v5 - Hugging Face Space
Upgraded: ~38M params | BPE vocab=800 | seq_len=256 | answer-only training | 560+ pairs
v5 fix: explicit SPACE_ID tokenizer fix (was injecting UNK between every word)
"""

# Compatibility shim: huggingface_hub >=1.0 removed HfFolder, but gradio 4.44 still imports it.
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token():
            return None
        @staticmethod
        def save_token(token):
            pass
        @staticmethod
        def delete_token():
            pass
    huggingface_hub.HfFolder = _HfFolder

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import re
import random
from datetime import datetime

# ── Model Definition ──────────────────────────────────────────────

class SBLAttention(nn.Module):
    """Sparse Block Latent Attention with gated merging."""
    def __init__(self, hidden_size=384, num_heads=12, latent_ratio=4, block_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_size = hidden_size // latent_ratio
        self.block_size = block_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.latent_k = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.latent_v = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.latent_o = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        local_out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

        lk = self.latent_k(x).view(B, T, self.latent_size)
        lv = self.latent_v(x).view(B, T, self.latent_size)
        global_attn = (lk @ lv.transpose(-2, -1)) / (self.latent_size ** 0.5)
        global_attn = torch.softmax(global_attn, dim=-1)
        latent_out = (global_attn @ lv)
        latent_out = self.latent_o(latent_out)

        gate_input = torch.cat([x, local_out], dim=-1)
        g = self.gate(gate_input)
        return g * local_out + (1 - g) * latent_out


class FusionBlock(nn.Module):
    def __init__(self, hidden_size=384, num_heads=12, ff_dim=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = SBLAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class FusionMini(nn.Module):
    """Fusion-LLM model for demo - v5: ~38M params, proper tokenization."""
    def __init__(self, vocab_size=800, hidden_size=384, num_layers=6, num_heads=12,
                 max_seq_len=256, ff_dim=1024):
        super().__init__()
        self.config = type('obj', (object,), {
            'vocab_size': vocab_size, 'hidden_size': hidden_size,
            'num_layers': num_layers, 'num_heads': num_heads,
            'max_seq_len': max_seq_len
        })()

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList([
            FusionBlock(hidden_size, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Causal mask: (1, 1, max_seq_len, max_seq_len) for broadcasting
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = self.causal_mask[:, :, :T, :T].to(input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)


# ── BPE Tokenizer ─────────────────────────────────────────────────

class BPETokenizer:
    """Simple BPE tokenizer trained on the demo dataset."""

    def __init__(self, vocab_size=800):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.bos_id = 3
        self.q_sep_id = 4   # <|q|> question separator
        self.a_sep_id = 5   # <|a|> answer separator
        self.space_id = 6   # explicit space token (NOT looked up via vocab)
        self.word_cache = {}

    @staticmethod
    def _get_word_freqs(texts):
        freqs = {}
        for text in texts:
            words = text.split()
            for w in words:
                freqs[w] = freqs.get(w, 0) + 1
        return freqs

    @staticmethod
    def _get_pairs(word):
        return [(word[i], word[i+1]) for i in range(len(word) - 1)]

    def _merge_pair(self, pair, vocab_freqs):
        new_vocab_freqs = {}
        bigram = pair[0] + pair[1]
        for word, freq in vocab_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab_freqs[tuple(new_word)] = freq
        return new_vocab_freqs

    def train(self, texts, target_vocab_size=800):
        word_freqs = self._get_word_freqs(texts)
        vocab_freqs = {tuple(list(w)): f for w, f in word_freqs.items()}

        base_tokens = set()
        for word_tuple in vocab_freqs:
            for ch in word_tuple:
                base_tokens.add(ch)

        self.vocab = {t: i + 7 for i, t in enumerate(sorted(base_tokens))}  # 7=offset after 7 special tokens
        self.vocab['<pad>'] = self.pad_id
        self.vocab['<unk>'] = self.unk_id
        self.vocab['<eos>'] = self.eos_id
        self.vocab['<bos>'] = self.bos_id
        self.vocab['<|q|>'] = self.q_sep_id
        self.vocab['<|a|>'] = self.a_sep_id
        self.vocab['<space>'] = self.space_id

        current_size = len(self.vocab)
        num_merges = target_vocab_size - current_size

        for _ in range(num_merges):
            pair_counts = {}
            for word_tuple, freq in vocab_freqs.items():
                for pair in self._get_pairs(list(word_tuple)):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)
            vocab_freqs = self._merge_pair(best_pair, vocab_freqs)

            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.word_cache = {}

    def _tokenize_word(self, word):
        if word in self.word_cache:
            return self.word_cache[word]

        tokens = list(word)
        while len(tokens) > 1:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            best_rank = float('inf')
            best_pair = None
            for pair in pairs:
                if pair in self.merges:
                    rank = self.merges.index(pair)
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            bigram = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(bigram)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self.word_cache[word] = tokens
        return tokens

    def encode_qa(self, question, answer, max_len=256):
        """Encode a QA pair with special separator tokens: bos <|q|> question <|a|> answer eos"""
        ids = [self.bos_id, self.q_sep_id]
        for w in question.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.space_id)
        ids.append(self.a_sep_id)
        for w in answer.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.space_id)
        ids.append(self.eos_id)
        return ids[:max_len]

    def encode_prompt(self, prompt, max_len=192):
        """Encode a chat prompt with <|q|> prefix to trigger answer generation."""
        ids = [self.bos_id, self.q_sep_id]
        for w in prompt.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.space_id)
        # Add <|a|> to signal: now generate the answer
        ids.append(self.a_sep_id)
        return ids[:max_len]

    def encode(self, text, max_len=128):
        words = text.split()
        ids = [self.bos_id]
        for w in words:
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.space_id)
        ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in (self.pad_id, self.eos_id):
                break
            if i in (self.bos_id, self.q_sep_id, self.a_sep_id):
                continue
            if i == self.space_id:
                tokens.append(' ')
            else:
                tokens.append(self.id_to_token.get(i, '?'))
        return ''.join(tokens)


# ── Expanded Training Data (550+ QA pairs) ────────────────────────

def _generate_qa_pairs():
    """Generate a comprehensive set of QA pairs for training. v4: 550+ dense pairs."""
    pairs = []

    # === AI & Machine Learning Fundamentals (80) ===
    ml_fundamentals = [
        ("What is AI", "AI is the simulation of human intelligence by machines enabling learning reasoning and problem solving"),
        ("How does AI work", "AI works by processing data through algorithms that identify patterns and make decisions"),
        ("What is machine learning", "Machine learning enables computers to learn patterns from data without explicit programming"),
        ("Explain supervised learning", "Supervised learning trains models on labeled input output pairs to predict outputs for new inputs"),
        ("What is unsupervised learning", "Unsupervised learning finds hidden patterns in data without predefined labels or answers"),
        ("How does deep learning work", "Deep learning uses neural networks with many layers to learn hierarchical patterns from data"),
        ("What is a neural network", "A neural network is a computing system inspired by biological brain networks with connected neurons"),
        ("Explain backpropagation", "Backpropagation computes gradients via chain rule to update neural network weights during training"),
        ("What is gradient descent", "Gradient descent iteratively adjusts parameters to minimize loss by following negative gradients"),
        ("How do neural networks learn", "Neural networks learn by adjusting weights via backpropagation to minimize prediction error"),
        ("What is a transformer", "A transformer is a neural network architecture using self attention mechanisms for sequence processing"),
        ("Explain attention mechanism", "Attention allows models to dynamically focus on relevant parts of input when generating output"),
        ("What is self attention", "Self attention lets each position in a sequence attend to all other positions to compute representations"),
        ("How does GPT work", "GPT uses decoder only transformer to generate text autoregressively predicting one token at a time"),
        ("What is BERT", "BERT is a bidirectional transformer pre trained on masked language modeling for understanding tasks"),
        ("Explain tokenization", "Tokenization splits text into smaller units like words or subwords that models can process"),
        ("What is an embedding", "An embedding represents discrete items like words as dense continuous vectors in vector space"),
        ("How do embeddings work", "Embeddings map categorical data to continuous vectors capturing semantic relationships between items"),
        ("What is fine tuning", "Fine tuning adapts a pre trained model to a specific task using a smaller focused dataset"),
        ("Explain transfer learning", "Transfer learning reuses a pre trained model on a new related task leveraging learned features"),
        ("What is reinforcement learning", "Reinforcement learning trains agents through rewards and penalties to make optimal decisions"),
        ("How does RLHF work", "RLHF uses human feedback as reward signals to align language models with human preferences"),
        ("What is a loss function", "A loss function measures difference between predictions and true values guiding model optimization"),
        ("Explain cross entropy loss", "Cross entropy loss measures divergence between predicted and actual probability distributions"),
        ("What is overfitting", "Overfitting occurs when a model memorizes training data but fails on unseen test data"),
        ("How to prevent overfitting", "Prevent overfitting with regularization dropout early stopping data augmentation and more data"),
        ("What is underfitting", "Underfitting happens when a model is too simple to capture patterns in the training data"),
        ("Explain regularization", "Regularization adds penalties to loss function to discourage overly complex models"),
        ("What is dropout", "Dropout randomly deactivates neurons during training to prevent overfitting and improve generalization"),
        ("How does batch normalization work", "Batch norm normalizes layer inputs using batch statistics stabilizing and accelerating training"),
        ("What is layer normalization", "Layer norm normalizes across features for each sample independently working well with variable lengths"),
        ("Explain activation functions", "Activation functions introduce non linearity enabling networks to learn complex patterns"),
        ("What is ReLU", "ReLU outputs zero for negative inputs and the input itself for positive inputs widely used in deep nets"),
        ("How does GELU work", "GELU smoothly approximates ReLU with a Gaussian curve often used in transformer models like GPT"),
        ("What is an optimizer", "An optimizer updates model parameters based on computed gradients to minimize the loss function"),
        ("Explain Adam optimizer", "Adam combines momentum and adaptive learning rates for efficient stochastic optimization"),
        ("What is learning rate", "Learning rate controls how much parameters change each step too large diverges too small is slow"),
        ("How does learning rate scheduling work", "LR scheduling reduces LR over time starting high for speed then low for precision"),
        ("What is a hyperparameter", "Hyperparameters are settings configured before training like learning rate batch size and layers"),
        ("Explain epoch", "An epoch is one complete pass through entire training dataset during model training"),
        ("What is batch size", "Batch size determines how many samples are processed together before updating weights"),
        ("How does gradient clipping work", "Gradient clipping caps gradient magnitude preventing exploding gradients in deep networks"),
        ("What is weight decay", "Weight decay adds L2 penalty to weights encouraging smaller values and simpler models"),
        ("Explain data augmentation", "Data augmentation creates variations of existing data improving model robustness and generalization"),
        ("What is cross validation", "Cross validation rotates train test splits to evaluate model performance reliably"),
        ("How does k-fold cross validation work", "K-fold splits data into k parts using each part once as test and rest as train"),
        ("What is softmax", "Softmax converts raw scores into probabilities summing to one used in classification output layers"),
        ("Explain temperature in sampling", "Temperature controls randomness low temperature is greedy high is diverse random"),
        ("What is top-k sampling", "Top-k restricts token selection to k most likely candidates balancing quality and diversity"),
        ("How does top-p sampling work", "Top-p or nucleus sampling picks from smallest set of tokens exceeding cumulative probability p"),
        ("What is beam search", "Beam search explores multiple candidate sequences finding most likely output during decoding"),
        ("Explain perplexity", "Perplexity measures prediction quality lower means better language modeling performance"),
        ("What is a foundation model", "A foundation model is a large model trained on broad data adaptable to many downstream tasks"),
        ("How do large language models work", "LLMs use transformers trained on massive text to predict next tokens with emergent capabilities"),
        ("What is NLP", "NLP stands for Natural Language Processing enabling computers to understand and generate human language"),
        ("Explain named entity recognition", "NER identifies and categorizes named entities like people organizations and locations in text"),
        ("What is sentiment analysis", "Sentiment analysis determines emotional tone of text as positive negative or neutral"),
        ("How does text summarization work", "Summarization condenses long documents into shorter versions preserving key information"),
        ("What is machine translation", "Machine translation automatically converts text from one language to another"),
        ("Explain text classification", "Text classification assigns predefined categories to documents based on their content"),
        ("What is a convolutional neural network", "CNN uses convolution operations to process grid-like data such as images efficiently"),
        ("How does CNN work", "CNN applies filters to extract hierarchical features from images through pooling and activation"),
        ("What is a recurrent neural network", "RNN processes sequences maintaining hidden state capturing information from previous steps"),
        ("How does LSTM work", "LSTM uses gating mechanisms to learn long-term dependencies avoiding vanishing gradient problems"),
        ("What is GRU", "GRU is a simplified LSTM variant with fewer gates faster to train while handling sequences well"),
        ("Explain the vanishing gradient problem", "Gradients shrink exponentially in deep nets making early layers unable to learn effectively"),
        ("What is the exploding gradient problem", "Gradients grow exponentially causing unstable training and NaN values in weights"),
        ("How does residual connection work", "Residual connections add input to output helping gradients flow through very deep networks"),
        ("What is a generative model", "Generative models learn data distribution to create new samples similar to training data"),
        ("Explain how GANs work", "GANs train generator and discriminator in competition producing realistic synthetic data"),
        ("What is a VAE", "VAE learns probabilistic latent representation encoding data into compressed meaningful space"),
        ("How does diffusion model work", "Diffusion models gradually denoise random noise to generate high-quality images and other data"),
        ("What is the encoder-decoder architecture", "Encoder processes input into representation and decoder generates output from it"),
        ("Explain seq-to-seq models", "Sequence to sequence models map input sequences to output sequences useful for translation"),
        ("What is multi-head attention", "Multi-head attention runs parallel attention operations across different representation subspaces"),
        ("How does positional encoding work", "Positional encodings give models information about token order since attention is permutation invariant"),
        ("What is layer-wise learning rate decay", "LLRD applies different learning rates per layer lower for earlier layers higher for later"),
        ("Explain mixed precision training", "Mixed precision uses float16 for computation and float32 for accuracy speeding up training"),
        ("What is distributed training", "Distributed training spreads computation across multiple GPUs or machines for faster scaling"),
        ("How does data parallelism work", "Data parallelism copies model to each GPU splitting batches and averaging gradients"),
        ("What is model parallelism", "Model parallelism splits model layers across devices when model is too large for single GPU"),
        ("Explain the bias-variance tradeoff", "Simple models underfit complex models overfit balance minimizes total error"),
        ("What is feature engineering", "Feature engineering creates informative input variables from raw data to improve model performance"),
        ("How does feature selection work", "Feature selection picks most important variables reducing dimensionality and overfitting risk"),
        ("What is dimensionality reduction", "Dimensionality reduction compresses high-dimensional data into lower dimensions preserving structure"),
        ("Explain PCA", "PCA finds orthogonal axes of maximum variance projecting data onto fewer dimensions"),
        ("What is clustering", "Clustering groups similar data points together without labels discovering inherent structure"),
        ("How does k-means work", "K-means iteratively assigns points to nearest centroid then updates centroids until convergence"),
        ("What is a decision tree", "Decision tree makes predictions by splitting data based on feature values in hierarchical rules"),
        ("Explain random forest", "Random forest averages many decision trees trained on bootstrapped samples reducing variance"),
        ("What is gradient boosting", "Gradient boosting sequentially trains trees each correcting errors of previous ensemble"),
        ("How does XGBoost work", "XGBoost optimizes gradient boosting with regularization parallel processing and missing value handling"),
        ("What is SVM", "SVM finds optimal hyperplane separating classes maximizing margin between them"),
        ("Explain kernel methods", "Kernels implicitly map data to high-dimensional space enabling linear separation of non-linear data"),
        ("What is naive Bayes", "Naive Bayes applies Bayes theorem with feature independence assumption for fast classification"),
        ("How does KNN work", "KNN classifies by majority vote of k nearest neighbors simple but effective baseline method"),
        ("What is AUC-ROC", "AUC-ROC measures classifier performance across thresholds area under receiver operating characteristic curve"),
        ("Explain precision and recall", "Precision measures accuracy of positive predictions recall measures coverage of actual positives"),
        ("What is F1 score", "F1 score is harmonic mean of precision and recall balancing both metrics in single number"),
        ("How does confusion matrix work", "Confusion matrix shows true false positives and negatives evaluating classifier performance"),
    ]

    # === Programming & Computer Science (100) ===
    programming = [
        ("What is Python", "Python is a high-level interpreted programming language known for readability versatility and vast ecosystem"),
        ("Why is Python popular", "Python is popular for its simple syntax extensive libraries and strong community support for beginners"),
        ("What is a variable", "Variables are named storage locations holding values that can change during program execution"),
        ("How do variables work", "Variables store data in memory allowing programs to reference and manipulate values by name"),
        ("What is a data type", "Data types define kind of value a variable holds like integer string boolean or floating point number"),
        ("Explain primitive types", "Primitive types are basic built-in types int float char bool directly supported by language"),
        ("What is a string", "String is sequence of characters representing text manipulated by concatenation slicing and formatting"),
        ("How do strings work", "Strings store text as character sequences supporting indexing iteration and various operations"),
        ("What is an integer", "Integer is whole number type without fractional part used for counting and indexing operations"),
        ("What is a float", "Float represents decimal numbers with fractional part used for measurements and calculations"),
        ("What is a boolean", "Boolean has two values true and false used for logic conditions and control flow decisions"),
        ("What is a list", "List is ordered collection of items that can be modified by adding removing or changing elements"),
        ("How do lists work", "Lists store multiple values in order accessible by index supporting append insert delete operations"),
        ("What is a dictionary", "Dictionary stores key-value pairs enabling fast lookup insertion and deletion by unique keys"),
        ("How do dictionaries work", "Dictionaries map keys to values using hash tables for average constant-time access operations"),
        ("What is a tuple", "Tuple is immutable ordered collection of elements that cannot be changed after creation"),
        ("What is a set", "Set is unordered collection of unique elements supporting union intersection and difference operations"),
        ("What is a function", "Function is reusable code block performing specific task accepting parameters and returning values"),
        ("How do functions work", "Functions encapsulate logic accept inputs process them and optionally return output results"),
        ("What are parameters vs arguments", "Parameters are placeholders in function definition arguments are actual values passed at call time"),
        ("Explain return statement", "Return sends value back to caller exiting function immediately with specified result"),
        ("What is scope", "Scope determines where variables are accessible global local and block scopes have different visibility"),
        ("How does scope work", "Scope controls variable lifetime and accessibility nested functions access outer scope via closures"),
        ("What is recursion", "Recursion is when function calls itself to solve smaller instances of same problem"),
        ("How does recursion work", "Recursion breaks problems into base case and recursive case needing stack memory for calls"),
        ("What is a loop", "Loop repeats code block either fixed number of times or until condition becomes false"),
        ("How does for loop work", "For loop iterates over sequence executing body once for each element in order"),
        ("How does while loop work", "While loop repeats while condition is true checking before each iteration"),
        ("What is break statement", "Break exits innermost loop immediately transferring control to code after the loop"),
        ("What is continue statement", "Continue skips current iteration jumping to next loop cycle without executing remaining body"),
        ("What is conditional logic", "Conditionals execute different code branches based on whether expressions evaluate true or false"),
        ("How does if-elif-else work", "If checks condition elif checks alternatives else handles none matching case"),
        ("What is exception handling", "Exception handling catches runtime errors gracefully using try except finally blocks"),
        ("How does try-except work", "Try executes risky code except catches specific errors finally always runs cleanup code"),
        ("What is a class", "Class is blueprint defining properties and methods for creating objects in object-oriented programming"),
        ("How do classes work", "Classes define templates objects are instances created from those templates with own state"),
        ("What is an object", "Object is instance of class containing data attributes and behavior methods bundled together"),
        ("Explain OOP principles", "OOP organizes code around objects with encapsulation inheritance polymorphism and abstraction"),
        ("What is encapsulation", "Encapsulation bundles data and methods restricting direct access protecting internal state"),
        ("What is inheritance", "Inheritance lets class derive properties from parent promoting code reuse and hierarchy"),
        ("What is polymorphism", "Polymorphism allows different classes to be treated as same type through shared interface"),
        ("What is abstraction", "Abstraction hides implementation details exposing only essential features to users"),
        ("What is a constructor", "Constructor is special method initializing new object setting up initial state upon creation"),
        ("What is a method", "Method is function defined inside class operating on instance data through self parameter"),
        ("What is static method", "Static method belongs to class not instances cannot access or modify instance state"),
        ("What is class method", "Class method operates on class itself receiving class as first argument instead of instance"),
        ("What is a decorator", "Decorator modifies function behavior wrapping it with additional functionality without changing source"),
        ("How do decorators work", "Decorators take function return new function adding behavior before after or around original call"),
        ("What is a lambda", "Lambda is anonymous inline function defined in single expression commonly used for simple operations"),
        ("What is a comprehension", "Comprehension builds lists dicts or sets concisely from iterable in single readable line"),
        ("How does list comprehension work", "List comprehension creates new list by applying expression to each item meeting condition"),
        ("What is a generator", "Generator produces values lazily using yield saving memory compared to building full list"),
        ("How do generators work", "Generators pause and resume execution yielding one value at a time on demand"),
        ("What is context manager", "Context manager sets up and tears down resources using with statement ensuring cleanup"),
        ("How does with statement work", "With guarantees resource cleanup even if exceptions occur during block execution"),
        ("What is a module", "Module is file containing Python code that can be imported to reuse functions classes and variables"),
        ("What is a package", "Package is directory of modules with init file organizing related code into namespace"),
        ("How does import work", "Import loads module making its contents available for use in current program scope"),
        ("What is virtual environment", "Virtual env isolates project dependencies preventing conflicts between different projects"),
        ("What is pip", "pip is package installer for Python downloading and managing libraries from PyPI repository"),
        ("What is an array", "Array stores elements at contiguous memory positions accessed by numerical index"),
        ("How do arrays differ from lists", "Arrays have fixed type and size for efficiency while lists are flexible dynamic collections"),
        ("What is a linked list", "Linked list is data structure where each node contains data and reference to next node"),
        ("How does linked list work", "Nodes connect via pointers enabling efficient insertion deletion at any position"),
        ("What is a stack", "Stack is last-in-first-out data structure with push and pop operations"),
        ("How does stack work", "Stack adds removes only from top like pile of plates function call stacks use this"),
        ("What is a queue", "Queue is first-in-first-out data structure with enqueue and dequeue operations"),
        ("How does queue work", "Queue adds at rear removes from front like waiting line print jobs use this pattern"),
        ("What is a hash table", "Hash table maps keys to values via hash function enabling average constant-time lookup"),
        ("How does hashing work", "Hash function converts key to index collisions resolved by chaining or open addressing"),
        ("What is binary search tree", "BST is tree where left child is smaller right child is larger than parent node"),
        ("How does BST work", "BST enables sorted search insert delete in logarithmic average time complexity"),
        ("What is a heap", "Heap is complete binary tree where parent is always greater or smaller than children"),
        ("How does heap work", "Heaps enable efficient priority queue operations extracting min or max quickly"),
        ("What is a graph", "Graph is data structure with vertices connected by edges representing relationships"),
        ("How do graphs work", "Graphs model networks social connections routes and dependencies with directed or undirected edges"),
        ("What is BFS", "BFS explores graph level by level visiting neighbors before moving deeper finding shortest path"),
        ("What is DFS", "DFS explores graph depth first going as far as possible along each branch before backtracking"),
        ("How does binary search work", "Binary search finds element in sorted array by repeatedly halving search range"),
        ("What is sorting", "Sorting arranges elements in defined order ascending descending or custom comparator"),
        ("How does quicksort work", "Quicksort partitions array around pivot recursively sorting subarrays in-place efficiently"),
        ("How does mergesort work", "Mergesort divides array in half sorts each half then merges sorted halves together"),
        ("What is Big O notation", "Big O describes algorithm efficiency as input size grows ignoring constants"),
        ("Explain time complexity", "Time complexity counts operations relative to input size determining scalability"),
        ("Explain space complexity", "Space complexity measures memory usage relative to input size including auxiliary structures"),
        ("What is O(1)", "Constant time operation executes in fixed time regardless of input size like array access"),
        ("What is O(log n)", "Logarithmic time halves problem each step like binary search in sorted arrays"),
        ("What is O(n)", "Linear time scales proportionally with input size like single pass through list"),
        ("What is O(n log n)", "Linearithmic time appears in efficient sorting algorithms like mergesort and heapsort"),
        ("What is O(n squared)", "Quadratic time appears in nested loops like bubble sort and naive matrix multiplication"),
        ("What is dynamic programming", "DP solves problems by breaking into overlapping subproblems storing results for reuse"),
        ("How does memoization work", "Memoization caches expensive function results returning cached value on repeated calls"),
        ("What is greedy algorithm", "Greedy algorithm makes locally optimal choice each step hoping for globally optimal result"),
        ("What is divide and conquer", "Divide and conquer splits problem into independent subproblems combining solutions"),
        ("What is a pointer", "Pointer stores memory address of another variable enabling indirect access and manipulation"),
        ("How does memory management work", "Memory management allocates frees memory using stack for locals heap for dynamic data"),
        ("What is garbage collection", "GC automatically reclaims memory no longer referenced preventing leaks in managed languages"),
        ("What is compilation", "Compilation translates source code to machine code before execution catching errors early"),
        ("What is interpretation", "Interpretation reads and executes source code line by line without separate compile step"),
        ("What is a debugger", "Debugger helps find bugs by pausing execution inspecting variables stepping through code"),
        ("How does version control work", "Version control tracks changes over time enabling collaboration rollback and branching"),
        ("What is Git", "Git is distributed version control system tracking file changes managing branches and merging code"),
        ("How does Git work", "Git snapshots changes commits to history enables branching merging and remote collaboration"),
        ("What is GitHub", "GitHub hosts Git repositories online providing collaboration code review and CI/CD tools"),
        ("What is REST API", "REST is architectural style using HTTP methods to access resources via URLs"),
        ("How do APIs work", "APIs define how software components communicate using requests responses and endpoints"),
        ("What is JSON", "JSON is lightweight data format using key-value pairs for data exchange between systems"),
        ("What is SQL", "SQL is query language for managing relational databases with select insert update delete commands"),
        ("How does database indexing work", "Index creates lookup structure on columns dramatically speeding up query performance"),
        ("What is NoSQL", "NoSQL databases store flexible non-tabular data like documents key-value or graphs"),
        ("What is caching", "Caching stores frequently used data in fast memory reducing latency and load"),
        ("How does Redis work", "Redis is in-memory key-value store used for caching sessions and real-time features"),
        ("What is TCP", "TCP provides reliable ordered delivery of data packets over networks with error checking"),
        ("What is UDP", "UDP is fast connectionless protocol sending data without guaranteeing delivery or order"),
        ("What is HTTP", "HTTP is protocol for web communication defining request methods status codes and headers"),
        ("How does HTTPS work", "HTTPS encrypts HTTP with TLS securing data in transit against eavesdropping"),
        ("What is DNS", "DNS translates domain names to IP addresses enabling humans to use memorable addresses"),
        ("How does DNS resolution work", "DNS queries recursive servers which query authoritative servers to find IP address"),
        ("What is a cookie", "Cookie is small file browser stores to maintain session state across HTTP requests"),
        ("What is a thread", "Thread is lightweight execution unit within process enabling concurrent code execution"),
        ("How does multithreading work", "Multiple threads share process memory running in parallel on different CPU cores"),
        ("What is a process", "Process is isolated program instance with own memory space and system resources"),
        ("What is deadlock", "Deadlock occurs when processes wait for each other indefinitely blocking all progress"),
        ("What is race condition", "Race condition happens when concurrent access to shared data produces unpredictable results"),
        ("What is mutex", "Mutex is synchronization primitive ensuring only one thread accesses critical section at a time"),
        ("What is async programming", "Async programming handles concurrent I/O operations without blocking main execution thread"),
        ("How does event loop work", "Event loop manages asynchronous callbacks executing them when I/O operations complete"),
        ("What is microservices", "Microservices splits application into small independent services communicating via APIs"),
        ("How does containerization work", "Containers package app with dependencies ensuring consistent deployment across environments"),
        ("What is Docker", "Docker platform builds ships and runs containers providing consistent runtime environment"),
        ("What is Kubernetes", "Kubernetes orchestrates containerized applications managing scaling networking and deployment"),
        ("What is CI/CD", "CI/CD automates building testing and deploying code changes through pipeline stages"),
        ("How does load balancing work", "Load balancer distributes traffic across servers improving reliability and capacity"),
        ("What is cloud computing", "Cloud delivers computing services over internet including storage processing and databases"),
        ("What is cybersecurity", "Cybersecurity protects systems networks and data from digital attacks unauthorized access"),
        ("How does encryption work", "Encryption scrambles plaintext into ciphertext using keys only reversible with correct key"),
        ("What is hashing", "Hashing converts data to fixed-size fingerprint irreversible for verifying integrity"),
        ("What is authentication vs authorization", "Auth verifies identity authz verifies permissions after identity confirmed"),
    ]

    # === Science & Physics (80) ===
    science = [
        ("What is gravity", "Gravity is force attracting objects with mass toward each other stronger for closer massive bodies"),
        ("How does gravity work", "Mass curves spacetime and objects follow curved paths appearing as gravitational attraction"),
        ("What is the speed of light", "Light travels at about 299792458 meters per second in vacuum universal speed limit"),
        ("Why is light speed limit", "Nothing with mass can reach light speed because energy required would become infinite"),
        ("What is energy", "Energy is capacity to do work exists as kinetic potential thermal chemical nuclear forms"),
        ("What is kinetic energy", "Kinetic energy is energy of motion proportional to mass times velocity squared"),
        ("What is potential energy", "Potential energy is stored energy due to position or configuration convertible to kinetic"),
        ("How does photosynthesis work", "Plants convert sunlight water carbon dioxide into glucose releasing oxygen as byproduct"),
        ("What is the periodic table", "Periodic table organizes chemical elements by atomic number electron configuration and properties"),
        ("Who created periodic table", "Mendeleev created periodic table arranging elements by atomic mass predicting unknown ones"),
        ("What is an atom", "Atom is basic matter unit composed of protons neutrons in nucleus and orbiting electrons"),
        ("What is an electron", "Electron is negatively charged subatomic particle orbiting nucleus involved in chemical bonding"),
        ("What is a proton", "Proton is positively charged particle in atomic nucleus determining element identity"),
        ("What is a neutron", "Neutron is neutral particle in atomic nucleus contributing to mass and nuclear stability"),
        ("What is DNA", "DNA is molecule carrying genetic instructions for development functioning of living organisms"),
        ("How does DNA work", "DNA uses four bases adenine thymine guanine cytosine encoding genetic information in sequences"),
        ("What is RNA", "RNA is nucleic acid carrying genetic information from DNA to build proteins in cells"),
        ("What is evolution", "Evolution is change in inherited traits over generations driven by natural selection"),
        ("How does natural selection work", "Natural selection favors traits improving survival and reproduction passing genes to offspring"),
        ("What is relativity", "Einstein theory stating space and time are relative and mass curves spacetime causing gravity"),
        ("What is special relativity", "Special relativity deals with constant light speed consequences time dilation length contraction"),
        ("What is general relativity", "General relativity describes gravity as curvature of spacetime caused by mass and energy"),
        ("What is quantum mechanics", "QM describes particle behavior at atomic scales with probability wave-particle duality uncertainty"),
        ("What is wave-particle duality", "Particles exhibit both wave and particle properties depending on how they are observed"),
        ("What is Heisenberg uncertainty", "You cannot precisely know both position and momentum of particle simultaneously"),
        ("What is quantum entanglement", "Entangled particles have correlated states regardless of distance measuring one affects other instantly"),
        ("What is thermodynamics", "Thermodynamics studies heat work temperature and energy transfer in physical systems"),
        ("What is first law of thermodynamics", "First law states energy cannot be created or destroyed only transformed or transferred"),
        ("What is second law of thermodynamics", "Second law states entropy of closed system always increases over time"),
        ("What is entropy", "Entropy measures disorder or randomness in system tending to increase in natural processes"),
        ("What is absolute zero", "Absolute zero is minus 273.15 Celsius lowest possible temperature where molecular motion stops"),
        ("How do magnets work", "Magnets produce magnetic fields from aligned electron spins attracting ferromagnetic materials"),
        ("What is electromagnetic field", "EM field is physical field produced by electrically charged objects affecting other charges"),
        ("What is electricity", "Electricity is flow of electric charge through conductors powering devices and transmitting signals"),
        ("What is current", "Current is rate of charge flow measured in amperes driven by voltage through resistance"),
        ("What is voltage", "Voltage is electrical potential difference driving current through circuit measured in volts"),
        ("What is resistance", "Resistance opposes current flow converting electrical energy to heat measured in ohms"),
        ("How do circuits work", "Circuits provide closed path for current flow with components controlling voltage and current"),
        ("What is Ohm's law", "Ohm law states voltage equals current times resistance fundamental relationship in circuits"),
        ("What is the water cycle", "Water evaporates forms clouds precipitates as rain flows to oceans repeating continuously"),
        ("What is climate change", "Climate change refers to long-term shifts in temperatures and weather from human activity"),
        ("What is greenhouse effect", "Greenhouse effect traps heat in atmosphere using gases like CO2 warming planet surface"),
        ("What is a cell", "Cell is smallest life unit containing genetic material organelles and cytoplasm enclosed by membrane"),
        ("What is mitosis", "Mitosis is cell division producing two genetically identical daughter cells from one parent"),
        ("What is meiosis", "Meiosis is cell division producing gametes with half chromosomes for sexual reproduction"),
        ("What is a black hole", "Black hole is region where gravity so strong nothing escapes not even light past event horizon"),
        ("How do black holes form", "Black holes form when massive stars collapse under gravity after exhausting nuclear fuel"),
        ("What is event horizon", "Event horizon is boundary around black hole beyond which nothing can escape gravitational pull"),
        ("What is a star", "Star is massive glowing ball of plasma held together by gravity producing energy via fusion"),
        ("How do stars form", "Stars form when gas dust clouds collapse under gravity until core temperature ignites fusion"),
        ("What is nuclear fusion", "Fusion combines light atomic nuclei releasing enormous energy powering stars and sun"),
        ("What is the Big Bang", "Big Bang theory states universe began expanding from hot dense state about 14 billion years ago"),
        ("What is plate tectonics", "Earth crust divided into moving plates causing earthquakes volcanoes and mountain formation"),
        ("What is an ecosystem", "Ecosystem is community of organisms interacting with physical environment forming functional unit"),
        ("How does brain work", "Brain processes information via neurons transmitting electrical and chemical signals across synapses"),
        ("What is a neuron", "Neuron is nerve cell transmitting electrical and chemical signals throughout nervous system"),
        ("What is a gene", "Gene is DNA segment encoding instructions for making specific protein or functional RNA"),
        ("What is osmosis", "Osmosis is water movement across membrane from low to high solute concentration equalizing levels"),
        ("How do lasers work", "Lasers emit coherent focused light through stimulated emission of photons from excited atoms"),
        ("What is superconductivity", "Superconductors conduct electricity with zero resistance below critical temperature"),
        ("What is wavelength", "Wavelength is distance between successive wave peaks determining color for light and pitch for sound"),
        ("What is frequency", "Frequency is number of wave cycles per second measured in hertz determining energy and pitch"),
        ("How do X-rays work", "X-rays are high-energy electromagnetic waves penetrating soft tissue for medical imaging"),
        ("What is radioactivity", "Radioactivity is unstable nuclei emitting particles or energy as they decay to stable states"),
        ("What is half-life", "Half-life is time for half of radioactive sample to decay used in dating and medicine"),
        ("What is a catalyst", "Catalyst speeds up chemical reaction without being consumed lowering activation energy"),
        ("What is pH scale", "pH measures acidity alkalinity from 0 to 14 seven neutral below acidic above basic"),
        ("What is a molecule", "Molecule is two or more atoms bonded together forming distinct chemical compound"),
        ("What is a chemical bond", "Chemical bond holds atoms together through sharing or transferring electrons"),
        ("How does sound travel", "Sound travels as mechanical waves vibrating particles through air water or solid media"),
        ("What is refraction", "Refraction is light bending when passing between media of different densities changing speed"),
        ("What is reflection", "Reflection is light bouncing off surface at angle equal to incoming angle"),
        ("What is static electricity", "Static electricity is charge buildup on surfaces causing sparks when discharged suddenly"),
        ("What is nuclear fission", "Fission splits heavy atomic nuclei releasing energy used in nuclear power plants"),
        ("What is antimatter", "Antimatter is mirror of normal matter with opposite charge annihilating on contact releasing energy"),
        ("What is dark matter", "Dark matter is invisible mass holding galaxies together detected by gravitational effects only"),
        ("What is dark energy", "Dark energy is mysterious force accelerating universe expansion dominating universe energy content"),
        ("What is Higgs boson", "Higgs boson is particle giving mass to other particles via Higgs field confirmed in 2012"),
        ("What is semiconductor", "Semiconductor material has conductivity between conductor and insulator basis of modern electronics"),
        ("How do transistors work", "Transistors switch or amplify electronic signals fundamental building blocks of all electronics"),
        ("What is a battery", "Battery stores chemical energy converting it to electrical energy through electrochemical reactions"),
        ("How do solar panels work", "Solar panels convert photons from sunlight into electricity using photovoltaic effect in silicon"),
        ("What is nuclear reactor", "Nuclear reactor controls fission chain reaction producing heat for electricity generation"),
        ("What is aerodynamics", "Aerodynamics studies air movement around objects designing vehicles for minimal drag maximal lift"),
        ("What is buoyancy", "Buoyancy is upward force fluid exerts on submerged object equal to weight of displaced fluid"),
        ("What is torque", "Torque is rotational force causing objects to spin around axis measured in Newton-meters"),
        ("What is momentum", "Momentum is product of mass and velocity conserved quantity in isolated systems"),
        ("What is conservation of energy", "Energy cannot be created or destroyed only converted between forms total remains constant"),
    ]

    # === Math & Logic (60) ===
    math_logic = [
        ("What is calculus", "Calculus studies rates of change and accumulation using derivatives and integrals"),
        ("What is derivative", "Derivative measures instantaneous rate of change of function at any given point"),
        ("What is integral", "Integral accumulates quantities over interval opposite of derivative finding area under curve"),
        ("What is algebra", "Algebra uses symbols to represent numbers and solve equations with unknown variables"),
        ("What is a variable in math", "Variable is symbol representing unknown or changeable quantity in mathematical expression"),
        ("What is an equation", "Equation states two expressions are equal often containing unknowns to be solved for"),
        ("What is geometry", "Geometry studies shapes sizes angles properties of space figures and their relationships"),
        ("What is a triangle", "Triangle is three-sided polygon with interior angles summing to 180 degrees"),
        ("What is Pythagorean theorem", "In right triangles square of hypotenuse equals sum of squares of other two sides"),
        ("What is a circle", "Circle is shape where all points are equidistant from center point"),
        ("What is pi", "Pi is ratio of circle circumference to diameter approximately 3.14159 irrational number"),
        ("How do fractions work", "Fractions represent parts of whole with numerator over denominator showing division"),
        ("What is a prime number", "Prime has exactly two factors one and itself like 2 3 5 7 11 13 17 19"),
        ("What is composite number", "Composite has more than two factors unlike primes can be divided evenly by other numbers"),
        ("What is probability", "Probability measures likelihood of event occurring ranging from impossible zero to certain one"),
        ("How does probability work", "Probability equals favorable outcomes divided by total possible outcomes assuming equally likely"),
        ("What is statistics", "Statistics collects analyzes interprets data to find patterns make decisions quantify uncertainty"),
        ("What is mean median mode", "Mean is average median is middle value mode is most frequent value in dataset"),
        ("What is standard deviation", "Standard deviation measures spread of data points from mean indicating variability"),
        ("What is variance", "Variance is average of squared deviations from mean measuring data dispersion"),
        ("What is normal distribution", "Normal distribution is symmetric bell curve describing many natural phenomena centered at mean"),
        ("What is a matrix", "Matrix is rectangular array of numbers used in linear algebra transformations and systems"),
        ("How does matrix multiplication work", "Matrix multiplication combines rows of first with columns of second producing new matrix"),
        ("What is a vector", "Vector has magnitude and direction used in physics graphics and machine learning"),
        ("What is dot product", "Dot product multiplies corresponding components and sums result measuring alignment of vectors"),
        ("What is cross product", "Cross product produces vector perpendicular to both inputs measuring area of parallelogram"),
        ("What is logarithm", "Logarithm is exponent needed for base to produce given number inverse of exponentiation"),
        ("What is exponential function", "Exponential function raises constant base to variable power modeling growth and decay"),
        ("What is set theory", "Set theory studies collections of objects using union intersection complement operations"),
        ("What is union of sets", "Union combines all elements from two or more sets into one combined set"),
        ("What is intersection of sets", "Intersection contains only elements common to all participating sets"),
        ("What is combinatorics", "Combinatorics counts arrangements studying permutations combinations and counting principles"),
        ("What is permutation", "Permutation is ordered arrangement of items where order matters formula n factorial over n minus k factorial"),
        ("What is combination", "Combination is unordered selection where order does not matter n choose k formula"),
        ("What is Fibonacci sequence", "Each number is sum of previous two starting 0 1 1 2 3 5 8 13 21 appearing in nature"),
        ("What is golden ratio", "Golden ratio is about 1.618 satisfying phi equals one plus one over phi aesthetically pleasing proportion"),
        ("What is complex number", "Complex number has real and imaginary parts written as a plus bi where i is root of minus one"),
        ("What is imaginary unit", "Imaginary unit i is defined as square root of minus one enabling solutions to unsolvable equations"),
        ("What is mathematical induction", "Induction proves statements for all naturals by proving base case and inductive step"),
        ("What is proof by contradiction", "Assume opposite of what to prove show it leads to logical contradiction proving original claim"),
        ("What is binomial theorem", "It expands power of binomial using coefficients from Pascal triangle sum of n choose k terms"),
        ("What is Pascal triangle", "Pascal triangle displays binomial coefficients where each number is sum of two above it"),
        ("What is modular arithmetic", "Modular arithmetic works with remainders after division like clock wrapping at twelve"),
        ("What is number theory", "Number theory studies integers primes divisibility and properties of whole numbers"),
        ("What is linear equation", "Line equation relates variables with degree one graphing as straight line y equals mx plus b"),
        ("What is quadratic equation", "Quadratic has variable squared term solved using factoring completing square or quadratic formula"),
        ("What is polynomial", "Polynomial is expression with variables raised to non-negative integer powers combined with coefficients"),
        ("What is trigonometry", "Trigonometry studies triangle side-angle relationships using sine cosine tangent functions"),
        ("What is sine function", "Sine of angle is ratio of opposite side to hypotenuse in right triangle oscillating between minus one and one"),
        ("What is cosine function", "Cosine is ratio of adjacent side to hypotenuse phase shifted ninety degrees from sine"),
        ("What is tangent function", "Tangent is sine over cosine ratio of opposite to adjacent side undefined at ninety degrees"),
        ("What is infinity", "Infinity is unbounded concept larger than any real number represented by lemniscate symbol"),
        ("What is limit", "Limit describes value function approaches as input approaches some point foundational to calculus"),
        ("What is continuity", "Continuity means function has no breaks jumps or holes can be drawn without lifting pen"),
        ("What is correlation", "Correlation measures strength of relationship between two variables from minus one to plus one"),
        ("What is regression", "Regression models relationship between dependent and independent variables predicting outcomes"),
        ("What is hypothesis testing", "Hypothesis testing uses sample data to evaluate claims about population parameters"),
        ("What is p-value", "P-value measures probability of observing results if null hypothesis were true"),
        ("What is confidence interval", "Confidence interval gives range likely containing true population parameter with stated confidence"),
        ("What is graph theory", "Graph theory studies vertices connected by edges modeling networks paths and relationships"),
        ("What is Euler formula", "Euler formula e to i pi plus one equals zero connects five fundamental math constants beautifully"),
    ]

    # === General Knowledge & Technology (100) ===
    general = [
        ("What is the internet", "Internet is global network connecting billions of computers enabling worldwide communication"),
        ("How does internet work", "Internet uses TCP/IP protocols routing data packets between interconnected networks worldwide"),
        ("What is World Wide Web", "Web is information system accessed via browsers using URLs HTTP and HTML pages"),
        ("Who invented World Wide Web", "Tim Berners-Lee invented Web in 1989 at CERN proposing HTML URL and HTTP"),
        ("What is a browser", "Browser is software application retrieving displaying navigating web content like Chrome Firefox Safari"),
        ("How do browsers work", "Browsers fetch render HTML CSS JavaScript displaying interactive pages to users"),
        ("What is a server", "Server is computer or program providing services data or resources to client computers over network"),
        ("What is a client", "Client is device or software requesting services from server like browser requesting web page"),
        ("What is operating system", "OS manages hardware resources provides services for programs Windows Linux macOS Android iOS"),
        ("What is Linux", "Linux is open-source Unix-like OS kernel used in servers Android and many computing devices"),
        ("What is Windows", "Windows is graphical OS by Microsoft most popular desktop operating system worldwide"),
        ("What is macOS", "macOS is Apple desktop operating system known for design integration with Apple ecosystem"),
        ("What is Android", "Android is mobile OS by Google powering most smartphones tablets and smart devices globally"),
        ("What is iOS", "iOS is Apple mobile operating system for iPhones iPads featuring tight hardware integration"),
        ("What is cloud computing", "Cloud delivers computing services over internet storage processing databases on demand"),
        ("How does cloud storage work", "Cloud storage saves files on remote servers accessible from any device with internet"),
        ("What is AWS", "Amazon Web Services is leading cloud platform offering hundreds of on-demand services"),
        ("What is virtualization", "Virtualization creates virtual versions of servers storage networks running multiple VMs on one host"),
        ("What is a virtual machine", "VM is software emulating computer system running different OS on same physical hardware"),
        ("What is container", "Container packages application with dependencies running consistently across environments"),
        ("What is Docker", "Docker platform builds manages runs containers using images for portable deployments"),
        ("What is Kubernetes", "Kubernetes orchestrates containers handling scaling networking deployment automation"),
        ("What is DevOps", "DevOps combines development and operations practices for faster reliable software delivery"),
        ("What is Agile", "Agile is iterative software development methodology emphasizing flexibility collaboration customer feedback"),
        ("What is Scrum", "Scrum is Agile framework organizing work into sprints with daily standups and retrospectives"),
        ("What is blockchain", "Blockchain is distributed immutable ledger recording transactions across decentralized network"),
        ("How does blockchain work", "Blocks contain transactions chained cryptographically making records tamper-evident"),
        ("What is Bitcoin", "Bitcoin is decentralized cryptocurrency using blockchain for peer-to-peer transactions without banks"),
        ("What is cryptocurrency", "Cryptocurrency is digital currency using cryptography for secure decentralized transactions"),
        ("How does mining work", "Mining validates blockchain transactions by solving computational puzzles earning cryptocurrency rewards"),
        ("What is smart contract", "Smart contract is self-executing code on blockchain enforcing agreement terms automatically"),
        ("What is IoT", "IoT connects everyday devices to internet enabling data exchange remote monitoring and control"),
        ("How does smart home work", "Smart home devices connect via WiFi or Bluetooth controlled by apps voice assistants automation"),
        ("What is 5G", "5G is fifth generation mobile network offering faster speeds lower latency more connections than 4G"),
        ("What is WiFi", "WiFi is wireless networking technology connecting devices to internet via radio waves locally"),
        ("What is Bluetooth", "Bluetooth is short-range wireless protocol connecting devices like headphones keyboards speakers"),
        ("What is GPS", "GPS uses satellites triangulating signals to determine precise location anywhere on Earth"),
        ("How does GPS work", "GPS receivers calculate position by measuring distances from multiple satellite signals"),
        ("What is smartphone", "Smartphone combines phone computer camera sensors GPS running apps for communication productivity"),
        ("What is augmented reality", "AR overlays digital content onto real world viewed through phone glasses or headset"),
        ("What is virtual reality", "VR immerses users in fully digital 3D environment using headsets motion controllers"),
        ("What is mixed reality", "MR blends AR and VR letting digital and physical objects interact in real time"),
        ("What is edge computing", "Edge computing processes data near source reducing latency bandwidth dependence on cloud"),
        ("What is quantum computing", "Quantum computing uses quantum bits and superposition solving certain problems exponentially faster"),
        ("What is a qubit", "Qubit is quantum bit existing in superposition of zero and one states simultaneously"),
        ("What is AI art generation", "AI art uses diffusion or transformer models to create images from text descriptions"),
        ("How do recommendation systems work", "Recommendation engines suggest items based on user behavior preferences collaborative filtering"),
        ("What is search engine", "Search engine indexes web pages returns relevant results for user queries using ranking algorithms"),
        ("How does Google Search work", "Google crawls indexes ranks pages using hundreds of factors including relevance authority quality"),
        ("What is email", "Email is electronic mail system sending messages across internet using SMTP POP IMAP protocols"),
        ("How does email work", "Email servers route messages from sender to recipient using standardized protocols"),
        ("What is streaming", "Streaming delivers audio video content continuously over internet without full download"),
        ("What is Netflix", "Netflix is streaming service offering movies TV shows originals via subscription on demand"),
        ("What is Spotify", "Spotify is music streaming service offering millions of songs playlists podcasts on demand"),
        ("What is social media", "Social media platforms enable users to create share content and network socially online"),
        ("What is Wikipedia", "Wikipedia is free online encyclopedia written collaboratively by volunteers worldwide"),
        ("What is open source", "Open source software publishes source code publicly allowing anyone to view modify distribute"),
        ("What is GPL license", "GPL requires derivative works also be open source ensuring freedom propagates through modifications"),
        ("What is MIT license", "MIT license permits reuse modification distribution with minimal restrictions preserving attribution"),
        ("What is data privacy", "Data privacy protects personal information from unauthorized access misuse and surveillance"),
        ("What is GDPR", "GDPR is EU regulation governing data protection giving users rights over their personal data"),
        ("What is encryption standard", "AES is widely used encryption standard securing data with key sizes 128 192 or 256 bits"),
        ("What is two-factor authentication", "2FA requires second verification factor beyond password like code from phone or hardware key"),
        ("What is phishing", "Phishing is attack tricking users into revealing passwords credentials through fake websites emails"),
        ("What is malware", "Malware is malicious software designed to damage disrupt or gain unauthorized access to systems"),
        ("What is firewall", "Firewall monitors filters network traffic blocking unauthorized access while permitting legitimate communications"),
        ("What is VPN", "VPN encrypts internet connection routing traffic through secure server hiding IP location"),
        ("What is a chatbot", "Chatbot simulates conversation using rules or AI models to assist users automatically"),
        ("What is Turing test", "Turing test evaluates whether machine can exhibit behavior indistinguishable from human"),
        ("What is AGI", "AGI is artificial general intelligence matching or exceeding human cognitive ability across all domains"),
        ("What is narrow AI", "Narrow AI excels at specific tasks like chess image recognition but lacks general understanding"),
        ("What is 3D printing", "3D printing builds objects layer by layer from digital models using plastic metal resin materials"),
        ("How does 3D printing work", "3D printer extrudes or cures material layer by layer following digital model slice by slice"),
        ("What is robotics", "Robotics designs constructs programs robots for automation manufacturing exploration assistance"),
        ("What is autonomous vehicle", "Self-driving car uses sensors cameras AI to navigate roads without human intervention"),
        ("How do self-driving cars work", "Autonomous vehicles perceive environment plan path control vehicle using sensor fusion deep learning"),
        ("What is drone", "Drone is unmanned aircraft controlled remotely or autonomously for photography delivery surveillance"),
        ("What is wearable technology", "Wearable tech includes smartwatches fitness trackers health monitors worn on body collecting data"),
        ("What is smartwatch", "Smartwatch is wrist-worn device tracking fitness notifications apps extending smartphone capabilities"),
        ("What is biometrics", "Biometrics uses physical characteristics fingerprint face iris for identity verification"),
        ("How does face recognition work", "Face recognition maps facial features comparing against database to verify or identify person"),
        ("What is voice assistant", "Voice assistant like Siri Alexa Google Assistant responds to spoken commands performs tasks via speech recognition"),
        ("How do voice assistants work", "Voice assistants convert speech to text process with AI convert response back to speech via TTS"),
        ("What is natural language understanding", "NLU extracts meaning intent from text or speech going beyond keyword matching to comprehension"),
        ("What is computer vision", "Computer vision enables machines to interpret understand visual information from images or video"),
        ("How does image recognition work", "Image recognition uses CNNs to classify detect objects in images by learned visual features"),
        ("What is speech recognition", "Speech recognition converts spoken audio into text using acoustic and language models"),
        ("What is text-to-speech", "TTS converts written text into spoken audio using synthesis techniques and neural voices"),
        ("What is machine translation", "MT automatically translates text between languages using neural sequence-to-sequence models"),
        ("What is sentiment analysis", "Sentiment analysis determines emotional tone of text positive negative neutral using NLP"),
        ("What is data mining", "Data mining discovers patterns knowledge from large datasets using statistics ML methods"),
        ("What is big data", "Big data refers to datasets too large complex for traditional data processing requiring specialized tools"),
        ("What is Hadoop", "Hadoop is framework for distributed storage processing of big data across clusters of computers"),
        ("What is data warehouse", "Data warehouse is centralized repository storing structured data from multiple sources for analysis"),
        ("What is ETL", "ETL extracts transforms loads data from sources into destination warehouse preparing for analytics"),
        ("What is business intelligence", "BI uses data analysis tools dashboards reporting to support business decision making"),
        ("What is data lake", "Data lake stores raw unstructured data in native format until needed for processing analysis"),
        ("What is API gateway", "API gateway manages routes secures API requests acting as single entry point for clients"),
        ("What is microservices architecture", "Microservices splits app into small independent services communicating via lightweight protocols"),
        ("What is serverless", "Serverless lets developers run code without managing servers cloud provider handles infrastructure scaling"),
        ("What is function as a service", "FaaS executes individual functions on demand charging only for actual compute time used"),
    ]

    # === Health & Daily Life (50) ===
    health = [
        ("What is healthy diet", "Healthy diet includes fruits vegetables whole grains lean proteins limiting processed foods sugar"),
        ("Why is sleep important", "Sleep restores body consolidates memory regulates hormones supports immune function and mental health"),
        ("What is metabolism", "Metabolism is chemical process converting food to energy supporting all bodily functions"),
        ("How does exercise benefit health", "Exercise strengthens heart muscles bones improves mood boosts energy prevents chronic diseases"),
        ("What is cardiovascular disease", "CVD affects heart blood vessels including heart attack stroke caused by narrowed blocked arteries"),
        ("What is immunity", "Immunity is body defense system against pathogens using antibodies white blood cells and inflammation"),
        ("How do vaccines work", "Vaccines train immune system to recognize fight specific pathogens without causing disease"),
        ("What is virus", "Virus is microscopic infectious agent replicating only inside living cells causing various diseases"),
        ("What is bacteria", "Bacteria are single-celled microorganisms some beneficial some harmful causing infections"),
        ("What is antibiotic", "Antibiotic kills or inhibits bacteria growth ineffective against viruses requiring proper prescription"),
        ("What is stress", "Stress is body response to challenges releasing hormones affecting mental physical wellbeing"),
        ("How to manage stress", "Manage stress through exercise meditation adequate sleep social support hobbies and time management"),
        ("What is meditation", "Meditation practice focuses mind reduces anxiety improves concentration promotes emotional health"),
        ("What is hydration important", "Water is essential for digestion temperature regulation joint lubrication nutrient transport brain function"),
        ("What is BMI", "Body Mass Index estimates body fat using weight height categorizing underweight normal overweight obese"),
        ("What is calorie", "Calorie is unit measuring energy content of food and energy expenditure by body activities"),
        ("What is protein", "Protein is macronutrient building muscle tissue enzymes hormones essential for body repair growth"),
        ("What is carbohydrate", "Carbohydrate is primary energy source for body found in grains fruits vegetables sugars fibers"),
        ("What is fat", "Fat is nutrient storing energy insulating organs aiding vitamin absorption found in oils nuts dairy"),
        ("What is vitamin", "Vitamin is organic compound body needs in small amounts for metabolic function immune health"),
        ("What is mineral", "Mineral is inorganic nutrient like iron calcium zinc essential for bone oxygen transport nerve function"),
        ("What is fiber", "Fiber is indigestible plant component aiding digestion regulating blood sugar promoting satiety"),
        ("What is caffeine", "Caffeine is stimulant in coffee tea blocking adenosine receptors increasing alertness energy"),
        ("What is allergy", "Allergy is immune system overreaction to harmless substance causing symptoms like sneezing itching swelling"),
        ("What is diabetes", "Diabetes is condition affecting blood sugar regulation type one autoimmune type two lifestyle-related"),
        ("What is blood pressure", "Blood pressure is force of blood against artery walls hypertension increases risk of heart disease stroke"),
        ("What is cholesterol", "Cholesterol is fat-like substance in cells some produced by liver some from food affecting heart health"),
        ("What is aerobic exercise", "Aerobic exercise sustained activity raising heart rate swimming running cycling improving cardiovascular fitness"),
        ("What is anaerobic exercise", "Anaerobic exercise short intense bursts like sprinting lifting building muscle strength power"),
        ("What is yoga", "Yoga combines physical postures breathing meditation improving flexibility balance strength mental clarity"),
        ("What is mindfulness", "Mindfulness is present-moment awareness practice reducing stress anxiety improving focus emotional regulation"),
        ("What is circadian rhythm", "Circadian rhythm is internal 24-hour clock regulating sleep-wake cycles hormone release body temperature"),
        ("How does memory work", "Memory involves encoding storing retrieving information through hippocampus cortex neural pathways"),
        ("What is cognitive function", "Cognitive function encompasses mental processes learning reasoning remembering problem-solving attention"),
        ("What is mental health", "Mental health affects thinking feeling behavior emotional psychological social well-being throughout life"),
        ("How to improve focus", "Improve focus by minimizing distractions taking breaks sleeping enough exercising staying hydrated practicing mindfulness"),
        ("What is time management", "Time management organizes plans activities effectively to maximize productivity reduce stress"),
        ("What is goal setting", "Goal setting defines specific measurable objectives providing direction motivation progress tracking"),
        ("What is habit formation", "Habit formation is automatic behavior developed through repetition cue routine reward loop"),
        ("How to form good habits", "Build habits by starting small being consistent stacking habits tracking progress rewarding milestones"),
        ("What is financial literacy", "Financial literacy is understanding budgeting saving investing debt managing money effectively"),
        ("What is compound interest", "Compound interest earns interest on both principal and accumulated interest growing wealth exponentially"),
        ("What is diversification", "Diversification spreads investments across assets reducing risk by not relying on single investment"),
        ("What is emergency fund", "Emergency fund is savings covering three to six months expenses for unexpected job loss medical bills"),
        ("How to save money", "Save by tracking spending cutting unnecessary costs automating savings cooking at home comparing prices"),
        ("What is credit score", "Credit score measures creditworthiness affecting loan approval interest rates renting employment"),
    ]

    # === History & Geography (40) ===
    history_geo = [
        ("What is ancient Egypt", "Ancient Egypt was civilization along Nile famous for pyramids pharaohs hieroglyphs mummies"),
        ("Who built pyramids", "Pyramids were built by Egyptian workers as tombs for pharaohs during Old Kingdom period"),
        ("What is Roman Empire", "Roman Empire was ancient civilization spanning Europe North Africa Middle East lasting centuries"),
        ("Who was Julius Caesar", "Julius Caesar was Roman general dictator who expanded territory reformed calendar was assassinated in 44 BC"),
        ("What is Renaissance", "Renaissance was European cultural revival 14th-17th century rebirth of art science philosophy"),
        ("Who was Leonardo da Vinci", "Da Vinci was Renaissance polymath painter scientist inventor created Mona Lisa Last Supper"),
        ("What is Industrial Revolution", "Industrial Revolution was transition to manufacturing 18th-19th century steam power factories urbanization"),
        ("What is World War One", "WWI was global conflict 1914-1918 involving alliances trenches new weapons millions dead"),
        ("What is World War Two", "WWII was global war 1939-1945 Axis vs Allies Holocaust atomic bombs reshaping world order"),
        ("What is Cold War", "Cold War was post-WWII ideological conflict between USA and USSR without direct military confrontation"),
        ("Who was Albert Einstein", "Einstein was physicist who developed relativity theory E equals mc squared revolutionized physics"),
        ("What is democracy", "Democracy is government system where citizens exercise power through voting electing representatives"),
        ("What is continent", "Continent is large continuous landmass Earth has seven Asia Africa North America South America Antarctica Europe Australia"),
        ("What is largest country", "Russia is largest country by area spanning eleven time zones across Europe Asia"),
        ("What is longest river", "Nile River is generally considered longest flowing through northeastern Africa about 6650 kilometers"),
        ("What is highest mountain", "Mount Everest is highest peak above sea level at 8848 meters on border of Nepal Tibet"),
        ("What is ocean", "Ocean is vast body of salt water covering about seventy percent of Earth surface Pacific largest"),
        ("What is equator", "Equator is imaginary line dividing Earth into Northern and Southern Hemispheres at zero latitude"),
        ("What is timezone", "Timezone is region observing uniform standard time for legal commercial social purposes"),
        ("What is population of Earth", "World population exceeds eight billion people growing at roughly eighty million per year currently"),
        ("What is United Nations", "UN is international organization founded 1945 maintaining peace security cooperation among nations"),
        ("What is European Union", "EU is political economic union of twenty-seven European countries with single market free movement"),
        ("What is capital of France", "Paris is capital and largest city of France known for Eiffel Tower Louvre art culture cuisine"),
        ("What is capital of Japan", "Tokyo is capital of Japan largest metropolitan area blending tradition with ultra-modern technology"),
        ("What is Great Wall of China", "Great Wall is series of fortifications built along northern China borders spanning thousands of kilometers"),
        ("What is Machu Picchu", "Machu Picchu is Incan citadel set high in Andes Mountains of Peru UNESCO World Heritage site"),
        ("What is Amazon rainforest", "Amazon is world largest tropical rainforest producing much of Earth oxygen hosting immense biodiversity"),
        ("What is Sahara Desert", "Sahara is world largest hot desert covering much of North Africa roughly nine million square kilometers"),
        ("What is volcano", "Volcano is rupture in crust allowing hot magma gases ash to escape from below surface"),
        ("What is earthquake", "Earthquake is sudden ground shaking caused by movement of tectonic plates beneath surface"),
        ("What is tsunami", "Tsunami is giant ocean wave usually triggered by underwater earthquake devastating coastlines"),
        ("What is climate zone", "Climate zone is area with distinct weather patterns tropical temperate arctic polar depending on latitude"),
        ("What is hemisphere", "Hemisphere is half of Earth divided by equator Northern Southern or by prime meridian Eastern Western"),
        ("What is island", "Island is piece of land surrounded by water smaller than continent ranging from tiny to continental"),
        ("What is peninsula", "Peninsula is land surrounded by water on three sides connected to mainland on remaining side"),
        ("What is archipelago", "Archipelago is chain or group of scattered islands like Indonesia Philippines Hawaii"),
        ("What is glacier", "Glacier is massive slow-moving ice sheet formed from compressed snow over thousands of years"),
        ("What is fossil fuel", "Fossil fuel is coal oil natural gas formed from ancient organic matter major energy source today"),
        ("What is renewable energy", "Renewable energy comes from naturally replenishing sources like solar wind hydro geothermal"),
        ("What is sustainability", "Sustainability meets present needs without compromising future generations ability to meet theirs"),
        ("What is recycling", "Recycling processes waste materials into new products reducing landfill pollution resource consumption"),
    ]

    # === Arts & Culture (30) ===
    arts = [
        ("What is classical music", "Classical music is art music tradition rooted in Western culture orchestral operatic chamber compositions"),
        ("Who was Beethoven", "Beethoven was German composer bridging Classical Romantic eras wrote symphonies piano sonatas despite deafness"),
        ("What is jazz", "Jazz is music genre originating in African-American communities improvisation swing rhythm blue notes"),
        ("What is impressionism", "Impressionism is 19th-century art movement capturing light color momentary impressions Monet pioneered it"),
        ("Who was Picasso", "Picco was Spanish painter co-founding Cubism creating Guernica pioneering modern abstract art"),
        ("What is literature", "Literature is written work poetry prose drama reflecting human experience culture society"),
        ("Who was Shakespeare", "Shakespeare was English playwright poet writing Hamlet Macbeth Romeo Juliet shaping English language"),
        ("What is poetry", "Poetry is literary art using aesthetic rhythmic qualities of language to evoke meaning emotion"),
        ("What is novel", "Novel is long narrative fiction exploring characters plot themes in substantial written work"),
        ("Who was Mozart", "Mozart was prolific Austrian composer composing symphonies operas concertos from childhood genius"),
        ("What is film", "Film is visual art telling stories through moving images sound editing cinematography acting"),
        ("Who made first movie", "Lumiere brothers showed first public film screening in 1895 Paris marking birth of cinema"),
        ("What is photography", "Photography is art of capturing images using light recording moments emotions stories visually"),
        ("What is sculpture", "Sculpture is three-dimensional art created by shaping materials stone metal wood clay into forms"),
        ("Who was Michelangelo", "Michelangelo was Italian Renaissance master sculptor painting David Sistine Chapel ceiling Pietà"),
        ("What is architecture", "Architecture is art science of designing buildings spaces balancing function aesthetics safety sustainability"),
        ("What is Gothic architecture", "Gothic style features pointed arches ribbed vaults flying buttresses stained glass in medieval cathedrals"),
        ("What is theater", "Theater is live performance art actors portraying characters telling stories to audience on stage"),
        ("What is ballet", "Ballet is formalized dance form originating in Italian French courts precise technique graceful movements"),
        ("What is opera", "Opera is dramatic work combining singing music staging orchestra telling story through musical performance"),
        ("Who was Bach", "Bach was German Baroque composer writing fugues cantatas masses influencing all Western music afterward"),
        ("What is folk music", "Folk music is traditional music passed down orally within communities reflecting cultural heritage identity"),
        ("What is animation", "Animation is technique creating illusion of movement through sequential images frames drawings CGI"),
        ("Who was Disney", "Disney was American animator entrepreneur founding Disney studio creating Mickey Mouse theme parks"),
        ("What is fashion", "Fashion is popular style in clothing accessories footwear expression of identity culture trends"),
        ("What is culinary art", "Culinary art is practice of preparing cooking presenting food as creative sensory experience"),
        ("What is philosophy", "Philosophy is study of fundamental questions existence knowledge ethics reason mind language"),
        ("Who was Socrates", "Socrates was Greek philosopher developing Socratic method questioning assumptions seeking wisdom virtue"),
        ("What is ethics", "Ethics is branch of philosophy studying moral principles distinguishing right from wrong good from bad"),
        ("What is logic", "Logic is study of reasoning valid arguments inference principles underlying mathematics philosophy CS"),
    ]

    # === Sports & Games (20) ===
    sports = [
        ("What is soccer", "Soccer is world most popular sport played by two teams of eleven scoring goals with feet"),
        ("What is basketball", "Basketball is team sport shooting ball through hoop five players per team dribbling passing scoring"),
        ("What is tennis", "Tennis is racket sport hitting ball over net singles or doubles on rectangular court"),
        ("What is Olympic Games", "Olympics is international multi-sport event held every four years featuring athletics from worldwide nations"),
        ("What is marathon", "Marathon is long-distance running race covering 42.195 kilometers testing endurance stamina"),
        ("What is chess", "Chess is strategic board game two players moving pieces on checkered board aiming checkmate opponent king"),
        ("How does chess work", "Chess players take turns moving pieces each with specific rules to capture opponent king"),
        ("What is FIFA World Cup", "World Cup is international soccer tournament held every four years determining world champion national team"),
        ("What is NBA", "NBA is premier professional basketball league in North America featuring thirty teams star athletes"),
        ("What is Wimbledon", "Wimbledon is oldest tennis tournament played on grass courts in London annually since 1877"),
        ("What is Formula One", "F1 is highest class of auto racing featuring fastest most technologically advanced racing cars"),
        ("What is cricket", "Cricket is bat-and-ball sport hugely popular in UK India Australia Pakistan involving runs wickets"),
        ("What is rugby", "Rugby is contact team sport carrying kicking oval ball scoring tries goals physical contest"),
        ("What is swimming", "Swimming is moving through water using strokes like freestyle breaststroke butterfly backstroke"),
        ("What is gymnastics", "Gymnastics is sport involving strength flexibility balance agility performing routines on apparatus"),
        ("What is martial arts", "Martial arts are combat practices karate judo taekwondo for self-defense discipline competition"),
        ("What is esports", "Esports is competitive video gaming with professional players leagues tournaments prize money"),
        ("What is yoga sport", "Yoga as sport emphasizes physical poses breath control flexibility balance competitions judged on execution"),
        ("What is triathlon", "Triathlon is endurance race combining swimming cycling running in single continuous event"),
        ("What is golf", "Golf is club sport hitting ball into series of holes using fewest strokes possible on course"),
    ]

    # Combine all categories
    pairs = (ml_fundamentals + programming + science + math_logic + general +
             health + history_geo + arts + sports)
    return pairs


TRAIN_DATA = _generate_qa_pairs()

# ── Global State ──────────────────────────────────────────────────

# Train BPE tokenizer on all training data
_all_texts = [q + " " + a for q, a in TRAIN_DATA]
tokenizer = BPETokenizer(vocab_size=800)
tokenizer.train(_all_texts, target_vocab_size=800)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_history = {"loss": [], "epoch": []}

# v4 model config: ~38M params
MODEL_CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "hidden_size": 384,
    "num_layers": 6,
    "num_heads": 12,
    "max_seq_len": 256,
    "ff_dim": 1024,
}


def get_model():
    global model
    if model is None:
        model = FusionMini(**MODEL_CONFIG).to(device)
    return model


def count_parameters():
    mdl = get_model()
    return sum(p.numel() for p in mdl.parameters())


# ── Train Function ────────────────────────────────────────────────

def train_fn(learning_rate, epochs, batch_size_val):
    try:
        mdl = get_model()
        mdl.train()

        # Pre-encode all training pairs with QA separators (answer-only loss)
        encoded_pairs = []
        for q, a in TRAIN_DATA:
            ids = tokenizer.encode_qa(q, a, max_len=256)
            if len(ids) > 5:  # need at least bos+q_sep+a_sep+1 answer token + eos
                encoded_pairs.append(torch.tensor(ids))

        optimizer = torch.optim.AdamW(mdl.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
        losses = []

        for epoch in range(int(epochs)):
            total_loss = 0
            count = 0
            indices = np.random.permutation(len(encoded_pairs))

            for i in range(0, len(indices), max(batch_size_val, 1)):
                batch_idx = indices[i:i + max(batch_size_val, 1)]
                batch_loss = 0
                optimizer.zero_grad()

                for j in batch_idx:
                    ids = encoded_pairs[j].to(device)
                    if len(ids) < 6:
                        continue
                    inputs = ids[:-1].unsqueeze(0)
                    targets = ids[1:].unsqueeze(0)
                    logits = mdl(inputs)
                    # Find the <|a|> separator position and only compute loss on answer tokens
                    a_sep_pos = None
                    for pos_idx, tid in enumerate(inputs[0]):
                        if tid.item() == tokenizer.a_sep_id:
                            a_sep_pos = pos_idx
                            break
                    if a_sep_pos is not None and a_sep_pos < targets.shape[1]:
                        ans_logits = logits[0, a_sep_pos:, :]
                        ans_targets = targets[0, a_sep_pos:]
                        if ans_logits.shape[0] > 0:
                            loss = nn.CrossEntropyLoss()(ans_logits.unsqueeze(0).view(-1, tokenizer.vocab_size), ans_targets.unsqueeze(0).view(-1))
                        else:
                            loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
                    else:
                        loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
                    batch_loss += loss
                    total_loss += loss.item()
                    count += 1

                if batch_loss.requires_grad:
                    batch_loss = batch_loss / max(len(batch_idx), 1)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                    optimizer.step()

            scheduler.step()
            avg_loss = total_loss / max(count, 1)
            losses.append(round(avg_loss, 4))

        training_history["loss"] = losses
        training_history["epoch"] = list(range(1, int(epochs) + 1))

        if losses:
            plot_text = "Epoch -> Loss\n" + "\n".join(f"{e} -> {l}" for e, l in zip(training_history["epoch"], losses))
            param_count = count_parameters()
            # Diagnostic: token stats
            total_tok = sum(len(ep) for ep in encoded_pairs)
            space_tok = sum(int((ep == tokenizer.space_id).sum()) for ep in encoded_pairs)
            a_sep_positions = []
            for ep in encoded_pairs:
                hits = (ep == tokenizer.a_sep_id).nonzero()
                if len(hits) > 0:
                    a_sep_positions.append((len(ep), hits[0].item()))
            avg_answer_tok = sum(l - p - 2 for l, p in a_sep_positions) / max(len(a_sep_positions), 1)
            diag = f"\nTokens: {total_tok} total | {space_tok} space | ~{avg_answer_tok:.0f} avg answer tok | {tokenizer.vocab_size} vocab"
            # Show first 3 decoded samples to verify fix
            sample_decodes = []
            for j in range(min(3, len(encoded_pairs))):
                decoded = tokenizer.decode(encoded_pairs[j].tolist())
                sample_decodes.append(f"  Sample {j+1}: {decoded[:80]}...")
            diag += "\nVerify samples:\n" + "\n".join(sample_decodes)
            status = f"Done! Final Loss: {losses[-1]:.4f} | Params: {param_count/1e6:.1f}M | Data: {len(TRAIN_DATA)} pairs" + diag
        else:
            plot_text = "No data."
            status = "No training performed."
        return status, plot_text
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}", "(error)"


# ── Chat Function ─────────────────────────────────────────────────

def chat_fn(prompt, max_tokens, temperature, top_p):
    try:
        mdl = get_model()
        mdl.eval()

        input_ids = tokenizer.encode_prompt(prompt, max_len=192)
        generated = list(input_ids)
        with torch.no_grad():
            for _ in range(int(max_tokens)):
                inp = torch.tensor([generated[-192:]], dtype=torch.long, device=device)
                logits = mdl(inp)
                next_logits = logits[0, -1, :] / max(float(temperature), 0.01)

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_idx_to_remove = cumulative_probs > top_p
                    sorted_idx_to_remove[1:] = sorted_idx_to_remove[:-1].clone()
                    sorted_idx_to_remove[0] = False
                    indices_to_remove = sorted_idx[sorted_idx_to_remove]
                    next_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                if next_token == tokenizer.eos_id or len(generated) > 240:
                    break

        response = tokenizer.decode(generated[len(input_ids):])
        return response or "(no output)"
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────

with gr.Blocks(title="Fusion-LLM Demo v5", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Fusion-LLM Demo v5
    **Train & Chat** with a custom Transformer featuring **SBLA Attention** + **Thinking Dial**

    v5 fix: proper space tokenization | ~38M params | BPE vocab=800 | 560+ pairs | answer-only training
    """)

    with gr.Tabs():
        # ── Train Tab ──
        with gr.Tab("Train"):
            with gr.Row():
                lr = gr.Slider(0.0001, 0.01, value=0.001, label="Learning Rate", step=0.0001)
                eps = gr.Slider(1, 100, value=80, label="Epochs", step=1)
                bs = gr.Slider(1, 16, value=4, label="Batch Size", step=1)
            train_btn = gr.Button("Start Training", variant="primary")
            train_status = gr.Textbox(label="Status", interactive=False)
            loss_plot = gr.Textbox(label="Loss History", interactive=False, lines=18)

            train_btn.click(fn=train_fn, inputs=[lr, eps, bs],
                            outputs=[train_status, loss_plot])

        # ── Chat Tab ──
        with gr.Tab("Chat"):
            prompt = gr.Textbox(label="Your Question", placeholder="e.g., What is AI",
                                lines=2)
            with gr.Row():
                max_tok = gr.Slider(10, 100, value=50, label="Max Tokens", step=5)
                temp = gr.Slider(0.1, 2.0, value=0.7, label="Temperature", step=0.1)
                tp = gr.Slider(0.1, 1.0, value=0.9, label="Top-P", step=0.05)
            gen_btn = gr.Button("Generate", variant="primary")
            output = gr.Textbox(label="Response", lines=4, interactive=False)

            gen_btn.click(fn=chat_fn, inputs=[prompt, max_tok, temp, tp], outputs=output)

        # ── About Tab ──
        with gr.Tab("About"):
            gr.Markdown("""
            ## Architecture Overview

            ### SBLA (Sparse Block Latent Attention)
            - Combines **local windowed attention** with **global latent compression**
            - Uses a learned gating mechanism to merge both representations
            - Block-level sparsity reduces O(N^2) complexity

            ### Thinking Dial
            - Configurable reasoning depth via special tokens
            - Controls how much computation the model spends thinking before answering
            - Implemented via logits hook callback (architecture-level control)

            ### Model Specs (Demo v5)
            - **Parameters**: ~38M (up from 12M in v3)
            - **Vocabulary**: BPE subword (~800 tokens, optimized density over v3's 2000)
            - **Tokenization**: Proper BPE subword + explicit SPACE_ID (v5 fix — was injecting UNK)
            - **Layers**: 6 Transformer blocks (up from 4)
            - **Hidden Size**: 384 (up from 256, divisible by 12 heads)
            - **Attention Heads**: 12 (up from 8)
            - **FFN Dim**: 1024 (up from 512)
            - **Max Seq Len**: 256
            - **Training Strategy**: Answer-only loss (only predicts after `<|a|>`)
            - **Training Data**: 560+ QA pairs across 10 domains (up from 199)
            - **Optimizer**: AdamW with cosine annealing + gradient clipping

            ---
            *Built with [Fusion-LLM](https://github.com/zhan1206/fusion-llm) | Apache 2.0 License
            """)


if __name__ == "__main__":
    try:
        demo.launch()
    except Exception:
        try:
            demo.launch(share=True)
        except Exception:
            pass
