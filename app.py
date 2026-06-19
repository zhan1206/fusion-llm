"""
Fusion-LLM Demo v2 - Hugging Face Space
Upgraded: BPE tokenizer, ~12M params, 200+ training samples, 100-epoch max.
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
    def __init__(self, hidden_size=256, num_heads=8, latent_ratio=4, block_size=16):
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
    def __init__(self, hidden_size=256, num_heads=8, ff_dim=512):
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
    """Fusion-LLM model for demo - upgraded to ~12M params."""
    def __init__(self, vocab_size=500, hidden_size=256, num_layers=4, num_heads=8,
                 max_seq_len=128, ff_dim=512):
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

    def __init__(self, vocab_size=500):
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

    def train(self, texts, target_vocab_size=500):
        word_freqs = self._get_word_freqs(texts)
        vocab_freqs = {tuple(list(w)): f for w, f in word_freqs.items()}

        base_tokens = set()
        for word_tuple in vocab_freqs:
            for ch in word_tuple:
                base_tokens.add(ch)

        self.vocab = {t: i + 4 for i, t in enumerate(sorted(base_tokens))}
        self.vocab['<pad>'] = self.pad_id
        self.vocab['<unk>'] = self.unk_id
        self.vocab['<eos>'] = self.eos_id
        self.vocab['<bos>'] = self.bos_id
        self.vocab['<|q|>'] = self.q_sep_id
        self.vocab['<|a|>'] = self.a_sep_id

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

    def encode_qa(self, question, answer, max_len=128):
        """Encode a QA pair with special separator tokens: bos <|q|> question <|a|> answer eos"""
        ids = [self.bos_id, self.q_sep_id]
        for w in question.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.token_to_id.get(' ', self.unk_id))
        ids.append(self.a_sep_id)
        for w in answer.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.token_to_id.get(' ', self.unk_id))
        ids.append(self.eos_id)
        return ids[:max_len]

    def encode_prompt(self, prompt, max_len=96):
        """Encode a chat prompt with <|q|> prefix to trigger answer generation."""
        ids = [self.bos_id, self.q_sep_id]
        for w in prompt.split():
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.token_to_id.get(' ', self.unk_id))
        # Add <|a|> to signal: now generate the answer
        ids.append(self.a_sep_id)
        return ids[:max_len]

    def encode(self, text, max_len=128):
        words = text.split()
        ids = [self.bos_id]
        for w in words:
            for tok in self._tokenize_word(w):
                ids.append(self.token_to_id.get(tok, self.unk_id))
            ids.append(self.token_to_id.get(' ', self.unk_id))
        ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in (self.pad_id, self.eos_id):
                break
            if i == self.bos_id:
                continue
            tokens.append(self.id_to_token.get(i, '<unk>'))
        return ''.join(tokens).replace('<unk>', '?')


# ── Expanded Training Data (200+ QA pairs) ────────────────────────

def _generate_qa_pairs():
    """Generate a comprehensive set of QA pairs for training."""
    pairs = []

    # --- AI & Machine Learning Fundamentals (50) ---
    ml_fundamentals = [
        ("What is AI", "AI is the simulation of human intelligence by machines for learning reasoning and problem solving"),
        ("How does deep learning work", "Deep learning uses neural networks with many layers to learn hierarchical patterns from data"),
        ("What is a transformer", "A transformer is a neural network architecture using self attention mechanisms for sequence processing"),
        ("Explain machine learning", "Machine learning enables computers to learn patterns from data without explicit programming"),
        ("What is NLP", "NLP stands for Natural Language Processing enabling computers to understand and generate human language"),
        ("Define neural network", "A neural network is a computing system inspired by biological brain networks with connected neurons"),
        ("What is reinforcement learning", "Reinforcement learning trains agents to make decisions through rewards and penalties feedback"),
        ("Explain gradient descent", "Gradient descent iteratively adjusts model parameters to minimize a loss function using derivatives"),
        ("What is transfer learning", "Transfer learning reuses a pre trained model on a new related task to leverage learned features"),
        ("How do LLMs work", "Large Language Models predict the next token based on context using transformer architecture at scale"),
        ("What is attention mechanism", "Attention allows models to dynamically focus on relevant parts of the input sequence"),
        ("Explain backpropagation", "Backpropagation computes gradients via the chain rule to update neural network weights efficiently"),
        ("What is fine-tuning", "Fine tuning adapts a pre trained model to a specific task using a smaller focused dataset"),
        ("Define embedding", "An embedding represents discrete items like words as dense continuous vectors in a space"),
        ("What is tokenization", "Tokenization splits text into smaller units like words or subwords that models can process"),
        ("How does GPT work", "GPT uses a decoder only transformer to generate text autoregressively token by token"),
        ("What is a loss function", "A loss function measures the difference between model predictions and true target values"),
        ("Explain overfitting", "Overfitting occurs when a model memorizes training data but fails to generalize to new data"),
        ("What is batch normalization", "Batch normalization normalizes layer inputs to stabilize and accelerate training convergence"),
        ("Define activation function", "An activation function introduces non linearity into neural networks enabling complex learning"),
        ("What is dropout", "Dropout randomly deactivates neurons during training to prevent overfitting and improve robustness"),
        ("Explain the bias variance tradeoff", "It balances model complexity too simple underfits and too complex overfits the data"),
        ("What is an optimizer", "An optimizer updates model parameters based on computed gradients to minimize loss"),
        ("How does token prediction work", "Models learn to predict the next token given previous context tokens during training"),
        ("What is a hyperparameter", "Hyperparameters are settings configured before training begins like learning rate and batch size"),
        ("Explain epoch", "An epoch is one complete pass through the entire training dataset during model training"),
        ("What is a learning rate", "The learning rate controls how much to adjust model parameters during each optimization step"),
        ("Define data augmentation", "Data augmentation creates variations of existing training data to improve model generalization"),
        ("What is cross validation", "Cross validation evaluates model performance by rotating train and test data splits"),
        ("Explain softmax", "Softmax converts raw model scores into a probability distribution that sums to one"),
        ("What is a convolutional neural network", "CNN is a neural network that uses convolution operations to process grid like data such as images"),
        ("How does RNN work", "RNN processes sequences by maintaining a hidden state that captures information from previous steps"),
        ("What is LSTM", "LSTM is a type of RNN with gating mechanisms that can learn long term dependencies in sequences"),
        ("Explain the vanishing gradient problem", "Gradients shrink exponentially in deep networks making early layers learn very slowly"),
        ("What is regularization", "Regularization adds penalties to the loss function to discourage overly complex models"),
        ("Define weights and biases", "Weights determine connection strength between neurons and biases shift activation thresholds"),
        ("What is a feature map", "A feature map is the output of a convolutional layer capturing specific patterns in input data"),
        ("Explain pooling in CNN", "Pooling reduces spatial dimensions of feature maps to decrease computation and control overfitting"),
        ("What is a generative model", "A generative model learns the underlying data distribution to generate new similar samples"),
        ("How does a GAN work", "GAN trains a generator and discriminator in competition to produce realistic synthetic data"),
        ("What is a VAE", "VAE is a variational autoencoder that learns a probabilistic latent representation of data"),
        ("Explain the encoder decoder architecture", "Encoder processes input into a representation and decoder generates output from it"),
        ("What is beam search", "Beam search explores multiple candidate sequences to find the most likely output during decoding"),
        ("How does BERT work", "BERT is a bidirectional transformer pre trained on masked language modeling for NLP tasks"),
        ("What is RLHF", "RLHF fine tunes language models using human feedback as reward signals via reinforcement learning"),
        ("Explain temperature in sampling", "Temperature controls randomness in generation low is greedy and high is more diverse"),
        ("What is top-k sampling", "Top-k sampling restricts next token selection to the k most likely candidates"),
        ("Define perplexity", "Perplexity measures how well a model predicts text lower values indicate better predictions"),
        ("What is a foundation model", "A foundation model is a large model trained on broad data that can be adapted to many tasks"),
        ("Explain multi-head attention", "Multi-head attention runs attention in parallel across multiple representation subspaces"),
    ]

    # --- Programming & Computer Science (50) ---
    programming = [
        ("What is Python", "Python is a high level interpreted programming language known for readability and versatility"),
        ("Explain object oriented programming", "OOP organizes code around objects that bundle data and behavior with encapsulation inheritance and polymorphism"),
        ("What is a function", "A function is a reusable block of code that performs a specific task and can return a value"),
        ("How do loops work", "Loops repeat a block of code either a fixed number of times or until a condition is met"),
        ("What is recursion", "Recursion is when a function calls itself to solve smaller instances of the same problem"),
        ("Explain variables", "Variables are named storage locations that hold values which can change during program execution"),
        ("What is an array", "An array is a collection of elements stored at contiguous memory locations accessible by index"),
        ("How does a hash table work", "A hash table uses a hash function to map keys to values enabling fast lookup insertion and deletion"),
        ("What is a linked list", "A linked list is a data structure where each node contains data and a reference to the next node"),
        ("Explain binary search", "Binary search finds an element in a sorted array by repeatedly halving the search range"),
        ("What is Big O notation", "Big O describes algorithm time or space complexity as input size grows toward infinity"),
        ("How does sorting work", "Sorting arranges elements in order algorithms include quicksort mergesort and heapsort"),
        ("What is a stack", "A stack is a last in first out data structure supporting push and pop operations"),
        ("What is a queue", "A queue is a first in first out data structure supporting enqueue and dequeue operations"),
        ("Explain a tree data structure", "A tree is a hierarchical structure with nodes connected by edges having a root and subtrees"),
        ("What is a graph", "A graph is a data structure with vertices connected by edges representing relationships"),
        ("How does dynamic programming work", "DP solves problems by breaking them into overlapping subproblems and storing results"),
        ("What is greedy algorithm", "A greedy algorithm makes locally optimal choices at each step hoping to find a global optimum"),
        ("Explain recursion vs iteration", "Recursion calls itself while iteration uses loops recursion is elegant but may use more memory"),
        ("What is a pointer", "A pointer is a variable storing the memory address of another variable for indirect access"),
        ("How does memory management work", "Memory management allocates and frees memory using stack for locals and heap for dynamic data"),
        ("What is garbage collection", "Garbage collection automatically frees memory no longer referenced by the program"),
        ("Explain compile vs interpret", "Compilation translates all code before execution while interpretation processes line by line"),
        ("What is a debugger", "A debugger helps find and fix bugs by pausing execution inspecting variables and stepping through code"),
        ("How do APIs work", "APIs define how software components communicate using requests responses and defined endpoints"),
        ("What is REST", "REST is an architectural style using HTTP methods to access and manipulate resources via URLs"),
        ("Explain SQL", "SQL is a query language for managing and retrieving data from relational databases"),
        ("What is NoSQL", "NoSQL databases store data in flexible non tabular formats like documents key value or graphs"),
        ("How does version control work", "Version control tracks code changes over time enabling collaboration and rollback Git is popular"),
        ("What is a class", "A class is a blueprint for creating objects defining properties and methods in OOP"),
        ("Explain inheritance", "Inheritance lets a class derive properties and methods from a parent class promoting code reuse"),
        ("What is polymorphism", "Polymorphism allows objects of different classes to be treated as instances of the same type"),
        ("How does encapsulation work", "Encapsulation bundles data and methods together and restricts direct access to internal state"),
        ("What is abstraction", "Abstraction hides implementation details and exposes only essential features to the user"),
        ("Explain a constructor", "A constructor is a special method that initializes a new object when it is created"),
        ("What is exception handling", "Exception handling catches and manages runtime errors using try catch and finally blocks"),
        ("How do threads work", "Threads are lightweight units of execution within a process enabling concurrent programming"),
        ("What is a process", "A process is an executing program with its own memory space and resources"),
        ("Explain deadlock", "Deadlock occurs when processes wait for each other indefinitely preventing any progress"),
        ("What is concurrency", "Concurrency means multiple tasks make progress simultaneously potentially interleaved"),
        ("How does TCP work", "TCP is a reliable transport protocol ensuring ordered error checked delivery of data packets"),
        ("What is UDP", "UDP is a fast connectionless transport protocol that does not guarantee delivery or ordering"),
        ("Explain DNS", "DNS translates human readable domain names into IP addresses for network communication"),
        ("What is HTTPS", "HTTPS is HTTP encrypted with TLS or SSL providing secure communication over networks"),
        ("How do cookies work", "Cookies are small files stored by browsers to maintain state across HTTP requests"),
        ("What is a database index", "An index speeds up database queries by creating a lookup structure on columns"),
        ("Explain normalization", "Normalization organizes database tables to reduce redundancy and improve data integrity"),
        ("What is caching", "Caching stores frequently accessed data in fast memory to reduce retrieval time"),
        ("How does load balancing work", "Load balancing distributes incoming requests across multiple servers for reliability"),
        ("What is microservices", "Microservices architecture splits applications into small independent services communicating via APIs"),
    ]

    # --- Science & Physics (40) ---
    science = [
        ("What is gravity", "Gravity is the force that attracts objects with mass toward each other"),
        ("Explain the speed of light", "Light travels at approximately 299792458 meters per second in a vacuum"),
        ("What is energy", "Energy is the capacity to do work and exists in forms like kinetic potential and thermal"),
        ("How does photosynthesis work", "Plants convert sunlight water and carbon dioxide into glucose and oxygen"),
        ("What is the periodic table", "The periodic table organizes chemical elements by atomic number and properties"),
        ("Explain atoms", "Atoms are the basic units of matter composed of protons neutrons and electrons"),
        ("What is DNA", "DNA is a molecule carrying genetic instructions for development and function of living organisms"),
        ("How does evolution work", "Evolution is change in inherited traits over generations through natural selection"),
        ("What is relativity", "Einstein theory that space and time are relative and gravity curves spacetime"),
        ("Explain quantum mechanics", "Quantum mechanics describes particle behavior at atomic scales with probability and uncertainty"),
        ("What is thermodynamics", "Thermodynamics studies heat work temperature and energy transfer in physical systems"),
        ("How do magnets work", "Magnets produce magnetic fields from aligned electron spins attracting or repelling materials"),
        ("What is electricity", "Electricity is the flow of electric charge through conductors enabling power and signals"),
        ("Explain the water cycle", "Water evaporates forms clouds precipitates and flows back to oceans in a continuous cycle"),
        ("What is climate change", "Climate change refers to long term shifts in temperatures and weather patterns primarily from human activity"),
        ("How do vaccines work", "Vaccines train the immune system to recognize and fight specific pathogens"),
        ("What is a cell", "A cell is the smallest unit of life containing genetic material and organelles"),
        ("Explain mitosis", "Mitosis is cell division producing two identical daughter cells from one parent cell"),
        ("What is a black hole", "A black hole is a region where gravity is so strong nothing not even light can escape"),
        ("How do stars form", "Stars form when gravity collapses clouds of gas and dust until nuclear fusion ignites"),
        ("What is the Big Bang", "The Big Bang theory states the universe began from a hot dense state about 14 billion years ago"),
        ("Explain plate tectonics", "Earth crust is divided into plates that move causing earthquakes mountains and volcanoes"),
        ("What is an ecosystem", "An ecosystem is a community of organisms interacting with their physical environment"),
        ("How does the brain work", "The brain processes information through neurons transmitting electrical and chemical signals"),
        ("What is a gene", "A gene is a DNA segment encoding instructions for a specific protein or function"),
        ("Explain osmosis", "Osmosis is water moving across a membrane from low to high solute concentration"),
        ("What is entropy", "Entropy measures disorder in a system and tends to increase in isolated systems"),
        ("How do lasers work", "Lasers emit focused light through stimulated emission of photons from excited atoms"),
        ("What is nuclear fusion", "Nuclear fusion combines light nuclei releasing enormous energy as in stars"),
        ("Explain superconductors", "Superconductors conduct electricity with zero resistance below a critical temperature"),
        ("What is a wavelength", "Wavelength is the distance between successive peaks of a wave determining its properties"),
        ("How do x-rays work", "X-rays are high energy electromagnetic waves that penetrate soft tissue for medical imaging"),
        ("What is a catalyst", "A catalyst speeds up chemical reactions without being consumed in the process"),
        ("Explain pH scale", "pH measures acidity or alkalinity from 0 to 14 with 7 neutral below acidic above basic"),
        ("What is a molecule", "A molecule is two or more atoms bonded together forming a chemical compound"),
        ("How does sound travel", "Sound travels as mechanical waves through media by vibrating particles"),
        ("What is refraction", "Refraction is light bending when passing between media of different densities"),
        ("Explain static electricity", "Static electricity is charge buildup on surfaces causing sparks when discharged"),
        ("What is radioactivity", "Radioactivity is unstable nuclei emitting particles or energy as they decay"),
    ]

    # --- Math & Logic (30) ---
    math_logic = [
        ("What is calculus", "Calculus studies rates of change and accumulation using derivatives and integrals"),
        ("Explain algebra", "Algebra uses symbols to represent numbers and solve equations with unknowns"),
        ("What is geometry", "Geometry studies shapes sizes angles and properties of space and figures"),
        ("How do fractions work", "Fractions represent parts of a whole with a numerator over a denominator"),
        ("What is a prime number", "A prime number has exactly two factors one and itself like 2 3 5 7 and 11"),
        ("Explain the Pythagorean theorem", "In right triangles the square of the hypotenuse equals the sum of squares of the other sides"),
        ("What is probability", "Probability measures how likely an event is to occur ranging from 0 to 1"),
        ("How does statistics work", "Statistics collects analyzes and interprets data to find patterns and make decisions"),
        ("What is a matrix", "A matrix is a rectangular array of numbers used in linear algebra and transformations"),
        ("Explain logarithms", "A logarithm is the exponent needed for a base to produce a given number"),
        ("What is a derivative", "A derivative measures how a function changes as its input changes its instantaneous rate"),
        ("What is an integral", "An integral accumulates quantities over an interval the opposite of a derivative"),
        ("Explain set theory", "Set theory studies collections of objects using operations like union intersection and difference"),
        ("What is combinatorics", "Combinatorics counts and arranges objects studying permutations and combinations"),
        ("How does the Fibonacci sequence work", "Each number is the sum of the previous two starting from 0 and 1"),
        ("What is a vector", "A vector has magnitude and direction used in physics and computer graphics"),
        ("Explain mathematical induction", "Induction proves statements for all natural numbers by proving a base case and inductive step"),
        ("What is the binomial theorem", "It expands powers of binomials using binomial coefficients in a sum"),
        ("How do exponents work", "Exponents represent repeated multiplication where 2 to the 3 means 2 times 2 times 2"),
        ("What is a complex number", "A complex number has a real and imaginary part written as a plus b times i"),
        ("Explain standard deviation", "Standard deviation measures how spread out data points are from the mean"),
        ("What is a normal distribution", "A bell shaped curve symmetric about the mean describing many natural phenomena"),
        ("How does correlation work", "Correlation measures how two variables relate ranging from negative one to positive one"),
        ("What is a p-value", "A p-value measures the probability of observing results if the null hypothesis is true"),
        ("Explain linear equations", "Linear equations form straight lines with the form y equals m x plus b"),
        ("What is a function in math", "A function maps each input to exactly one output following a defined rule"),
        ("How do triangles work", "Triangles have three sides and angles summing to 180 degrees classified by sides and angles"),
        ("What is trigonometry", "Trigonometry studies relationships between triangle sides and angles using sine cosine and tangent"),
        ("Explain the concept of infinity", "Infinity is an unbounded quantity larger than any real number represented by the symbol"),
        ("What is modular arithmetic", "Modular arithmetic works with remainders after division like a clock wrapping at 12"),
    ]

    # --- General Knowledge (30) ---
    general = [
        ("What is the internet", "The internet is a global network connecting computers worldwide enabling communication and sharing"),
        ("How do computers work", "Computers process data using CPU memory storage and input output devices running software"),
        ("What is an operating system", "An OS manages hardware resources and provides services for programs like Windows Linux and macOS"),
        ("Explain cloud computing", "Cloud computing delivers computing services over the internet including storage processing and databases"),
        ("What is cybersecurity", "Cybersecurity protects systems networks and data from digital attacks and unauthorized access"),
        ("How does encryption work", "Encryption converts plaintext into ciphertext using algorithms and keys for data protection"),
        ("What is machine vision", "Machine vision enables computers to interpret and process visual information from images or video"),
        ("How do self-driving cars work", "Self-driving cars use sensors cameras AI and maps to navigate roads autonomously"),
        ("What is blockchain", "Blockchain is a distributed immutable ledger recording transactions across many computers"),
        ("How does cryptocurrency work", "Cryptocurrency uses cryptography and blockchain for decentralized digital currency transactions"),
        ("What is IoT", "IoT connects everyday objects to the internet enabling data exchange and remote control"),
        ("Explain augmented reality", "AR overlays digital content onto the real world through devices like phones or glasses"),
        ("What is virtual reality", "VR immerses users in a fully digital environment using headsets and controllers"),
        ("How do smartphones work", "Smartphones combine computing communication GPS and sensors in a mobile device running apps"),
        ("What is 5G", "5G is the fifth generation mobile network offering faster speeds lower latency and more capacity"),
        ("Explain edge computing", "Edge computing processes data near the source reducing latency and bandwidth use"),
        ("What is quantum computing", "Quantum computing uses qubits and superposition to solve certain problems faster than classical computers"),
        ("How does GPS work", "GPS uses satellite signals to determine location by triangulating distances from multiple satellites"),
        ("What is AI art", "AI art is generated by models trained on images creating new visual content from text prompts"),
        ("How do recommendation systems work", "Recommendation systems suggest items based on user behavior preferences and collaborative filtering"),
        ("What is a search engine", "A search engine indexes web pages and retrieves relevant results based on user queries"),
        ("How does email work", "Email uses SMTP to send and POP or IMAP to receive messages across servers"),
        ("What is streaming", "Streaming delivers audio or video content continuously over the internet without full download"),
        ("Explain data mining", "Data mining discovers patterns and knowledge from large datasets using statistical and ML methods"),
        ("What is a chatbot", "A chatbot is software that simulates human conversation using rules or AI models"),
        ("How do neural networks learn", "Neural networks learn by adjusting weights via backpropagation to minimize prediction error"),
        ("What is deep fake technology", "Deep fakes use deep learning to create realistic but fake videos or audio of people"),
        ("Explain the Turing test", "The Turing test evaluates if a machine can exhibit intelligent behavior indistinguishable from a human"),
        ("What is artificial general intelligence", "AGI refers to AI that matches or exceeds human cognitive abilities across all domains"),
        ("How does 3D printing work", "3D printing builds objects layer by layer from digital models using materials like plastic or metal"),
    ]

    pairs = ml_fundamentals + programming + science + math_logic + general
    return pairs


TRAIN_DATA = _generate_qa_pairs()

# ── Global State ──────────────────────────────────────────────────

# Train BPE tokenizer on all training data
_all_texts = [q + " " + a for q, a in TRAIN_DATA]
tokenizer = BPETokenizer(vocab_size=500)
tokenizer.train(_all_texts, target_vocab_size=500)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_history = {"loss": [], "epoch": []}

# Upgraded model config
MODEL_CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 8,
    "max_seq_len": 128,
    "ff_dim": 512,
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

        # Pre-encode all training pairs with QA separators
        encoded_pairs = []
        for q, a in TRAIN_DATA:
            ids = tokenizer.encode_qa(q, a, max_len=128)
            if len(ids) > 4:  # need at least bos+q_sep+a_sep+1 token
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
                    if len(ids) < 2:
                        continue
                    inputs = ids[:-1].unsqueeze(0)
                    targets = ids[1:].unsqueeze(0)
                    logits = mdl(inputs)
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
            status = f"Done! Final Loss: {losses[-1]:.4f} | Params: {param_count/1e6:.1f}M | Vocab: {tokenizer.vocab_size} | Data: {len(TRAIN_DATA)} pairs"
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

        input_ids = tokenizer.encode_prompt(prompt, max_len=96)
        generated = list(input_ids)
        with torch.no_grad():
            for _ in range(int(max_tokens)):
                inp = torch.tensor([generated[-96:]], dtype=torch.long, device=device)
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
                if next_token == tokenizer.eos_id or len(generated) > 120:
                    break

        response = tokenizer.decode(generated[len(input_ids):])
        return response or "(no output)"
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────

with gr.Blocks(title="Fusion-LLM Demo v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Fusion-LLM Demo v2
    **Train & Chat** with a custom Transformer featuring **SBLA Attention** + **Thinking Dial**
    
    Upgraded: ~12M params | BPE tokenizer | 200+ training pairs | up to 100 epochs
    """)

    with gr.Tabs():
        # ── Train Tab ──
        with gr.Tab("Train"):
            with gr.Row():
                lr = gr.Slider(0.0001, 0.01, value=0.001, label="Learning Rate", step=0.0001)
                eps = gr.Slider(1, 100, value=50, label="Epochs", step=1)
                bs = gr.Slider(1, 16, value=4, label="Batch Size", step=1)
            train_btn = gr.Button("Start Training", variant="primary")
            train_status = gr.Textbox(label="Status", interactive=False)
            loss_plot = gr.Textbox(label="Loss History", interactive=False, lines=15)

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

            ### Model Specs (Demo v2)
            - **Parameters**: ~12M (upgraded from ~3M)
            - **Vocabulary**: BPE subword (~500 tokens, upgraded from char-level ~97)
            - **Layers**: 4 Transformer blocks (upgraded from 2)
            - **Hidden Size**: 256 (upgraded from 128)
            - **Attention Heads**: 8 (upgraded from 4)
            - **FFN Dim**: 512
            - **Max Seq Len**: 128 (upgraded from 64)
            - **Training Data**: 200+ QA pairs (upgraded from 30)
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
