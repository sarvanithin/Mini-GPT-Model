
# minGPT

minGPT

minGPT is a PyTorch re-implementation of GPT, covering both training and inference. The primary goals of minGPT are to be small, clean, interpretable, and educational. Unlike some sprawling GPT implementations, minGPT maintains simplicity, with the entire codebase consisting of approximately 300 lines of code (refer to mingpt/model.py). The core concept involves a sequence of indices fed into a Transformer, producing a probability distribution over the next index in the sequence. The implementation focuses on clever batching, both across examples and sequence length, for efficiency.

Library Installation
If you wish to integrate minGPT into your project, follow these steps:

bash
Copy code
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .
Usage
Here's how you can instantiate a GPT-2 model (124M param version) using minGPT:

python
Copy code
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257  # OpenAI's model vocabulary
model_config.block_size = 1024   # OpenAI's model block_size (i.e., input context length)

model = GPT(model_config)
And here's an example of training:

python
Copy code
# Your subclass of torch.utils.data.Dataset that emits examples
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = YourDataset()

from mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4  # Many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32

trainer = Trainer(train_config, model, train_dataset)
trainer.run()
Refer to demo.ipynb for a more concrete example.

Unit Tests
Although the coverage is not comprehensive, you can run the following command for tests:

bash
Copy code
python -m unittest discover tests
Todos
Add GPT-2 finetuning demo on an arbitrary given text file.
Introduce a dialog agent demo.
Enhance documentation of outcomes for existing projects (adder, chargpt).
Add mixed precision and related training scaling improvements.
Support distributed training.
Reproduce some benchmarks in projects/ (e.g., text8 or other language modeling).
Implement proper logging instead of print statements.
Include a requirements.txt file.
Enable loading of many other model weights other than just gpt2-*.
References
Code
openai/gpt-2: Model definition in TensorFlow but without the training code.
openai/image-gpt: Contains some modern GPT-3-like modifications in its code.
huggingface/transformers: Provides a language-modeling example, full-featured but somewhat challenging to trace.
Papers + Implementation Notes
Improving Language Understanding by Generative Pre-Training (GPT-1)
Trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads).
Adam max learning rate of 2.5e-4.
Linear LR decay, warmup over the first 2000 updates, and annealed to 0 using a cosine schedule.
Trained for 100 epochs on minibatches of 64 randomly sampled contiguous sequences of 512 tokens.
Bytepair encoding (BPE) vocabulary with 40,000 merges.
GPT-1 model is 12 layers and d_model 768, ~117M params.
Language Models are Unsupervised Multitask Learners (GPT-2)
LayerNorm moved to the input of each sub-block.
Additional layer normalization added after the final self-attention block.
Modified initialization accounting for accumulation on the residual path with model depth.
Vocabulary expanded to 50,257.
Context size increased from 512 to 1024 tokens.
Larger batch size of 512.
GPT-2 used 48 layers and d_model 1600, ~1.542B params.
Language Models are Few-Shot Learners (GPT-3)
GPT-3: 96 layers, 96 heads, with d_model of 12,288 (175B parameters).
Alternating dense and locally banded sparse attention patterns.
Feedforward layer four times the size of the bottleneck layer.
Context window of nctx = 2048 tokens.
Adam with β1 = 0.9, β2 = 0.95, and eps = 10^-8.
Linear LR warmup over the first 375 million tokens, cosine decay over 260 billion tokens.
Gradual increase in batch size linearly.
Full 2048-sized time context window is always used.
Generative Pretraining from Pixels (Image GPT)
Uses identity permutation πi = i for 1 ≤ i ≤ n when working with images.
Creates a 9-bit color palette by clustering (R, G, B) pixel values using k-means with k = 512.
Largest model (iGPT-XL) contains L = 60 layers with an embedding size of d = 3072 (6.8B parameters).
iGPT-L is essentially identical to GPT-2 with L = 48 layers, d = 1536 (1.4B parameters).
iGPT-M: 455M parameter model with L = 36 and d = 1024.
iGPT-S: 76M parameter model with L = 24 and d = 512.
Pre-training batch sizes and Adam configurations are adapted to image input.

### License

MIT
