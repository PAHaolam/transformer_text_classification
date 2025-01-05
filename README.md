link dataset
https://drive.google.com/drive/folders/1Lu9axyLkw7dMx80uLRgvCnZsmNzhJWAa

## Model Training

<pre> <code>python main.py</code></pre> 
For loging training procedure, you can use [Weight & Bias](https://wandb.ai/) ([How to initialize it](https://docs.wandb.ai/quickstart)) by setting an argument.
<pre> <code>python train.py --wandb online</code></pre>

## Evaluation with Output Files
<pre> <code>python evaluate.py --checkpoint_path your_checkpoint_dir</code></pre>
