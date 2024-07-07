# Base image
FROM continuumio/miniconda3

# Create the conda environment and install dependencies
RUN conda create -n famix_env python=3.8 \
    && conda install -n famix_env -c pytorch pytorch torchvision torchaudio cudatoolkit=10.2 -y \
    && conda install -n famix_env -c conda-forge numpy matplotlib pillow scikit-learn -y \
    && conda install -n famix_env pip -y \
    && conda run -n famix_env pip install pickle5 tqdm ftfy regex \
    && conda run -n famix_env pip install git+https://github.com/openai/CLIP.git

# Activate the environment
RUN echo "source activate famix_env" > ~/.bashrc
ENV PATH /opt/conda/envs/famix_env/bin:$PATH

# Default command
CMD ["python"]

