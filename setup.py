# Run this after install requirements file packages

# Use Transformers offline by downloading the files ahead of time, 
# and then point to their local path when you need to use them offline. 
# https://huggingface.co/docs/transformers/installation
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


# 1. Download your files ahead of time with PreTrainedModel.from_pretrained():
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# 2. Save your files to a specified directory with PreTrainedModel.save_pretrained():
tokenizer.save_pretrained("src/test_models/custom_sentence-transformer/all-MiniLM-L6-v2")
model.save_pretrained("src/models/custom_sentence-transformer/all-MiniLM-L6-v2")

# 3. Now when you’re offline, reload your files with PreTrainedModel.from_pretrained() from the specified directory:
# step 3 will be ran in the main scripts