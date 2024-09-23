# H&M Outfit Recommender with Generative AI

This project implements an advanced outfit recommendation system using the H&M dataset, combining deep learning techniques for semantic similarity and natural language generation. Prepared for Flipkart Grid 5.0

## Overview

### Core Components:

1. **Sentence Transformer Model**: Utilizes the `sentence-transformers/paraphrase-MiniLM-L6-v2` model for encoding product descriptions into high-dimensional embeddings.

2. **GPT-2 Medium**: Implements the `gpt2-medium` model for generating natural language fashion suggestions.

3. **Cosine Similarity**: Employs cosine similarity metrics for efficient similarity search in the embedding space.

4. **Dynamic Recommendation Engine**: Features a feedback-driven system that adapts recommendations based on user preferences.

### Features:

- **Multi-modal Data Integration**: Combines textual product descriptions with image data for a comprehensive representation.
- **Efficient Similarity Search**: Leverages PyTorch's GPU acceleration for fast cosine similarity computations.
- **Natural Language Interaction**: Utilizes GPT-2 for generating contextual fashion advice.
- **Adaptive Recommendations**: Implements a feedback loop to refine suggestions based on user input.

## Implementation

### Preprocessing:
- Combines multiple product attributes into a single text field for richer embeddings.
- Handles missing data in product descriptions.

### Architecture:
1. **Embedding Generation**:
   - Uses SentenceTransformer to create dense vector representations of product descriptions.
   - Embeddings are stored in GPU memory for faster retrieval.

2. **Similarity Computation**:
   - Implements `util.pytorch_cos_sim` for efficient batch similarity calculations.

3. **Natural Language Generation**:
   - Employs GPT-2 Medium for generating fashion suggestions.
   - Uses custom prompts to guide the generation process.

4. **Recommendation Engine**:
   - Implements top-k retrieval with dynamic exclusion of disliked items.
   - Utilizes PyTorch's `topk` function for efficient ranking.

### User Interaction:
- Implements a conversational interface for query input and feedback collection.
- Displays product images alongside recommendations for visual context.

## Performance

- GPU acceleration is used where available for embedding computations and similarity searches.
- Efficient batching of operations to minimize memory transfers.
- Caching of embeddings to reduce redundant computations.

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
- pandas
- matplotlib
- scikit-learn

## Setup and Execution

1. Install required packages:
   ```
   pip install transformers sentence-transformers torch pandas matplotlib scikit-learn
   ```

2. Download the H&M dataset and update the file path in the script.

3. Run the main script to start the interactive fashion bot:
   ```python
   python fashion_recommender.py
   ```

## Acknowledgements

This project utilizes the H&M dataset and builds upon state-of-the-art language models from Hugging Face.
