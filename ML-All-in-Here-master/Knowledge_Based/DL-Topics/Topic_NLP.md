# Natural Language Processing (My mission is to expand each point listed here)

# Recurrent Neural Networks
    + Architectures (Limitations and inspiration behind every model) ([Blog 1](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)) ([Blog 2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/))
	    + Vanilla
		+ GRU
		+ LSTM
		+ Bidirectional
	+ Vanishing and Exploding Gradients
# Word Embeddings (pratical cases: https://github.com/zlsdu/Word-Embedding)
1. Word2Vec:
    - Word2vec is a neural network structure to generate word embedding by training the model on a supervised classification problem. 
    - Architecture: Word2vec has two algorithm names are **“continuous bag of words” (CBOW) and “skip-gram” (SG)**. 
    - Theory: Word2vec is a predictive model which learns their vectors in order to improve their predictive ability of Loss(target word | context words; Vectors), i.e. the loss of predicting the target words from the context words given the vector representations, which is cast as a feed-forward neural network and optimized as such using SGD, etc.
    - Advantages: Word2Vec shows that we can use a vector (a list of numbers) to properly represent words in a way that captures semantic or meaning-related relationships (e.g. the ability to tell if words are similar, or opposites, or that a pair of words like “Stockholm” and “Sweden” have the same relationship between them as “Cairo” and “Egypt” have between them) as well as syntactic, or grammar-based, relationships (e.g. the relationship between “had” and “has” is the same as that between “was” and “is”).
    - Issue: Word2vec only takes local contexts into account and **does not take advantage of global context**. 

    ![model overview](/Knowledge_Based/pictures/word2vec.png)
    
    Example: https://towardsdatascience.com/word2vec-from-scratch-with-numpy-8786ddd49e72
	+ CBOW
	+ Glove
	+ FastText
	+ SkipGram, NGram
	+ ELMO
	+ OpenAI GPT
	+ BERT ([Blog](http://jalammar.github.io/illustrated-bert/))
# Transformers ([Paper](https://arxiv.org/abs/1706.03762)) ([Code](https://nlp.seas.harvard.edu/2018/04/03/attention.html)) ([Blog](http://jalammar.github.io/illustrated-transformer/))
	+ BERT ([Paper](https://arxiv.org/abs/1810.04805))
	+ Universal Sentence Encoder