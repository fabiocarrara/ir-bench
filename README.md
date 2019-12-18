# Large-scale Image Retrieval using Textual Search Engines

The code implements the techniques presented in the following papers:

> Amato, G., Bolettieri, P., Carrara, F., Falchi, F. and Gennaro, C., 2018, June. Large-Scale Image Retrieval with Elasticsearch. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (pp. 925-928). ACM.
> 
> Amato, G., Carrara, F., Falchi, F., Gennaro, C. and Vadicamo, L., 2019. Large-scale instance-level image retrieval. Information Processing & Management, p.102100.

## Main files

 - [baseline.py](baseline.py): non-approximated bruteforce matching using original features
 - [scalar_quantization.py](scalar_quantization.py): thresholded scalar quantization in main memory
 - [deep_permutation_basics.py](deep_permutation_basics.py): example for computing CReLU-ed permutations
 - [ivfpq.py](ivfpq.py): main-memory product-quantized inverted files (using FAISS)
 - [sq_lucene.py](sq_lucene.py): implements thresholded scalar quantization in Apache Lucene (see [lucene/Dockerfile](lucene/Dockerfile))
 - [sq_elastic.py](sq_elastic.py): implements thresholded scalar quantization in Elasticsearch (see [elasticsearch/setup_elastic.sh](elasticsearch/setup_elastic.sh))
 
 