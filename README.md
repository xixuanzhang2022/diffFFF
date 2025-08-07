A comprehensive toolkit for analyzing content patterns, community interactions, network structure, and diffusion dynamics in social media —developed in Python.
The code in this repository was originally developed to support the data collection and analyses presented in the paper: **Zhang, X. (2023). Diffusion Dynamics and Digital Movement: the Emergence and Proliferation of the German-speaking #FridaysForFuture Network on Twitter. *Social Movement Studies*. [https://doi.org/10.1177/14614448251336418](https://doi.org/10.1080/14742837.2023.2211015)**


**Data Preparation & Loading**

load_data.py: Handles importing and initial preprocessing of raw Twitter or cascade datasets.

**Cascade Modeling & Metrics**

cascade_builder.py: Constructs cascade structures from social interaction logs.

cascade_network_metric.py: Computes network-level metrics for diffusion cascades.

cascade_analysis.py: Performs in-depth analysis of cascade behavior.

centrality_sliding.py: Tracks changes in centrality over time using sliding windows.

time_series_analysis.py: Analyzes temporal dynamics within cascades.

**Embedding & Clustering**

embedding_clustering.py: Applies embeddings (e.g., user or content-level) and clustering techniques for cascade segmentation.

**Topic & Content Analysis**

topic_model.py: Extracts thematic patterns from cascade content via topic modeling.

**Pipeline Entry Point**

main.py: Orchestrates overall analysis, allowing streamlined execution of loading, modeling, and analysis workflows.
