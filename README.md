A Summary of the SHAP Approach As Can Be Applied to Explaining Deep Learning Classifiers
By: Tian Liao and Shreena Mehta
Introduction
Background
By the fall of 2020, more than 1 million people have died from COVID-19 world wide due to the pneumonia-like symptoms in combination of immune system reactions to it. The pandemic has affected lives across continents and ages, while the virus targets the lungs of the people it infects. Because of this, COVID-19 can be classified as a respiratory illness that can be distinguished by comparing infected lungs to healthy lungs. Many samples of X-Ray scans of the chests of infected and uninfected patients have been compiled in the COVID-19 Chest X-Ray / CT Imaging Dataset [5]. This dataset is compiled with the hopes of bringing together medical and machine learning practitioners that could pool together their joint efforts in creating a suitable algorithm for accurate and meaningful classification of infected and healthy lungs from X-Ray images.

The goals of this project are threefold: 
- (1) to explore development of a machine learning algorithm to distinguish chest X-rays of individuals with respiratory illness testing positive for COVID-19 from other X-rays, 
- (2) to promote discovery of patterns in such X-rays via machine learning interpretability algorithms, and 
- (3) to build more robust and extensible machine learning infrastructure trained on a variety of data types, to aid in the global response to COVID-19. [5]
Purpose
In our goal of developing an AI system that detects pneumonia on the above dataset, Local Interpretable Model-Agnostic Explanations (i.e. LIME) [3] was previously implemented with a ResNet Network in our Colab Notebook with GPU Enabled kernel. The SHapley Additive exPlanations method (SHAP) can be very well be applied to explain deep learning classifiers such as those used in the LIME implementation. In writing this paper, our goal would be to summarize this application of SHAP as described in A Unified Approach to Interpreting Model Predictions [2], as well as provide consolidated details of the SHAP method's parameters and features.

Additive Feature Attribution Methods
The best explanation of a simple model is the model itself; it perfectly represents itself and is easy to understand.[2]

Complex models, including, but not limited to, ensemble methods and deep neural networks are not able to be easily explained using their original models, for the primary reason of the original models being inherently difficult to understand. This predicament can be avoided by using an approximated interpretation of that original model. These such explanation models often use mapping functions to map simplified inputs to the original inputs.

Additive feature attribution methods have an explanation model that is a linear function of binary variables [2]

There are many methods fitting this definition which have been summarized below.

Methods:
LIME [8]
Local Interpretable Model-Agnostic Explanations (i.e. LIME) uses a local linear explanation model that closely matches the Additive feature attribution methods definition.

The LIME method interprets individual model predictions based on locally approximating the model around a given prediction [9]

The mapping for LIME converts a binary vector of interpretable inputs into the original input space, turning text vectors into original word count, and images into representations of super pixel replacement status. LIME also minimizes its objective function and enforces accuracy by calculating loss over a set of samples in the simplified input space weighted by the local kernel.

DeepLIFT
DeepLIFT is a method more specialized for recursive prediction explanations. Because it assigns a binary value "that represents the effect of that input being set to a reference value as opposed to its original value" to each input. The model can be classified as an Additive feature attribution method through some derivation and manipulation of its explanation.

Layer-Wise Relevance Propagation
Layer-Wise Relevance Propagation interprets deep network predictions in a way that is similar to DeepLIFT except with its activation neuron references set to zero. Like DeepLIFT, this model can be classified as an Additive feature attribution method through some derivation and manipulation of its explanation.

Classic Shapley Value Estimation [2]
The following methods use classic equations from cooperative game theory to compute explanations of model predictions.

Shapley regression values
Shapley regression values are feature importances for linear models in the presence of multicollinearity. [2]

This method involves retraining the model on all feature subsets and assigns an importance value to each feature that represents the effect on the model prediction of including that feature. In summary, two models are trained: one with present feature, and one with witheld feature. Then, the current input is passed through both and the difference in predictions is taken for all possible subsets. Finally, the Shapley regression values are then computed as a weighted average of all possible differences, and used as feature attributions.

This model can be classified as an additive feature attribution method through some derivation and manipulation of its explanation.

Shapley sampling values
Shapley sampling values remove the need to retrain the model by applying sampling approximations to the previously mentioned Shapley regression values and by "approximating the effect of removing a variable from the model by integrating over samples from the training dataset"[2].

Similarly to Shapley regression values, this model can be classified as an additive feature attribution method through some derivation and manipulation of its explanation.

Quantitative Input Influence
Quantitative Input Influence independently proposes a sampling approximation to Shapley values that is nearly identical to Shapley sampling values, but addresses more than feature attributions.

Similarly to Shapley sampling values, this model can be classified as an additive feature attribution method through some derivation and manipulation of its explanation.

Simple Properties Uniquely Determine Additive Feature Attributions [2]
The following properties are applicable to the Classic Shapley Value Estimation methods and were previously unknown for other additive feature attribution methods.

Property 1 - Local Accuracy
When approximating the original model for a specific input , local accuracy requires the explanation model to at least match the output of the original model for the simplified input (which corresponds to the original input).

Property 2 - Missingness
If the simplified inputs represent feature presence, then missingness requires features missing in the original input to have no impact.

Property 3 - Consistency
Consistency states that if a model changes so that some simplified input’s contribution increases or stays the same regardless of the other inputs, that input’s attribution should not decrease

Theorem 1
Under Properties 1-3, for a given simplified input mapping, there is only one possible additive feature attribution method. Methods not based on Shapley values violate local accuracy and/or consistency.

SHAP (SHapley Additive exPlanation) Values
SHAP (SHapley Additive exPlanation) Values are intended as a unified measure of feature importance and are are the Shapley values of a conditional expectation function of the original model. [2][4] SHAP values adhere to Properties 1-3 and uses conditional expectations to define simplified inputs.

To approximate SHAP Values, one can use either model-agnostic approximation methods (Shapley sampling values, Kernel SHAP) or model-type-specific approximation methods (Linear SHAP, Low-Order SHAP, Max SHAP, Deep SHAP). NOTE: When using these methods, feature independence and model linearity are two optional assumptions simplifying the computation of the expected values.

Model-Agnostic Approximations
If we assume feature independence when approximating conditional expectations, then SHAP values can be estimated directly using the Shapley sampling values method or equivalently the Quantitative Input Influence method, which use a sampling approximation of a permutation version of the classic Shapley value equations.

Kernel SHAP (Linear LIME + Shapley values)
The Kernel SHAP method requires fewer evaluations of the original model to obtain similar approximation accuracy. The LIME choices for loss function, weighting kernel, and regularization term are made heuristically. Since LIME uses a simplified input mapping that is equivalent to the approximation of SHAP mapping which can be computed using weighted linear regression, this enables regression-based, model-agnostic estimation of SHAP values, which is more efficient than classical Shapley equations.

The intuitive connection between linear regression and Shapley values is that Equation 8 is a difference of means. Since the mean is also the best least squares point estimate for a set of data points, it is natural to search for a weighting kernel that causes linear least squares regression to recapitulate the Shapley values. This leads to a kernel that distinctly differs from previous heuristically chosen kernels. [2]

Model-Specific Approximations [2]
We can develop faster model-specific approximation methods by restricting our attention to specific model types.

Linear SHAP
For linear models, if we assume input feature independence, SHAP values can be approximated directly from the model’s weight coefficients.

Low-Order SHAP
For linear models, if we assume input feature independence, SHAP values can be approximated directly from the model’s weight coefficients.

Max SHAP
Using a permutation formulation of Shapley values, we can calculate the probability that each input will increase the maximum value over every other input. Doing this on a sorted order of input values lets us compute the Shapley values of a max function with M inputs in O(M2) time instead of O(M2M).

Deep SHAP (DeepLIFT + Shapley values)
DeepLIFT approximates SHAP values assuming that the input features are independent of one another and the deep model is linear ... uses a linear composition rule, which is equivalent to linearizing the non-linear components of a neural network. Its back-propagation rules defining how each component is linearized are intuitive but were heuristically chosen ... Shapley values represent the only attribution values that satisfy consistency ... [and] become a compositional approximation of SHAP values.

By recursively passing DeepLIFT’s multipliers, Deep SHAP combines SHAP values computed for smaller components of the network into SHAP values for the whole network. This composition rule enables a fast approximation of values for the whole model.

Computational and User Study Experiments
The benefits of SHAP values using the Kernel SHAP and Deep SHAP approximation methods were compared. Overall, SHAP values prove more consistent with human intuition than other methods that fail to meet Properties 1-3.

Comptutational Efficiency
Comparing Shapley sampling, SHAP, and LIME on both dense and sparse decision tree models illustrates both the improved sample efficiency of Kernel SHAP and that values from LIME can differ significantly from SHAP values that satisfy local accuracy and consistency. [2]

Consistency with Human Intuition
A much stronger agreement between human explanations and SHAP than with other methods was found through testing [2].

Conclusions
The development of methods that help users interpret predictions has been spurred by the growing need for both accuracy and interpretability in deep learning model predictions. The SHAP framework identifies the class of additive feature importance methods. It also shows there is a unique solution in this class that adheres to desirable properties. SHAP is overall flexible and in conjunction with the additive feature importance methods can aide in the proper and effective explanation of deep learning classifiers.

Citations
https://pantelis.github.io/cs677/docs/common/projects/pneumonia/
https://arxiv.org/pdf/1705.07874.pdf
https://arxiv.org/pdf/1602.04938.pdf
https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf
https://github.com/aildnont/covid-cxr
https://github.com/slundberg/shap
https://github.com/ieee8023/covid-chestxray-dataset
https://towardsdatascience.com/investigation-of-explainable-predictions-of-covid-19-infection-from-chest-x-rays-with-machine-cb370f46af1d
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why should i trust you?: Explaining the predictions of any classifier”. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM. 2016, pp. 1135–1144.
